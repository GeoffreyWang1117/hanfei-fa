"""Benchmark tests with a real ResNet18 model.

These tests verify the full pipeline against actual PyTorch model weights
and include performance benchmarks using pytest-benchmark style timing.
"""

import copy
import time

import pytest
import torch
import torchvision.models as models

from merkle_verify.merkle_tree import build_model_merkle_tree, ModelMerkleTree
from merkle_verify.comparison import compare_model_trees, estimate_sync_savings


def _state_dict_to_blobs(state_dict):
    """Convert PyTorch state dict to {name: bytes} blobs."""
    return {k: v.cpu().numpy().tobytes() for k, v in state_dict.items()}


@pytest.fixture(scope="module")
def resnet18_state_dict():
    """Load ResNet18 weights (cached per module)."""
    model = models.resnet18(weights=None)
    return model.state_dict()


@pytest.fixture(scope="module")
def resnet18_blobs(resnet18_state_dict):
    return _state_dict_to_blobs(resnet18_state_dict)


@pytest.fixture(scope="module")
def resnet18_tree(resnet18_blobs):
    return build_model_merkle_tree(resnet18_blobs, model_name="resnet18")


# ── Correctness ──────────────────────────────────────────────────────────


class TestResNet18Correctness:
    def test_tree_builds(self, resnet18_tree):
        assert resnet18_tree.model_root is not None
        assert len(resnet18_tree.layer_trees) > 0

    def test_verify(self, resnet18_tree):
        assert resnet18_tree.verify() is True

    def test_param_count_reasonable(self, resnet18_tree):
        # ResNet18 has ~11.7M parameters
        total = resnet18_tree.total_params
        assert 5_000_000 < total < 50_000_000

    def test_all_layers_have_roots(self, resnet18_tree):
        for name, layer in resnet18_tree.layer_trees.items():
            assert layer.layer_root is not None, f"layer {name} has no root"

    def test_identical_comparison(self, resnet18_blobs):
        t1 = build_model_merkle_tree(resnet18_blobs, parallel=False)
        t2 = build_model_merkle_tree(resnet18_blobs, parallel=False)
        diff = compare_model_trees(t1, t2)
        assert diff["change_percentage"] == 0.0
        assert diff["changed_params"] == []

    def test_single_layer_finetuned(self, resnet18_state_dict):
        """Simulate fine-tuning only the final FC layer."""
        sd_original = resnet18_state_dict
        sd_finetuned = copy.deepcopy(sd_original)
        # Perturb only fc.weight
        sd_finetuned["fc.weight"] = torch.randn_like(sd_finetuned["fc.weight"])

        t1 = build_model_merkle_tree(_state_dict_to_blobs(sd_original), parallel=False)
        t2 = build_model_merkle_tree(_state_dict_to_blobs(sd_finetuned), parallel=False)

        diff = compare_model_trees(t1, t2, detailed=True)
        assert "fc.weight" in diff["changed_params"]
        # Most params should be unchanged
        assert len(diff["unchanged_params"]) > len(diff["changed_params"])

    def test_sync_savings_finetuned(self, resnet18_state_dict):
        sd = copy.deepcopy(resnet18_state_dict)
        sd["fc.weight"] = torch.randn_like(sd["fc.weight"])

        t_orig = build_model_merkle_tree(_state_dict_to_blobs(resnet18_state_dict), parallel=False)
        t_fine = build_model_merkle_tree(_state_dict_to_blobs(sd), parallel=False)

        savings = estimate_sync_savings(t_orig, t_fine)
        # Only fc.weight changed → high savings
        assert savings["savings_percentage"] > 80

    def test_proof_for_resnet_chunks(self, resnet18_tree):
        """Verify Merkle proofs for a few chunks in a ResNet layer."""
        for layer_tree in resnet18_tree.layer_trees.values():
            for param_tree in layer_tree.param_trees.values():
                if param_tree.num_chunks >= 2:
                    proof = param_tree.get_proof(0)
                    assert proof.verify()
                    return  # one successful proof is enough
        pytest.fail("No param with >=2 chunks found")

    def test_json_roundtrip(self, resnet18_tree, tmp_path):
        path = str(tmp_path / "resnet18.json")
        resnet18_tree.to_json(path)
        restored = ModelMerkleTree.from_json(path)
        assert restored.model_root == resnet18_tree.model_root
        assert restored.verify() is True

    def test_parallel_vs_serial(self, resnet18_blobs):
        t_serial = build_model_merkle_tree(resnet18_blobs, parallel=False)
        t_parallel = build_model_merkle_tree(resnet18_blobs, parallel=True)
        assert t_serial.model_root == t_parallel.model_root


# ── Benchmarks ───────────────────────────────────────────────────────────


class TestResNet18Benchmarks:
    def test_build_time(self, resnet18_blobs):
        """Building a ResNet18 Merkle tree should complete within 5 seconds."""
        start = time.perf_counter()
        build_model_merkle_tree(resnet18_blobs, parallel=True)
        elapsed = time.perf_counter() - start
        print(f"\n  Build time (parallel): {elapsed:.3f}s")
        assert elapsed < 5.0

    def test_build_time_serial(self, resnet18_blobs):
        start = time.perf_counter()
        build_model_merkle_tree(resnet18_blobs, parallel=False)
        elapsed = time.perf_counter() - start
        print(f"\n  Build time (serial):   {elapsed:.3f}s")
        assert elapsed < 10.0

    def test_compare_time(self, resnet18_blobs, resnet18_state_dict):
        """Comparison with tree-walk diff should be fast."""
        sd2 = copy.deepcopy(resnet18_state_dict)
        sd2["fc.weight"] = torch.randn_like(sd2["fc.weight"])
        t1 = build_model_merkle_tree(resnet18_blobs, parallel=False)
        t2 = build_model_merkle_tree(_state_dict_to_blobs(sd2), parallel=False)

        start = time.perf_counter()
        diff = compare_model_trees(t1, t2, detailed=True)
        elapsed = time.perf_counter() - start
        print(f"\n  Compare time:          {elapsed:.4f}s  ({diff['hash_comparisons']} hash comparisons)")
        assert elapsed < 2.0

    def test_verify_time(self, resnet18_tree):
        start = time.perf_counter()
        resnet18_tree.verify()
        elapsed = time.perf_counter() - start
        print(f"\n  Verify time:           {elapsed:.4f}s")
        assert elapsed < 2.0

    def test_hash_comparisons_sublinear(self, resnet18_blobs, resnet18_state_dict):
        """With 1 param changed, hash comparisons should be << total chunks."""
        sd2 = copy.deepcopy(resnet18_state_dict)
        sd2["fc.weight"] = torch.randn_like(sd2["fc.weight"])
        t1 = build_model_merkle_tree(resnet18_blobs, parallel=False)
        t2 = build_model_merkle_tree(_state_dict_to_blobs(sd2), parallel=False)

        diff = compare_model_trees(t1, t2, detailed=True)

        # Count total chunks across all params
        total_chunks = sum(
            pt.num_chunks
            for lt in t1.layer_trees.values()
            for pt in lt.param_trees.values()
        )
        print(f"\n  Hash comparisons: {diff['hash_comparisons']} vs total chunks: {total_chunks}")
        # Should be much less than total chunks due to layer + param + tree pruning
        assert diff["hash_comparisons"] < total_chunks
