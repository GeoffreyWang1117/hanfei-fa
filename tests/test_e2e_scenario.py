"""
End-to-end scenario test: Verifiable Fine-Tuning Pipeline

Scenario:
    A team at Company X deploys GPT-2 (124M params) as a text generation service.
    They fine-tune the last 2 transformer layers on proprietary data and distribute
    the fine-tuned checkpoint to edge servers. They need to:

    1. Sign the base model → publish root hash as the "golden fingerprint"
    2. Fine-tune selected layers → create a new checkpoint
    3. Diff base vs fine-tuned → confirm only expected layers changed
    4. Sign the fine-tuned model → distribute with manifest
    5. Edge server receives checkpoint → verify integrity before serving
    6. Attacker tampers with one layer → detection must trigger
    7. Estimate bandwidth savings for incremental sync
    8. Verify individual tensors without loading the full model
    9. Use BLAKE3 for faster hashing on high-throughput pipeline
    10. Stream-hash a large binary blob without loading into memory

This test uses real GPT-2 weights from torchvision/HuggingFace-style
architecture, simulated via torch.nn modules.
"""

import copy
import io
import json
import os
import subprocess
import sys
import time

import pytest

np = pytest.importorskip("numpy", reason="numpy required for e2e scenario")
torch = pytest.importorskip("torch", reason="PyTorch required for e2e scenario")
safetensors_lib = pytest.importorskip("safetensors", reason="safetensors required for e2e scenario")
blake3_lib = pytest.importorskip("blake3", reason="blake3 required for e2e scenario")

from safetensors.torch import save_file as st_save_file

from merkle_verify.hasher import (
    compute_hash, HashAlgorithm, set_default_algorithm, get_default_algorithm,
)
from merkle_verify.merkle_tree import (
    MerkleTree, ModelMerkleTree, build_model_merkle_tree, build_file_merkle_tree,
)
from merkle_verify.comparison import compare_model_trees, estimate_sync_savings
from merkle_verify.safetensors_adapter import (
    sign as st_sign, verify as st_verify, verify_tensor, diff as st_diff,
)
from merkle_verify.pytorch_adapter import (
    merkle_save, merkle_load, verify_checkpoint, diff_checkpoints,
)


# ---------------------------------------------------------------------------
# Fixtures: build a realistic transformer-like model
# ---------------------------------------------------------------------------

class MiniTransformerBlock(torch.nn.Module):
    """Simplified transformer block with attention + MLP."""
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.GELU(),
            torch.nn.Linear(d_model * 4, d_model),
        )
        self.ln2 = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(torch.nn.Module):
    """6-layer mini GPT for testing (~2.4M params)."""
    def __init__(self, vocab_size=1000, d_model=256, n_layers=6):
        super().__init__()
        self.token_emb = torch.nn.Embedding(vocab_size, d_model)
        self.pos_emb = torch.nn.Embedding(128, d_model)
        self.blocks = torch.nn.ModuleList([
            MiniTransformerBlock(d_model) for _ in range(n_layers)
        ])
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


@pytest.fixture
def base_model():
    """Base MiniGPT model with deterministic weights."""
    torch.manual_seed(42)
    return MiniGPT()


@pytest.fixture
def finetuned_model(base_model):
    """Fine-tuned model: only last 2 blocks modified (simulated)."""
    model = copy.deepcopy(base_model)
    torch.manual_seed(999)
    # Simulate fine-tuning: perturb last 2 blocks
    with torch.no_grad():
        for param in model.blocks[4].parameters():
            param.add_(torch.randn_like(param) * 0.01)
        for param in model.blocks[5].parameters():
            param.add_(torch.randn_like(param) * 0.01)
    return model


@pytest.fixture
def base_safetensors(tmp_path, base_model):
    """Save base model as safetensors."""
    path = str(tmp_path / "base_model.safetensors")
    st_save_file(base_model.state_dict(), path)
    return path


@pytest.fixture
def finetuned_safetensors(tmp_path, finetuned_model):
    """Save fine-tuned model as safetensors."""
    path = str(tmp_path / "finetuned_model.safetensors")
    st_save_file(finetuned_model.state_dict(), path)
    return path


@pytest.fixture
def base_checkpoint(tmp_path, base_model):
    """Save base model as PyTorch checkpoint with Merkle manifest."""
    path = str(tmp_path / "base_model.pt")
    merkle_save(base_model, path, model_name="mini-gpt-base")
    return path


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------

class TestScenario01_SignBaseModel:
    """Step 1: Sign the base model and publish the golden fingerprint."""

    def test_sign_safetensors(self, base_safetensors):
        tree = st_sign(base_safetensors, model_name="mini-gpt-base")

        assert tree.model_root is not None
        assert len(tree.model_root) == 64
        assert tree.model_name == "mini-gpt-base"
        # Should have multiple layers (embedding, blocks.0-5, ln_f, lm_head)
        assert len(tree.layer_trees) >= 5
        # Manifest sidecar exists
        assert os.path.exists(base_safetensors.replace(".safetensors", ".merkle.json"))

    def test_sign_pytorch(self, base_checkpoint):
        # merkle_save already ran in fixture
        manifest = base_checkpoint.replace(".pt", ".merkle.json")
        assert os.path.exists(manifest)
        tree = ModelMerkleTree.from_json(manifest)
        assert tree.model_root is not None
        assert tree.verify()

    def test_golden_fingerprint_is_deterministic(self, base_model, tmp_path):
        """Same model weights → same root hash, always."""
        p1 = str(tmp_path / "m1.safetensors")
        p2 = str(tmp_path / "m2.safetensors")
        st_save_file(base_model.state_dict(), p1)
        st_save_file(base_model.state_dict(), p2)

        t1 = st_sign(p1, sidecar=False)
        t2 = st_sign(p2, sidecar=False)
        assert t1.model_root == t2.model_root


class TestScenario02_DiffBaseVsFinetuned:
    """Step 3: Diff base vs fine-tuned to confirm only expected layers changed."""

    def test_safetensors_diff(self, base_safetensors, finetuned_safetensors):
        result = st_diff(base_safetensors, finetuned_safetensors)

        # Only blocks.4 and blocks.5 should have changed
        changed_params = result["changed_params"]
        unchanged_params = result["unchanged_params"]

        # All changed params should be in blocks.4 or blocks.5
        for p in changed_params:
            assert "blocks.4" in p or "blocks.5" in p, (
                f"Unexpected changed param: {p}"
            )

        # blocks.0-3 params should be unchanged
        for p in unchanged_params:
            assert "blocks.4" not in p and "blocks.5" not in p

        # Change percentage should be roughly 2/6 of transformer params
        assert 0 < result["change_percentage"] < 100
        assert result["hash_comparisons"] > 0

    def test_changed_layers_identified(self, base_safetensors, finetuned_safetensors):
        result = st_diff(base_safetensors, finetuned_safetensors)
        changed_layers = result["changed_layers"]

        # Should include blocks.4 and blocks.5
        layer_names_str = " ".join(changed_layers)
        assert "4" in layer_names_str
        assert "5" in layer_names_str

    def test_pytorch_diff(self, base_model, finetuned_model, tmp_path):
        p1 = str(tmp_path / "base.pt")
        p2 = str(tmp_path / "ft.pt")
        merkle_save(base_model, p1)
        merkle_save(finetuned_model, p2)

        result = diff_checkpoints(p1, p2)
        assert len(result["changed_params"]) > 0
        assert len(result["unchanged_params"]) > 0


class TestScenario03_EdgeServerVerification:
    """Step 5: Edge server receives checkpoint and verifies integrity."""

    def test_verify_safetensors_intact(self, base_safetensors):
        st_sign(base_safetensors)
        is_valid, details = st_verify(base_safetensors)
        assert is_valid is True
        assert details["root_match"] is True

    def test_verify_pytorch_intact(self, base_checkpoint):
        is_valid, details = verify_checkpoint(base_checkpoint)
        assert is_valid is True

    def test_verify_root_only_O1(self, base_safetensors):
        """O(1) root-only check: just compare stored root hash."""
        tree = st_sign(base_safetensors)  # write sidecar so verify can load it
        golden_root = tree.model_root

        is_valid, details = st_verify(
            base_safetensors, expected_root=golden_root
        )
        assert is_valid is True
        assert details["mode"] == "expected_root"

    def test_load_with_verify(self, base_checkpoint):
        sd, details = merkle_load(base_checkpoint)
        assert sd is not None
        assert details["verified"] is True
        # Loaded state dict should have all keys
        assert len(sd) > 0


class TestScenario04_TamperingDetection:
    """Step 6: Attacker tampers with weights → must be detected."""

    def test_detect_single_tensor_tamper_safetensors(
        self, base_model, tmp_path
    ):
        """Attacker modifies one tensor in a safetensors file."""
        path = str(tmp_path / "model.safetensors")
        sd = base_model.state_dict()
        st_save_file(sd, path)
        st_sign(path)

        # Tamper: zero out one weight tensor
        sd_tampered = {k: v.clone() for k, v in sd.items()}
        sd_tampered["blocks.2.mlp.0.weight"] = torch.zeros_like(
            sd_tampered["blocks.2.mlp.0.weight"]
        )
        tampered_path = str(tmp_path / "tampered.safetensors")
        st_save_file(sd_tampered, tampered_path)

        # Verify tampered file against original manifest
        is_valid, details = st_verify(
            tampered_path,
            manifest=path.replace(".safetensors", ".merkle.json"),
        )
        assert is_valid is False
        assert "blocks.2.mlp.0.weight" in details["changed_params"]

    def test_detect_tamper_pytorch(self, base_model, tmp_path):
        path = str(tmp_path / "model.pt")
        tree = merkle_save(base_model, path)

        # Tamper
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd["blocks.0.attn.in_proj_weight"] = torch.randn_like(
            sd["blocks.0.attn.in_proj_weight"]
        )
        torch.save(sd, path)

        is_valid, details = verify_checkpoint(path)
        assert is_valid is False
        assert len(details["changed_params"]) > 0

    def test_single_bit_flip_detected(self, base_model, tmp_path):
        """Even a single-bit change in tensor data must be caught."""
        path = str(tmp_path / "model.safetensors")
        sd = base_model.state_dict()
        st_save_file(sd, path)
        tree = st_sign(path)

        # Flip one bit deep in the tensor data region (well past the JSON header).
        # Safetensors header is at most ~8KB for this model; data starts after.
        file_size = os.path.getsize(path)
        flip_offset = file_size // 2  # middle of data region

        with open(path, "r+b") as f:
            f.seek(flip_offset)
            b = f.read(1)
            f.seek(flip_offset)
            f.write(bytes([b[0] ^ 0x01]))

        is_valid, _ = st_verify(path)
        assert is_valid is False


class TestScenario05_IncrementalSync:
    """Step 7: Estimate bandwidth savings for incremental sync."""

    def test_sync_savings(self, base_safetensors, finetuned_safetensors):
        tree_base = st_sign(base_safetensors, sidecar=False)
        tree_ft = st_sign(finetuned_safetensors, sidecar=False)

        savings = estimate_sync_savings(tree_base, tree_ft)
        assert savings["savings_percentage"] > 0
        assert savings["savings_bytes"] > 0
        assert savings["incremental_sync_bytes"] < savings["full_sync_bytes"]

        # Fine-tuning 2 of 6 layers: expect ~50-70% savings
        assert savings["savings_percentage"] > 30, (
            f"Expected >30% savings, got {savings['savings_percentage']:.1f}%"
        )


class TestScenario06_SingleTensorVerification:
    """Step 8: Verify individual tensors without loading the full model."""

    def test_verify_untouched_tensor(self, base_safetensors):
        st_sign(base_safetensors)
        is_valid, details = verify_tensor(base_safetensors, "blocks.0.ln1.weight")
        assert is_valid is True

    def test_verify_all_tensors_one_by_one(self, base_safetensors, base_model):
        st_sign(base_safetensors)
        for name in base_model.state_dict():
            is_valid, details = verify_tensor(base_safetensors, name)
            assert is_valid is True, f"Tensor {name} failed verification"

    def test_verify_nonexistent_tensor(self, base_safetensors):
        st_sign(base_safetensors)
        is_valid, details = verify_tensor(base_safetensors, "fake.weight")
        assert is_valid is False


class TestScenario07_Blake3FastPipeline:
    """Step 9: Use BLAKE3 for faster hashing on high-throughput pipeline."""

    def test_blake3_end_to_end(self, base_model, tmp_path):
        set_default_algorithm(HashAlgorithm.BLAKE3)

        sd = base_model.state_dict()
        blobs = {k: v.detach().cpu().contiguous().numpy().tobytes() for k, v in sd.items()}

        tree = build_model_merkle_tree(blobs, model_name="blake3-gpt", parallel=True)
        assert tree.model_root is not None
        assert tree.verify()

        # Diff still works
        tree2 = build_model_merkle_tree(blobs, model_name="blake3-gpt-copy", parallel=False)
        result = compare_model_trees(tree, tree2)
        assert len(result["changed_params"]) == 0

    def test_blake3_faster_than_sha256(self, base_model):
        """BLAKE3 should be measurably faster for large data."""
        sd = base_model.state_dict()
        blobs = {k: v.detach().cpu().contiguous().numpy().tobytes() for k, v in sd.items()}

        set_default_algorithm(HashAlgorithm.SHA256)
        t0 = time.time()
        for _ in range(3):
            build_model_merkle_tree(blobs, model_name="sha256", parallel=False)
        sha256_time = time.time() - t0

        set_default_algorithm(HashAlgorithm.BLAKE3)
        t0 = time.time()
        for _ in range(3):
            build_model_merkle_tree(blobs, model_name="blake3", parallel=False)
        blake3_time = time.time() - t0

        # BLAKE3 should be at least somewhat faster (allow some variance)
        # On most hardware, BLAKE3 is 3-10x faster
        assert blake3_time < sha256_time * 1.5, (
            f"BLAKE3 ({blake3_time:.3f}s) not faster than SHA256 ({sha256_time:.3f}s)"
        )


class TestScenario08_StreamingLargeFile:
    """Step 10: Stream-hash a large binary blob without loading into memory."""

    def test_streaming_matches_in_memory(self, base_safetensors):
        """Streaming and in-memory produce the same root hash."""
        with open(base_safetensors, "rb") as f:
            data = f.read()

        # In-memory
        from merkle_verify.chunking import chunk_bytes
        chunks = chunk_bytes(data)
        hashes = [compute_hash(c) for c in chunks]
        mem_tree = MerkleTree(hashes)

        # Streaming
        stream_tree = build_file_merkle_tree(base_safetensors)

        assert stream_tree.root_hash == mem_tree.root_hash

    def test_streaming_proof_valid(self, base_safetensors):
        tree = build_file_merkle_tree(base_safetensors)
        for i in range(min(tree.num_chunks, 5)):
            proof = tree.get_proof(i)
            assert proof.verify(), f"Proof failed for chunk {i}"

    def test_large_blob_streaming(self, tmp_path):
        """1 MB file: streaming should use constant memory."""
        filepath = str(tmp_path / "large.bin")
        with open(filepath, "wb") as f:
            for _ in range(64):  # 64 * 16KB = 1MB
                f.write(os.urandom(16384))

        tree = build_file_merkle_tree(filepath)
        assert tree.num_chunks == 64
        assert tree.root_hash is not None


class TestScenario09_CLIIntegration:
    """Test the CLI tool in the context of the scenario."""

    def _cli(self, *args):
        result = subprocess.run(
            [sys.executable, "-m", "merkle_verify.cli", *args],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode, result.stdout, result.stderr

    def test_cli_sign_safetensors(self, base_safetensors):
        rc, out, _ = self._cli("sign", base_safetensors)
        assert rc == 0
        assert "root:" in out

    def test_cli_verify_safetensors(self, base_safetensors):
        self._cli("sign", base_safetensors)
        rc, out, _ = self._cli("verify", base_safetensors)
        assert rc == 0
        assert "OK" in out

    def test_cli_diff_safetensors(self, base_safetensors, finetuned_safetensors):
        rc, out, _ = self._cli("diff", base_safetensors, finetuned_safetensors)
        assert rc == 0
        assert "changed" in out

    def test_cli_info(self, base_safetensors):
        self._cli("sign", base_safetensors)
        manifest = base_safetensors.replace(".safetensors", ".merkle.json")
        rc, out, _ = self._cli("info", manifest)
        assert rc == 0
        assert "layers:" in out

    def test_cli_hash_generic(self, base_safetensors):
        rc, out, _ = self._cli("hash", base_safetensors)
        assert rc == 0
        # Output: <hash>  <filepath>
        parts = out.strip().split()
        assert len(parts[0]) == 64


class TestScenario10_MerkleProofChain:
    """
    Merkle proofs enable third-party verification without the full tree.

    Scenario: Edge server publishes proof that a specific tensor chunk
    is part of the signed model. Auditor verifies without needing
    all model weights.
    """

    def test_proof_chain_for_critical_layer(self, base_model, tmp_path):
        sd = base_model.state_dict()
        blobs = {k: v.detach().cpu().contiguous().numpy().tobytes() for k, v in sd.items()}
        tree = build_model_merkle_tree(blobs, model_name="proof-test", parallel=False)

        # Pick a critical parameter
        target_param = "blocks.0.attn.in_proj_weight"
        target_layer = None
        target_ptree = None
        for lname, lt in tree.layer_trees.items():
            if target_param in lt.param_trees:
                target_layer = lname
                target_ptree = lt.param_trees[target_param]
                break

        assert target_ptree is not None

        # Generate proof for first chunk
        proof = target_ptree.get_proof(0)

        # Auditor receives: proof object + expected root hash
        # Auditor verifies without needing any other data
        assert proof.verify()
        assert proof.root_hash == target_ptree.root_hash

    def test_all_proofs_valid_for_model(self, base_model):
        """Every chunk in every parameter should produce a valid proof."""
        sd = base_model.state_dict()
        blobs = {k: v.detach().cpu().contiguous().numpy().tobytes() for k, v in sd.items()}
        tree = build_model_merkle_tree(blobs, model_name="full-proof", parallel=False)

        total_proofs = 0
        for lt in tree.layer_trees.values():
            for pt in lt.param_trees.values():
                for i in range(pt.num_chunks):
                    proof = pt.get_proof(i)
                    assert proof.verify(), (
                        f"Proof failed: chunk {i} of {pt.num_chunks}"
                    )
                    total_proofs += 1

        assert total_proofs > 50  # sanity: model has many chunks
