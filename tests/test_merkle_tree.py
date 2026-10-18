"""Tests for merkle_verify.merkle_tree module — MerkleTree, LayerMerkleTree, ModelMerkleTree."""

import json
import os
import tempfile

import pytest

from merkle_verify.hasher import compute_hash, hash_pair
from merkle_verify.chunking import chunk_bytes
from merkle_verify.merkle_tree import (
    MerkleTree,
    MerkleNode,
    MerkleProof,
    LayerMerkleTree,
    ModelMerkleTree,
    build_model_merkle_tree,
    _extract_layer_name,
)


# ── MerkleTree ──────────────────────────────────────────────────────────


class TestMerkleTreeConstruction:
    def test_single_chunk(self):
        h = [compute_hash(b"only")]
        tree = MerkleTree(h)
        assert tree.root_hash == h[0]
        assert tree.num_chunks == 1

    def test_two_chunks(self):
        h = [compute_hash(b"a"), compute_hash(b"b")]
        tree = MerkleTree(h, version=1)
        assert tree.root_hash == hash_pair(h[0], h[1])

    def test_four_chunks(self):
        h = [compute_hash(f"c{i}".encode()) for i in range(4)]
        tree = MerkleTree(h, version=1)
        left = hash_pair(h[0], h[1])
        right = hash_pair(h[2], h[3])
        assert tree.root_hash == hash_pair(left, right)

    def test_odd_chunks_duplication(self):
        """v1: with an odd number of chunks the last node is duplicated."""
        h = [compute_hash(f"c{i}".encode()) for i in range(3)]
        tree = MerkleTree(h, version=1)
        left = hash_pair(h[0], h[1])
        right = hash_pair(h[2], h[2])  # duplicated
        assert tree.root_hash == hash_pair(left, right)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            MerkleTree([])

    def test_deterministic(self):
        h = [compute_hash(b"x")] * 5
        t1 = MerkleTree(h)
        t2 = MerkleTree(h)
        assert t1.root_hash == t2.root_hash

    def test_chunk_size_stored(self):
        h = [compute_hash(b"a")]
        tree = MerkleTree(h, chunk_size=4096)
        assert tree.chunk_size == 4096


class TestMerkleProof:
    def test_proof_verifies(self, simple_tree):
        for i in range(simple_tree.num_chunks):
            proof = simple_tree.get_proof(i)
            assert proof.verify()

    def test_proof_root_matches(self, simple_tree):
        proof = simple_tree.get_proof(0)
        assert proof.root_hash == simple_tree.root_hash

    def test_proof_out_of_range(self, simple_tree):
        with pytest.raises(ValueError):
            simple_tree.get_proof(simple_tree.num_chunks)

    def test_proof_single_chunk(self):
        h = [compute_hash(b"only")]
        tree = MerkleTree(h)
        proof = tree.get_proof(0)
        assert proof.verify()
        assert proof.sibling_hashes == []

    def test_proof_odd_chunks(self):
        """Proofs work for ALL leaves in odd-count trees, including the last."""
        for num_chunks in [3, 5, 7, 9, 11]:
            h = [compute_hash(f"c{i}".encode()) for i in range(num_chunks)]
            tree = MerkleTree(h)
            for i in range(num_chunks):
                proof = tree.get_proof(i)
                assert proof.verify(), (
                    f"Proof failed for chunk {i} in {num_chunks}-chunk tree"
                )

    def test_tampered_proof_fails(self, simple_tree):
        proof = simple_tree.get_proof(0)
        proof.chunk_hash = compute_hash(b"tampered")
        assert not proof.verify()


class TestMerkleTreeDiff:
    def test_identical_trees(self):
        h = [compute_hash(f"c{i}".encode()) for i in range(8)]
        t1 = MerkleTree(h)
        t2 = MerkleTree(h[:])
        changed, comparisons = t1.diff_tree(t2)
        assert changed == []
        assert comparisons == 1  # only root compared

    def test_single_change(self):
        h1 = [compute_hash(f"c{i}".encode()) for i in range(8)]
        h2 = h1[:]
        h2[3] = compute_hash(b"modified")
        t1, t2 = MerkleTree(h1), MerkleTree(h2)
        changed, _ = t1.diff_tree(t2)
        assert changed == [3]

    def test_all_changed(self):
        h1 = [compute_hash(f"a{i}".encode()) for i in range(4)]
        h2 = [compute_hash(f"b{i}".encode()) for i in range(4)]
        t1, t2 = MerkleTree(h1), MerkleTree(h2)
        changed, _ = t1.diff_tree(t2)
        assert changed == [0, 1, 2, 3]

    def test_different_sizes(self):
        h1 = [compute_hash(f"c{i}".encode()) for i in range(4)]
        h2 = [compute_hash(f"c{i}".encode()) for i in range(6)]
        t1, t2 = MerkleTree(h1), MerkleTree(h2)
        changed, _ = t1.diff_tree(t2)
        assert 4 in changed and 5 in changed

    def test_odd_chunk_count_diff(self):
        h1 = [compute_hash(f"c{i}".encode()) for i in range(3)]
        h2 = h1[:]
        h2[2] = compute_hash(b"changed")
        t1, t2 = MerkleTree(h1), MerkleTree(h2)
        changed, _ = t1.diff_tree(t2)
        assert 2 in changed

    def test_comparisons_sublinear(self):
        """With 1 change in 64 chunks, comparisons should be << 64."""
        h1 = [compute_hash(f"c{i}".encode()) for i in range(64)]
        h2 = h1[:]
        h2[30] = compute_hash(b"modified")
        t1, t2 = MerkleTree(h1), MerkleTree(h2)
        _, comparisons = t1.diff_tree(t2)
        assert comparisons < 20  # O(log n) path


class TestMerkleTreeSerialisation:
    def test_to_dict_roundtrip(self, simple_tree):
        d = simple_tree.to_dict()
        restored = MerkleTree.from_dict(d)
        assert restored.root_hash == simple_tree.root_hash
        assert restored.chunk_hashes == simple_tree.chunk_hashes
        assert restored.chunk_size == simple_tree.chunk_size

    def test_to_dict_keys_v1(self):
        tree = MerkleTree([compute_hash(b"a"), compute_hash(b"b")], chunk_size=16, version=1)
        assert set(tree.to_dict()) == {"root_hash", "num_chunks", "chunk_size", "chunk_hashes"}

    def test_to_dict_keys_v2(self, simple_tree):
        d = simple_tree.to_dict()
        assert set(d.keys()) == {"format", "format_version", "algorithm", "root_hash", "num_bytes",
                                 "num_chunks", "chunk_size", "dtype", "shape", "chunk_hashes"}
        assert d["format_version"] == 2


# ── LayerMerkleTree ─────────────────────────────────────────────────────


class TestLayerMerkleTree:
    def _make_layer(self):
        layer = LayerMerkleTree(layer_name="layer.0")
        for name in ["weight", "bias"]:
            data = os.urandom(512)
            layer.param_trees[name] = MerkleTree.from_bytes(data, chunk_size=128)
        return layer

    def test_compute_layer_root(self):
        layer = self._make_layer()
        root = layer.compute_layer_root()
        assert isinstance(root, str) and len(root) == 64
        assert layer.layer_root == root

    def test_deterministic_ordering(self):
        """Layer root must be the same regardless of insertion order."""
        data_w = os.urandom(512)
        data_b = os.urandom(256)

        def make_layer(order):
            layer = LayerMerkleTree(layer_name="l")
            mapping = {"weight": data_w, "bias": data_b}
            for name in order:
                layer.param_trees[name] = MerkleTree.from_bytes(mapping[name], 128)
            layer.compute_layer_root()
            return layer

        l1 = make_layer(["weight", "bias"])
        l2 = make_layer(["bias", "weight"])
        assert l1.layer_root == l2.layer_root

    def test_empty_raises(self):
        layer = LayerMerkleTree(layer_name="empty")
        with pytest.raises(ValueError, match="No parameter"):
            layer.compute_layer_root()

    def test_serialisation_roundtrip(self):
        layer = self._make_layer()
        layer.compute_layer_root()
        d = layer.to_dict()
        restored = LayerMerkleTree.from_dict(d)
        assert restored.layer_root == layer.layer_root
        assert restored.layer_name == layer.layer_name
        assert set(restored.param_trees.keys()) == set(layer.param_trees.keys())


# ── ModelMerkleTree ──────────────────────────────────────────────────────


class TestModelMerkleTree:
    def test_compute_model_root(self, model_tree):
        assert model_tree.model_root is not None
        assert len(model_tree.model_root) == 64

    def test_verify_true(self, model_tree):
        assert model_tree.verify() is True

    def test_verify_false_after_tamper(self, model_tree):
        model_tree.model_root = "0" * 64
        assert model_tree.verify() is False

    def test_verify_empty(self):
        m = ModelMerkleTree()
        assert m.verify() is False

    def test_empty_model_root_raises(self):
        m = ModelMerkleTree()
        with pytest.raises(ValueError, match="No layer"):
            m.compute_model_root()

    def test_total_params(self, model_tree):
        assert model_tree.total_params > 0

    def test_get_changed_layers_identical(self, sample_blobs):
        t1 = build_model_merkle_tree(sample_blobs, parallel=False)
        t2 = build_model_merkle_tree(sample_blobs, parallel=False)
        assert t1.get_changed_layers(t2) == []

    def test_get_changed_layers_different(self, sample_blobs):
        t1 = build_model_merkle_tree(sample_blobs, parallel=False)
        blobs2 = dict(sample_blobs)
        blobs2["layer.0.weight"] = os.urandom(4096)
        t2 = build_model_merkle_tree(blobs2, parallel=False)
        changed = t1.get_changed_layers(t2)
        assert len(changed) >= 1

    def test_get_changed_chunks(self, sample_blobs):
        t1 = build_model_merkle_tree(sample_blobs, parallel=False)
        blobs2 = dict(sample_blobs)
        blobs2["layer.0.weight"] = os.urandom(4096)
        t2 = build_model_merkle_tree(blobs2, parallel=False)
        # Find a common layer
        common_layers = set(t1.layer_trees) & set(t2.layer_trees)
        for layer_name in common_layers:
            chunks = t1.get_changed_chunks(t2, layer_name)
            assert isinstance(chunks, dict)

    def test_get_changed_chunks_missing_layer(self, model_tree):
        other = ModelMerkleTree(model_name="empty")
        with pytest.raises(ValueError, match="not in both"):
            model_tree.get_changed_chunks(other, "nonexistent")

    def test_json_roundtrip(self, model_tree):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            model_tree.to_json(path)
            restored = ModelMerkleTree.from_json(path)
            assert restored.model_root == model_tree.model_root
            assert restored.model_name == model_tree.model_name
            assert set(restored.layer_trees.keys()) == set(model_tree.layer_trees.keys())
        finally:
            os.unlink(path)

    def test_dict_roundtrip(self, model_tree):
        d = model_tree.to_dict()
        restored = ModelMerkleTree.from_dict(d)
        assert restored.model_root == model_tree.model_root

    def test_deterministic_ordering(self, sample_blobs):
        """Model root must be the same regardless of dict ordering."""
        t1 = build_model_merkle_tree(sample_blobs, parallel=False)
        reversed_blobs = dict(reversed(list(sample_blobs.items())))
        t2 = build_model_merkle_tree(reversed_blobs, parallel=False)
        assert t1.model_root == t2.model_root


# ── build_model_merkle_tree ──────────────────────────────────────────────


class TestBuildModelMerkleTree:
    def test_basic(self, sample_blobs):
        tree = build_model_merkle_tree(sample_blobs, parallel=False)
        assert tree.model_root is not None
        assert len(tree.layer_trees) > 0

    def test_parallel_matches_serial(self, sample_blobs):
        # Parallel only kicks in with >4 blobs
        big_blobs = {f"layer.{i}.weight": os.urandom(4096) for i in range(8)}
        t_serial = build_model_merkle_tree(big_blobs, parallel=False)
        t_parallel = build_model_merkle_tree(big_blobs, parallel=True, num_workers=2)
        assert t_serial.model_root == t_parallel.model_root

    def test_custom_chunk_size(self, sample_blobs):
        t1 = build_model_merkle_tree(sample_blobs, chunk_size=256, parallel=False)
        t2 = build_model_merkle_tree(sample_blobs, chunk_size=512, parallel=False)
        # Different chunk sizes → different trees
        assert t1.model_root != t2.model_root

    def test_model_name(self, sample_blobs):
        tree = build_model_merkle_tree(sample_blobs, model_name="my-model", parallel=False)
        assert tree.model_name == "my-model"


# ── _extract_layer_name ─────────────────────────────────────────────────


class TestExtractLayerName:
    @pytest.mark.parametrize("param,expected", [
        ("transformer.h.0.attn.weight", "transformer.h.0"),
        ("model.layers.3.mlp.gate_proj.weight", "model.layers.3"),
        ("encoder.layer.11.attention.weight", "encoder.layer.11"),
        ("fc.weight", "fc"),
        ("embeddings.word_embeddings.weight", "embeddings"),
    ])
    def test_known_patterns(self, param, expected):
        assert _extract_layer_name(param) == expected

    def test_generic_numbered(self):
        result = _extract_layer_name("block.5.conv.weight")
        assert result == "block.5"
