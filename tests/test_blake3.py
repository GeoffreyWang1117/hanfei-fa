"""Tests for BLAKE3 hash algorithm support."""

import pytest

from merkle_verify.hasher import (
    compute_hash,
    hash_pair,
    verify_hash,
    HashAlgorithm,
    set_default_algorithm,
    get_default_algorithm,
)
from merkle_verify.merkle_tree import MerkleTree, build_model_merkle_tree

blake3 = pytest.importorskip("blake3", reason="blake3 package not installed")


class TestBlake3Hash:
    def test_compute_hash_explicit(self):
        h = compute_hash(b"hello", algorithm=HashAlgorithm.BLAKE3)
        assert isinstance(h, str)
        assert len(h) == 64  # BLAKE3 default output is 32 bytes = 64 hex chars

    def test_deterministic(self):
        h1 = compute_hash(b"test data", algorithm=HashAlgorithm.BLAKE3)
        h2 = compute_hash(b"test data", algorithm=HashAlgorithm.BLAKE3)
        assert h1 == h2

    def test_different_data_different_hash(self):
        h1 = compute_hash(b"a", algorithm=HashAlgorithm.BLAKE3)
        h2 = compute_hash(b"b", algorithm=HashAlgorithm.BLAKE3)
        assert h1 != h2

    def test_differs_from_sha256(self):
        h_blake3 = compute_hash(b"hello", algorithm=HashAlgorithm.BLAKE3)
        h_sha256 = compute_hash(b"hello", algorithm=HashAlgorithm.SHA256)
        assert h_blake3 != h_sha256

    def test_hash_pair(self):
        a = compute_hash(b"left", algorithm=HashAlgorithm.BLAKE3)
        b = compute_hash(b"right", algorithm=HashAlgorithm.BLAKE3)
        p = hash_pair(a, b, algorithm=HashAlgorithm.BLAKE3)
        assert isinstance(p, str) and len(p) == 64

    def test_hash_pair_order_matters(self):
        a = compute_hash(b"left", algorithm=HashAlgorithm.BLAKE3)
        b = compute_hash(b"right", algorithm=HashAlgorithm.BLAKE3)
        assert hash_pair(a, b, algorithm=HashAlgorithm.BLAKE3) != hash_pair(b, a, algorithm=HashAlgorithm.BLAKE3)

    def test_verify_hash(self):
        data = b"payload"
        h = compute_hash(data, algorithm=HashAlgorithm.BLAKE3)
        assert verify_hash(data, h, algorithm=HashAlgorithm.BLAKE3) is True
        assert verify_hash(b"wrong", h, algorithm=HashAlgorithm.BLAKE3) is False

    def test_set_as_default(self):
        set_default_algorithm(HashAlgorithm.BLAKE3)
        assert get_default_algorithm() == HashAlgorithm.BLAKE3
        h = compute_hash(b"test")
        assert len(h) == 64

    def test_algorithm_by_string(self):
        h1 = compute_hash(b"test", algorithm="blake3")
        h2 = compute_hash(b"test", algorithm=HashAlgorithm.BLAKE3)
        assert h1 == h2

    def test_hash_length_property(self):
        assert HashAlgorithm.BLAKE3.hash_length == 64


class TestBlake3MerkleTree:
    def test_build_tree(self):
        set_default_algorithm(HashAlgorithm.BLAKE3)
        hashes = [compute_hash(f"chunk{i}".encode()) for i in range(4)]
        tree = MerkleTree(hashes, chunk_size=16)
        assert tree.root_hash is not None
        assert len(tree.root_hash) == 64

    def test_tree_diff(self):
        set_default_algorithm(HashAlgorithm.BLAKE3)
        h1 = [compute_hash(f"chunk{i}".encode()) for i in range(4)]
        h2 = list(h1)
        h2[2] = compute_hash(b"changed")
        tree1 = MerkleTree(h1)
        tree2 = MerkleTree(h2)
        changed, _ = tree1.diff_tree(tree2)
        assert changed == [2]

    def test_proof_verify(self):
        set_default_algorithm(HashAlgorithm.BLAKE3)
        hashes = [compute_hash(f"chunk{i}".encode()) for i in range(4)]
        tree = MerkleTree(hashes, chunk_size=16)
        proof = tree.get_proof(1)
        assert proof.verify()

    def test_model_tree(self):
        set_default_algorithm(HashAlgorithm.BLAKE3)
        import os
        blobs = {
            "layer.0.weight": os.urandom(2048),
            "layer.0.bias": os.urandom(128),
            "layer.1.weight": os.urandom(2048),
        }
        tree = build_model_merkle_tree(blobs, model_name="blake3-test", parallel=False)
        assert tree.model_root is not None
        assert tree.verify()
