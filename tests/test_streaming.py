"""Tests for streaming Merkle tree construction."""

import io
import os
import pytest

from merkle_verify.hasher import compute_hash
from merkle_verify.chunking import chunk_bytes
from merkle_verify.merkle_tree import MerkleTree, build_file_merkle_tree


class TestMerkleTreeFromFile:
    def test_matches_in_memory(self):
        """Streaming build produces same root as in-memory build."""
        data = os.urandom(100_000)
        chunks = chunk_bytes(data, 16384)
        hashes = [compute_hash(c) for c in chunks]
        in_memory = MerkleTree(hashes, chunk_size=16384)

        stream = io.BytesIO(data)
        streamed = MerkleTree.from_file(stream, chunk_size=16384)

        assert streamed.root_hash == in_memory.root_hash
        assert streamed.num_chunks == in_memory.num_chunks

    def test_small_file(self):
        """Works for files smaller than one chunk."""
        data = b"tiny"
        stream = io.BytesIO(data)
        tree = MerkleTree.from_file(stream, chunk_size=16384)
        assert tree.num_chunks == 1
        assert tree.root_hash == compute_hash(data)

    def test_exact_chunk_boundary(self):
        """Works when file size is exact multiple of chunk size."""
        data = os.urandom(16384 * 4)
        stream = io.BytesIO(data)
        tree = MerkleTree.from_file(stream, chunk_size=16384)
        assert tree.num_chunks == 4

    def test_empty_file_raises(self):
        stream = io.BytesIO(b"")
        with pytest.raises(ValueError, match="empty"):
            MerkleTree.from_file(stream)

    def test_custom_chunk_size(self):
        data = os.urandom(10_000)
        stream = io.BytesIO(data)
        tree = MerkleTree.from_file(stream, chunk_size=1024)
        assert tree.num_chunks == 10  # ceil(10000/1024) = 10

    def test_proof_works_on_streamed_tree(self):
        data = os.urandom(50_000)
        stream = io.BytesIO(data)
        tree = MerkleTree.from_file(stream, chunk_size=16384)
        for i in range(tree.num_chunks):
            assert tree.get_proof(i).verify()


class TestBuildFileMerkleTree:
    def test_from_real_file(self, tmp_path):
        data = os.urandom(80_000)
        filepath = str(tmp_path / "test.bin")
        with open(filepath, "wb") as f:
            f.write(data)

        tree = build_file_merkle_tree(filepath)

        # Compare with in-memory
        chunks = chunk_bytes(data, 16384)
        hashes = [compute_hash(c) for c in chunks]
        expected = MerkleTree(hashes, chunk_size=16384)
        assert tree.root_hash == expected.root_hash

    def test_custom_chunk_size(self, tmp_path):
        data = os.urandom(5000)
        filepath = str(tmp_path / "small.bin")
        with open(filepath, "wb") as f:
            f.write(data)

        tree = build_file_merkle_tree(filepath, chunk_size=1024)
        assert tree.num_chunks == 5
