"""Tests for merkle_verify.chunking module."""

import os

import pytest

from merkle_verify.chunking import (
    chunk_bytes,
    chunk_bytes_iter,
    chunk_tensor,
    estimate_chunk_count,
    DEFAULT_CHUNK_SIZE,
)


class TestChunkBytes:
    def test_exact_multiple(self):
        data = b"a" * 32
        chunks = chunk_bytes(data, chunk_size=16)
        assert len(chunks) == 2
        assert all(len(c) == 16 for c in chunks)

    def test_non_exact_multiple(self):
        data = b"a" * 30
        chunks = chunk_bytes(data, chunk_size=16)
        assert len(chunks) == 2
        assert len(chunks[0]) == 16
        assert len(chunks[1]) == 14

    def test_smaller_than_chunk(self):
        data = b"abc"
        chunks = chunk_bytes(data, chunk_size=16)
        assert len(chunks) == 1
        assert chunks[0] == data

    def test_empty(self):
        assert chunk_bytes(b"", chunk_size=16) == []

    def test_roundtrip(self):
        data = b"hello world"
        chunks = chunk_bytes(data, chunk_size=4)
        assert b"".join(chunks) == data

    def test_default_chunk_size(self):
        data = b"x" * (DEFAULT_CHUNK_SIZE * 3 + 100)
        chunks = chunk_bytes(data)
        assert len(chunks) == 4


class TestChunkBytesIter:
    def test_matches_list_version(self):
        data = b"x" * 100
        assert list(chunk_bytes_iter(data, 16)) == chunk_bytes(data, 16)


class TestChunkTensor:
    def test_same_as_chunk_bytes(self):
        tensor_bytes = os.urandom(400)
        assert chunk_tensor(tensor_bytes, 64) == chunk_bytes(tensor_bytes, 64)


class TestEstimateChunkCount:
    def test_exact(self):
        assert estimate_chunk_count(1024, 256) == 4

    def test_ceiling(self):
        assert estimate_chunk_count(1025, 256) == 5

    def test_zero(self):
        assert estimate_chunk_count(0, 256) == 0
