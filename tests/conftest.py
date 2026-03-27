"""Shared fixtures for hanfei-fa tests."""

import pytest
import os

from merkle_verify.hasher import compute_hash, set_default_algorithm, HashAlgorithm
from merkle_verify.chunking import chunk_bytes, DEFAULT_CHUNK_SIZE
from merkle_verify.merkle_tree import (
    MerkleTree,
    LayerMerkleTree,
    ModelMerkleTree,
    build_model_merkle_tree,
)


@pytest.fixture(autouse=True)
def reset_default_algorithm():
    """Ensure default hash algorithm is reset between tests."""
    set_default_algorithm(HashAlgorithm.SHA256)
    yield
    set_default_algorithm(HashAlgorithm.SHA256)


@pytest.fixture
def sample_data():
    """1 KB of deterministic bytes."""
    return bytes(range(256)) * 4


@pytest.fixture
def sample_blobs():
    """Small dict of named byte blobs mimicking model params."""
    return {
        "layer.0.weight": os.urandom(4096),
        "layer.0.bias": os.urandom(256),
        "layer.1.weight": os.urandom(4096),
        "layer.1.bias": os.urandom(256),
    }


@pytest.fixture
def simple_tree():
    """MerkleTree with 4 known chunk hashes."""
    hashes = [compute_hash(f"chunk{i}".encode()) for i in range(4)]
    return MerkleTree(hashes, chunk_size=16)


@pytest.fixture
def model_tree(sample_blobs):
    """ModelMerkleTree built from sample blobs."""
    return build_model_merkle_tree(sample_blobs, model_name="test-model", parallel=False)
