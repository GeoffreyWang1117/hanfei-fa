"""
hanfei-fa: ML model weight integrity verification via hierarchical Merkle trees.

Verify integrity of ML model weights, datasets, or any large binary files
with O(1) root hash comparison and fine-grained chunk-level diff.
"""

from merkle_verify.merkle_tree import (
    MerkleTree,
    LayerMerkleTree,
    ModelMerkleTree,
    build_model_merkle_tree,
    build_file_merkle_tree,
)
from merkle_verify.hasher import (
    compute_hash,
    hash_chunk,
    hash_pair,
    verify_hash,
    HashAlgorithm,
    set_default_algorithm,
    get_default_algorithm,
)
from merkle_verify.chunking import chunk_bytes, chunk_tensor
from merkle_verify.comparison import (
    compare_model_trees,
    estimate_sync_savings,
    ComparisonResult,
)
from merkle_verify.strategies import (
    ChunkStrategy,
    FixedChunkStrategy,
    AdaptiveChunkStrategy,
    TheoreticalOptimalStrategy,
)

__version__ = "0.2.0"

# Optional integrations (imported on demand, not at top level)
# - merkle_verify.safetensors_adapter: pip install hanfei-fa[safetensors]
# - merkle_verify.pytorch_adapter: pip install hanfei-fa[torch]

__all__ = [
    "MerkleTree",
    "LayerMerkleTree",
    "ModelMerkleTree",
    "compute_hash",
    "hash_chunk",
    "hash_pair",
    "verify_hash",
    "HashAlgorithm",
    "set_default_algorithm",
    "get_default_algorithm",
    "chunk_bytes",
    "chunk_tensor",
    "compare_model_trees",
    "estimate_sync_savings",
    "ComparisonResult",
    "ChunkStrategy",
    "FixedChunkStrategy",
    "AdaptiveChunkStrategy",
    "TheoreticalOptimalStrategy",
    "build_file_merkle_tree",
]
