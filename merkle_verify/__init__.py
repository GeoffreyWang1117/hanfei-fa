"""
merkle-weight-verify: Hierarchical Merkle tree for large file integrity verification.

Verify integrity of ML model weights, datasets, or any large binary files
with O(1) root hash comparison and fine-grained chunk-level diff.
"""

from merkle_verify.merkle_tree import (
    MerkleTree,
    LayerMerkleTree,
    ModelMerkleTree,
    build_model_merkle_tree,
    build_file_merkle_tree,
    assemble_model_tree,
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
    check_roots,
    ComparisonResult,
)
from merkle_verify._tensor_bytes import RawTensor
from merkle_verify.format import (
    DEFAULT_FORMAT_VERSION,
    FORMAT_V1,
    FORMAT_V2,
    hash_leaf,
    hash_node,
)
from merkle_verify.merkle_tree import ChunkProof, MerkleProof, TensorProof
from merkle_verify.strategies import (
    ChunkStrategy,
    FixedChunkStrategy,
    AdaptiveChunkStrategy,
    TheoreticalOptimalStrategy,
)

__version__ = "0.3.0.dev0"

# Integrations (imported on demand, not at top level)
# - merkle_verify.safetensors_adapter: no extra dependency (built-in format reader)
# - merkle_verify.pytorch_adapter: pip install merkle-weight-verify[torch]

__all__ = [
    "build_model_merkle_tree",
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
    "assemble_model_tree",
    "check_roots",
    "RawTensor",
    "DEFAULT_FORMAT_VERSION",
    "FORMAT_V1",
    "FORMAT_V2",
    "hash_leaf",
    "hash_node",
    "MerkleProof",
    "TensorProof",
    "ChunkProof",
]
