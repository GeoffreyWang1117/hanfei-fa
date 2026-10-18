"""
Alternative tree structures for ablation study.

Flat: all params at same level under model root
Two-level: chunks -> params -> model (no layer grouping)
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from merkle_verify.format import FORMAT_V1
from merkle_verify.merkle_tree import MerkleTree
from merkle_verify.hasher import compute_hash
from merkle_verify.chunking import chunk_bytes, DEFAULT_CHUNK_SIZE
from merkle_verify.comparison import ComparisonResult


@dataclass
class FlatModelTree:
    """Flat model tree: all params at same level, no layer grouping."""
    model_name: str = ""
    param_trees: Dict[str, MerkleTree] = field(default_factory=dict)
    model_root: Optional[str] = None

    def compute_model_root(self) -> str:
        """Compute root hash from all parameter roots (sorted for determinism)."""
        param_names = sorted(self.param_trees.keys())
        combined = ""
        for name in param_names:
            combined += self.param_trees[name].root_hash
        self.model_root = compute_hash(combined)
        return self.model_root


def build_flat_merkle_tree(
    blobs: Dict[str, bytes],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: str = "",
    **kwargs,
) -> FlatModelTree:
    """
    Build flat Merkle tree (no layer grouping).

    All blobs are placed directly under the model root.
    Each blob still gets its own MerkleTree of chunks, but
    there is no intermediate layer level for pruning.

    Args:
        blobs: Dict mapping names to raw bytes
        chunk_size: Size of chunks in bytes
        model_name: Name of model

    Returns:
        FlatModelTree
    """
    model_tree = FlatModelTree(model_name=model_name)
    for param_name, data in blobs.items():
        chunks = chunk_bytes(data, chunk_size)
        chunk_hashes = [compute_hash(chunk) for chunk in chunks]
        model_tree.param_trees[param_name] = MerkleTree(chunk_hashes, chunk_size, version=FORMAT_V1)
    model_tree.compute_model_root()
    return model_tree


def flat_compare(
    tree_a: FlatModelTree,
    tree_b: FlatModelTree,
    detailed: bool = True,
) -> Dict[str, Any]:
    """
    Compare two flat trees. No layer pruning possible.

    Without layer-level grouping, every parameter root must be compared
    individually, and changed parameters require a linear scan of all
    chunk hashes (no tree-walk optimisation).

    Args:
        tree_a: First flat model tree
        tree_b: Second flat model tree
        detailed: Whether to include chunk-level details

    Returns:
        Dictionary with comparison results (same format as compare_model_trees)
    """
    result = ComparisonResult()
    hash_comparisons = 0
    total_chunks_a = 0
    changed_chunks = 0

    all_params = set(tree_a.param_trees.keys()) | set(tree_b.param_trees.keys())

    for param_name in sorted(all_params):
        param_a = tree_a.param_trees.get(param_name)
        param_b = tree_b.param_trees.get(param_name)

        if param_a:
            total_chunks_a += param_a.num_chunks

        if param_a is None or param_b is None:
            result.changed_params.append(param_name)
            existing = param_a or param_b
            changed_chunks += existing.num_chunks
            continue

        # Compare param root hashes
        hash_comparisons += 1
        if param_a.root_hash != param_b.root_hash:
            result.changed_params.append(param_name)
            if detailed:
                # Linear scan -- no tree walk for flat comparison
                max_c = max(len(param_a.chunk_hashes), len(param_b.chunk_hashes))
                changed_indices = []
                for i in range(max_c):
                    hash_a = param_a.chunk_hashes[i] if i < len(param_a.chunk_hashes) else None
                    hash_b = param_b.chunk_hashes[i] if i < len(param_b.chunk_hashes) else None
                    hash_comparisons += 1
                    if hash_a != hash_b:
                        changed_indices.append(i)
                        changed_chunks += 1
                if changed_indices:
                    result.chunk_changes[param_name] = changed_indices
            else:
                changed_chunks += max(param_a.num_chunks, param_b.num_chunks)
        else:
            result.unchanged_params.append(param_name)

    # Compute byte estimates
    avg_chunk_size = DEFAULT_CHUNK_SIZE
    if tree_a.param_trees:
        first_param = next(iter(tree_a.param_trees.values()))
        avg_chunk_size = first_param.chunk_size

    result.total_bytes = total_chunks_a * avg_chunk_size
    result.total_changed_bytes = changed_chunks * avg_chunk_size
    if result.total_bytes > 0:
        result.change_percentage = (result.total_changed_bytes / result.total_bytes) * 100
    result.hash_comparisons = hash_comparisons

    return result.to_dict()


# TwoLevelModelTree is structurally identical to FlatModelTree:
# chunks -> params -> model (no layer grouping).
# We alias it to keep experiment code explicit about which variant is used.
TwoLevelModelTree = FlatModelTree
build_two_level_merkle_tree = build_flat_merkle_tree
two_level_compare = flat_compare
