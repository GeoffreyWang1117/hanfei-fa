"""
Model comparison utilities.

Provides functions for comparing Merkle trees and calculating differences.
Uses tree-walk diff and layer pruning for O(k * log C) performance
instead of linear scanning all chunks.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from merkle_verify.merkle_tree import ModelMerkleTree


@dataclass
class ComparisonResult:
    """
    Result of comparing two model Merkle trees.

    Attributes:
        changed_params: List of parameter names that changed
        unchanged_params: List of parameter names that stayed the same
        changed_layers: List of layer names that changed
        total_changed_bytes: Estimated bytes of changed data
        total_bytes: Total model size in bytes
        change_percentage: Percentage of model that changed
        chunk_changes: Detailed chunk-level changes per parameter
        hash_comparisons: Number of hash comparisons performed
    """
    changed_params: List[str] = field(default_factory=list)
    unchanged_params: List[str] = field(default_factory=list)
    changed_layers: List[str] = field(default_factory=list)
    total_changed_bytes: int = 0
    total_bytes: int = 0
    change_percentage: float = 0.0
    chunk_changes: Dict[str, List[int]] = field(default_factory=dict)
    hash_comparisons: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'changed_params': self.changed_params,
            'unchanged_params': self.unchanged_params,
            'changed_layers': self.changed_layers,
            'total_changed_bytes': self.total_changed_bytes,
            'total_bytes': self.total_bytes,
            'change_percentage': self.change_percentage,
            'chunk_changes': self.chunk_changes,
            'hash_comparisons': self.hash_comparisons,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Changed: {len(self.changed_params)} params, "
            f"{len(self.changed_layers)} layers, "
            f"{self.change_percentage:.2f}% of model "
            f"({self.hash_comparisons} hash comparisons)"
        )


def compare_model_trees(
    tree_a: ModelMerkleTree,
    tree_b: ModelMerkleTree,
    detailed: bool = True
) -> Dict[str, Any]:
    """
    Compare two model Merkle trees and return differences.

    Uses a three-level pruning strategy:
    1. Layer-level: skip layers with identical layer_root hashes
    2. Param-level: skip params with identical root_hash values
    3. Chunk-level: tree-walk diff for O(k * log C) per parameter

    Args:
        tree_a: First model Merkle tree
        tree_b: Second model Merkle tree
        detailed: Whether to include chunk-level details

    Returns:
        Dictionary containing comparison results:
        - changed_params: List of changed parameter names
        - unchanged_params: List of unchanged parameter names
        - changed_layers: List of changed layer names
        - total_changed_bytes: Estimated bytes changed
        - total_bytes: Total model bytes
        - change_percentage: Percentage of change
        - hash_comparisons: Number of hash comparisons performed

    Example:
        >>> tree1 = build_model_merkle_tree(state_dict1)
        >>> tree2 = build_model_merkle_tree(state_dict2)
        >>> diff = compare_model_trees(tree1, tree2)
        >>> print(f"Changed: {diff['change_percentage']:.2f}%")
    """
    result = ComparisonResult()
    hash_comparisons = 0

    # Calculate total and changed bytes
    total_chunks_a = 0
    total_chunks_b = 0
    changed_chunks = 0

    # Get all layer names
    all_layers = set(tree_a.layer_trees.keys()) | set(tree_b.layer_trees.keys())

    for layer_name in sorted(all_layers):
        layer_a = tree_a.layer_trees.get(layer_name)
        layer_b = tree_b.layer_trees.get(layer_name)

        if layer_a is None or layer_b is None:
            # Layer exists in only one tree - all params changed
            result.changed_layers.append(layer_name)
            existing = layer_a or layer_b
            for param_name, param_tree in existing.param_trees.items():
                result.changed_params.append(param_name)
                changed_chunks += param_tree.num_chunks
                if existing == layer_a:
                    total_chunks_a += param_tree.num_chunks
                else:
                    total_chunks_b += param_tree.num_chunks
            continue

        # --- Layer-level pruning ---
        # Compare layer root hashes; skip entirely if identical
        hash_comparisons += 1
        if layer_a.layer_root == layer_b.layer_root:
            # Layer unchanged: add all params to unchanged, count chunks
            for param_name, param_tree in layer_a.param_trees.items():
                result.unchanged_params.append(param_name)
                total_chunks_a += param_tree.num_chunks
            for param_name, param_tree in layer_b.param_trees.items():
                total_chunks_b += param_tree.num_chunks
            continue

        # Layer differs - record it and drill into params
        result.changed_layers.append(layer_name)

        all_params = set(layer_a.param_trees.keys()) | set(layer_b.param_trees.keys())

        for param_name in sorted(all_params):
            param_a = layer_a.param_trees.get(param_name)
            param_b = layer_b.param_trees.get(param_name)

            if param_a:
                total_chunks_a += param_a.num_chunks
            if param_b:
                total_chunks_b += param_b.num_chunks

            if param_a is None or param_b is None:
                # Parameter exists in only one tree
                result.changed_params.append(param_name)
                existing = param_a or param_b
                changed_chunks += existing.num_chunks
                continue

            # --- Param-level pruning ---
            hash_comparisons += 1
            if param_a.root_hash == param_b.root_hash:
                result.unchanged_params.append(param_name)
                continue

            # Parameter differs
            result.changed_params.append(param_name)

            if detailed:
                # --- Chunk-level: tree-walk diff ---
                changed_indices, tree_comparisons = param_a.diff_tree(param_b)
                hash_comparisons += tree_comparisons

                if changed_indices:
                    result.chunk_changes[param_name] = changed_indices
                changed_chunks += len(changed_indices)
            else:
                # Approximate: assume all chunks changed
                changed_chunks += max(param_a.num_chunks, param_b.num_chunks)

    result.hash_comparisons = hash_comparisons

    # Calculate byte estimates
    avg_chunk_size = 16384  # Default
    if tree_a.layer_trees:
        first_layer = next(iter(tree_a.layer_trees.values()))
        if first_layer.param_trees:
            first_param = next(iter(first_layer.param_trees.values()))
            avg_chunk_size = first_param.chunk_size

    result.total_bytes = max(total_chunks_a, total_chunks_b) * avg_chunk_size
    result.total_changed_bytes = changed_chunks * avg_chunk_size

    if result.total_bytes > 0:
        result.change_percentage = (result.total_changed_bytes / result.total_bytes) * 100
    else:
        result.change_percentage = 0.0

    return result.to_dict()


def estimate_sync_savings(
    tree_a: ModelMerkleTree,
    tree_b: ModelMerkleTree
) -> Dict[str, Any]:
    """
    Estimate bandwidth savings from incremental sync.

    Args:
        tree_a: Source model Merkle tree
        tree_b: Target model Merkle tree

    Returns:
        Dictionary with sync statistics:
        - full_sync_bytes: Bytes needed for full sync
        - incremental_sync_bytes: Bytes needed for incremental sync
        - savings_bytes: Bytes saved
        - savings_percentage: Percentage saved

    Example:
        >>> savings = estimate_sync_savings(old_tree, new_tree)
        >>> print(f"Savings: {savings['savings_percentage']:.1f}%")
    """
    diff = compare_model_trees(tree_a, tree_b, detailed=True)

    full_sync_bytes = diff['total_bytes']
    incremental_sync_bytes = diff['total_changed_bytes']
    savings_bytes = full_sync_bytes - incremental_sync_bytes

    if full_sync_bytes > 0:
        savings_percentage = (savings_bytes / full_sync_bytes) * 100
    else:
        savings_percentage = 0.0

    return {
        'full_sync_bytes': full_sync_bytes,
        'incremental_sync_bytes': incremental_sync_bytes,
        'savings_bytes': savings_bytes,
        'savings_percentage': savings_percentage,
    }
