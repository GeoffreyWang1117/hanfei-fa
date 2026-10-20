"""
Model comparison utilities.

Provides functions for comparing Merkle trees and calculating differences.
Uses tree-walk diff and layer pruning for O(k * log C) performance
instead of linear scanning all chunks.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from merkle_verify.chunking import DEFAULT_CHUNK_SIZE
from merkle_verify.format import DEFAULT_FORMAT_VERSION, check_format_version, parse_root
from merkle_verify.hasher import HashAlgorithm, resolve_algorithm
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


def _check_same_format(tree_a: ModelMerkleTree, tree_b: ModelMerkleTree) -> None:
    if not tree_a.layer_trees or not tree_b.layer_trees:
        return
    fa = (tree_a.format_version, tree_a.algorithm)
    fb = (tree_b.format_version, tree_b.algorithm)
    if fa != fb:
        raise ValueError(
            f"cannot compare a v{fa[0]}/{fa[1].value} tree with a v{fb[0]}/{fb[1].value} tree; "
            "rebuild both with the same format and algorithm"
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
    _check_same_format(tree_a, tree_b)
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
            # (v2: the tensor root also binds name, dtype and shape)
            hash_comparisons += 1
            if param_a.tensor_root(param_name) == param_b.tensor_root(param_name):
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


@dataclass
class RehashPlan:
    """How to re-hash weights for verification, and whether the manifest can be compared."""
    format_version: int
    algorithm: HashAlgorithm
    chunk_size: int
    stored: Optional[ModelMerkleTree]
    note: Optional[str] = None


def _manifest_chunk_size(tree: ModelMerkleTree) -> Optional[int]:
    for layer_tree in tree.layer_trees.values():
        for param_tree in layer_tree.param_trees.values():
            return param_tree.chunk_size
    return None


def plan_rehash(
    stored_tree: Optional[ModelMerkleTree],
    expected_root: Optional[str] = None,
    format_version: Optional[int] = None,
    algorithm=None,
    chunk_size: Optional[int] = None,
) -> RehashPlan:
    """
    Decide the format, algorithm and chunk size to re-hash the weights with.

    * Against a manifest (no ``expected_root``): everything comes from the
      manifest, since any other choice could never reproduce its root. An
      explicit argument that disagrees is an error.
    * Against a trusted ``expected_root``: explicit arguments win, then the
      root's ``"algorithm:"`` prefix, then the manifest, then the defaults.
      A manifest hashed differently is not used to localise changes.
    """
    if format_version is not None:
        check_format_version(format_version)
    alg_arg = resolve_algorithm(algorithm) if algorithm is not None else None
    have = None
    if stored_tree is not None and stored_tree.layer_trees:
        have = (stored_tree.format_version, stored_tree.algorithm, _manifest_chunk_size(stored_tree))

    if expected_root is None:
        if have is None:
            raise ValueError("need a manifest or an expected_root to verify against")
        for label, given, value in (("format_version", format_version, have[0]),
                                    ("algorithm", alg_arg, have[1]),
                                    ("chunk_size", chunk_size, have[2])):
            if given is not None and given != value:
                shown = (lambda v: v.value if isinstance(v, HashAlgorithm) else v)
                raise ValueError(
                    f"{label} {shown(given)} differs from the manifest's {label} {shown(value)}"
                )
        return RehashPlan(have[0], have[1], have[2], stored_tree)

    prefix_alg = parse_root(expected_root)[0]
    prefix_alg = resolve_algorithm(prefix_alg) if prefix_alg else None
    if alg_arg is not None and prefix_alg is not None and alg_arg != prefix_alg:
        raise ValueError(
            f"algorithm {alg_arg.value} conflicts with the expected root's prefix {prefix_alg.value}"
        )
    version = format_version or (have[0] if have else DEFAULT_FORMAT_VERSION)
    alg = alg_arg or prefix_alg or (have[1] if have else resolve_algorithm(None))
    cs = chunk_size or (have[2] if have else DEFAULT_CHUNK_SIZE)
    if have is None:
        return RehashPlan(version, alg, cs, None)
    if have == (version, alg, cs):
        return RehashPlan(version, alg, cs, stored_tree)
    return RehashPlan(version, alg, cs, None,
                      note="manifest uses a different format, algorithm or chunk size; "
                           "it was not used to localise changes")


def check_roots(
    current_tree: ModelMerkleTree,
    stored_tree: Optional[ModelMerkleTree] = None,
    expected_root: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide whether freshly computed weights are the expected ones.

    ``current_tree`` must have been computed from the weights just now; a root
    read from a manifest proves nothing about the file next to it.

    * With ``expected_root`` (a trusted root obtained out of band) the verdict
      is ``current_tree.model_root == expected_root``. ``stored_tree`` is then
      only used to localise differences and to flag a manifest that disagrees
      with the trusted root.
    * Without it, the verdict is ``current_tree.model_root == stored_tree.model_root``.

    Returns:
        (is_valid, details) where details has ``mode`` ("expected_root" or
        "full"), ``computed_root``, ``root_match`` and, on mismatch with a
        manifest available, ``changed_layers`` / ``changed_params`` /
        ``change_percentage``.
    """
    if expected_root is None and stored_tree is None:
        raise ValueError("need a stored manifest or an expected_root to verify against")

    details: Dict[str, Any] = {"computed_root": current_tree.model_root}
    if expected_root is not None:
        is_valid = current_tree.model_root == parse_root(expected_root)[1]
        details["mode"] = "expected_root"
        details["expected_root"] = expected_root
        if stored_tree is not None:
            details["stored_root"] = stored_tree.model_root
            if stored_tree.model_root != parse_root(expected_root)[1]:
                details["manifest_mismatch"] = True
    else:
        is_valid = current_tree.model_root == stored_tree.model_root
        details["mode"] = "full"
        details["stored_root"] = stored_tree.model_root
    details["root_match"] = is_valid

    if not is_valid and stored_tree is not None:
        result = compare_model_trees(current_tree, stored_tree, detailed=True)
        details["changed_layers"] = result["changed_layers"]
        details["changed_params"] = result["changed_params"]
        details["change_percentage"] = result["change_percentage"]
    return is_valid, details
