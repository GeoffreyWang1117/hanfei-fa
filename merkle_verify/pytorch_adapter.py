"""
PyTorch integration for hanfei-fa.

Provides merkle_save/merkle_load/verify_checkpoint for PyTorch models
with integrity verification via hierarchical Merkle trees.

Requires: pip install torch
"""

import json
from typing import Dict, Optional, Any, Tuple, Union
from pathlib import Path

from merkle_verify.merkle_tree import (
    ModelMerkleTree,
    build_model_merkle_tree,
)
from merkle_verify.comparison import compare_model_trees
from merkle_verify.chunking import DEFAULT_CHUNK_SIZE


def _import_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch required. "
            "Install with: pip install hanfei-fa[torch]"
        )


def _state_dict_to_blobs(state_dict: Dict) -> Dict[str, bytes]:
    """Convert a PyTorch state_dict to raw byte blobs."""
    torch = _import_torch()
    blobs = {}
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            blobs[name] = tensor.detach().cpu().contiguous().numpy().tobytes()
        elif isinstance(tensor, bytes):
            blobs[name] = tensor
    return blobs


def _sidecar_path(filepath: str) -> str:
    """Get the sidecar .merkle.json path for a checkpoint file."""
    return str(Path(filepath).with_suffix(".merkle.json"))


def merkle_save(
    obj: Any,
    filepath: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: Optional[str] = None,
    parallel: bool = True,
    **torch_save_kwargs,
) -> ModelMerkleTree:
    """
    Save a PyTorch model/state_dict and generate Merkle manifest.

    Args:
        obj: Model, state_dict, or any picklable object with state_dict
        filepath: Path to save checkpoint
        chunk_size: Chunk size for Merkle tree
        model_name: Model name (default: filename stem)
        parallel: Use parallel hashing
        **torch_save_kwargs: Extra args forwarded to torch.save

    Returns:
        The constructed ModelMerkleTree
    """
    torch = _import_torch()

    # Extract state_dict
    if hasattr(obj, 'state_dict'):
        state_dict = obj.state_dict()
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise TypeError(
            f"Expected model with state_dict() or dict, got {type(obj)}"
        )

    # Save checkpoint via torch.save
    torch.save(state_dict, filepath, **torch_save_kwargs)

    # Build Merkle tree
    if model_name is None:
        model_name = Path(filepath).stem

    blobs = _state_dict_to_blobs(state_dict)
    tree = build_model_merkle_tree(
        blobs,
        chunk_size=chunk_size,
        model_name=model_name,
        parallel=parallel,
    )

    # Write sidecar manifest
    tree.to_json(_sidecar_path(filepath))

    return tree


def merkle_load(
    filepath: str,
    verify: bool = True,
    manifest: Optional[str] = None,
    map_location: Any = None,
    weights_only: bool = True,
    parallel: bool = True,
) -> Tuple[Dict, Optional[Dict[str, Any]]]:
    """
    Load a PyTorch checkpoint with optional Merkle verification.

    Args:
        filepath: Path to checkpoint file
        verify: Whether to verify integrity (default: True)
        manifest: Path to .merkle.json (default: auto-detect sidecar)
        map_location: torch.load map_location argument
        weights_only: torch.load weights_only argument (default: True)
        parallel: Use parallel hashing for verification

    Returns:
        Tuple of (state_dict, verification_details).
        verification_details is None if verify=False.
    """
    torch = _import_torch()

    state_dict = torch.load(
        filepath,
        map_location=map_location,
        weights_only=weights_only,
    )

    if not verify:
        return state_dict, None

    # Verify
    if manifest is None:
        manifest = _sidecar_path(filepath)

    try:
        stored_tree = ModelMerkleTree.from_json(manifest)
    except FileNotFoundError:
        details = {"verified": False, "error": f"Manifest not found: {manifest}"}
        return state_dict, details

    blobs = _state_dict_to_blobs(state_dict)
    chunk_size = _infer_chunk_size(stored_tree)
    current_tree = build_model_merkle_tree(
        blobs,
        chunk_size=chunk_size,
        model_name=stored_tree.model_name,
        parallel=parallel,
    )

    root_match = current_tree.model_root == stored_tree.model_root
    details: Dict[str, Any] = {
        "verified": root_match,
        "stored_root": stored_tree.model_root,
        "computed_root": current_tree.model_root,
    }

    if not root_match:
        result = compare_model_trees(current_tree, stored_tree, detailed=True)
        details["changed_layers"] = result["changed_layers"]
        details["changed_params"] = result["changed_params"]
        details["change_percentage"] = result["change_percentage"]

    return state_dict, details


def verify_checkpoint(
    filepath: str,
    manifest: Optional[str] = None,
    expected_root: Optional[str] = None,
    map_location: Any = None,
    weights_only: bool = True,
    parallel: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify a PyTorch checkpoint against its Merkle manifest.

    Args:
        filepath: Path to checkpoint file
        manifest: Path to .merkle.json (default: auto-detect sidecar)
        expected_root: If provided, only compare root hash (O(1))
        map_location: torch.load map_location
        weights_only: torch.load weights_only (default: True)
        parallel: Use parallel hashing

    Returns:
        Tuple of (is_valid, details_dict)
    """
    if manifest is None:
        manifest = _sidecar_path(filepath)

    try:
        stored_tree = ModelMerkleTree.from_json(manifest)
    except FileNotFoundError:
        return False, {"error": f"Manifest not found: {manifest}"}

    # Quick root-only check
    if expected_root is not None:
        match = stored_tree.model_root == expected_root
        return match, {"mode": "root_only", "stored_root": stored_tree.model_root}

    # Full verification
    torch = _import_torch()
    state_dict = torch.load(
        filepath,
        map_location=map_location,
        weights_only=weights_only,
    )

    blobs = _state_dict_to_blobs(state_dict)
    chunk_size = _infer_chunk_size(stored_tree)
    current_tree = build_model_merkle_tree(
        blobs,
        chunk_size=chunk_size,
        model_name=stored_tree.model_name,
        parallel=parallel,
    )

    root_match = current_tree.model_root == stored_tree.model_root
    details: Dict[str, Any] = {
        "mode": "full",
        "stored_root": stored_tree.model_root,
        "computed_root": current_tree.model_root,
    }

    if not root_match:
        result = compare_model_trees(current_tree, stored_tree, detailed=True)
        details["changed_layers"] = result["changed_layers"]
        details["changed_params"] = result["changed_params"]
        details["change_percentage"] = result["change_percentage"]

    return root_match, details


def diff_checkpoints(
    filepath_a: str,
    filepath_b: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    map_location: Any = None,
    weights_only: bool = True,
    parallel: bool = True,
) -> Dict[str, Any]:
    """
    Compare two PyTorch checkpoints.

    Args:
        filepath_a: Path to first checkpoint
        filepath_b: Path to second checkpoint
        chunk_size: Chunk size for tree construction
        map_location: torch.load map_location
        weights_only: torch.load weights_only (default: True)
        parallel: Use parallel hashing

    Returns:
        ComparisonResult with layer/param/chunk-level changes
    """
    torch = _import_torch()

    sd_a = torch.load(filepath_a, map_location=map_location, weights_only=weights_only)
    sd_b = torch.load(filepath_b, map_location=map_location, weights_only=weights_only)

    tree_a = build_model_merkle_tree(
        _state_dict_to_blobs(sd_a), chunk_size=chunk_size,
        model_name=Path(filepath_a).stem, parallel=parallel,
    )
    tree_b = build_model_merkle_tree(
        _state_dict_to_blobs(sd_b), chunk_size=chunk_size,
        model_name=Path(filepath_b).stem, parallel=parallel,
    )

    return compare_model_trees(tree_a, tree_b, detailed=True)


def _infer_chunk_size(tree: ModelMerkleTree) -> int:
    """Infer chunk size from a stored tree."""
    for layer_tree in tree.layer_trees.values():
        for param_tree in layer_tree.param_trees.values():
            return param_tree.chunk_size
    return DEFAULT_CHUNK_SIZE
