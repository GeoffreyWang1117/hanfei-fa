"""
PyTorch integration for merkle-weight-verify.

Provides sign_checkpoint / merkle_save / merkle_load / verify_checkpoint /
diff_checkpoints for PyTorch checkpoints, with integrity verification via
hierarchical Merkle trees.

Checkpoints are flattened before hashing: a training checkpoint such as
``{"model": state_dict, "optimizer": {...}, "epoch": 3}`` is hashed as
``{"model.<param>": tensor, "optimizer.state.0.exp_avg": tensor, ...}``.
Non-tensor values (ints, strings) are not weights and are ignored.

Requires: pip install torch
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from merkle_verify._tensor_bytes import flatten_state_dict, to_bytes
from merkle_verify.chunking import DEFAULT_CHUNK_SIZE
from merkle_verify.comparison import check_roots, compare_model_trees
from merkle_verify.merkle_tree import (
    ModelMerkleTree,
    build_model_merkle_tree,
)
from merkle_verify.safetensors_adapter import _infer_chunk_size, _resolve_chunk_size


def _import_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch required. "
            "Install with: pip install merkle-weight-verify[torch]"
        )


def _state_dict_to_blobs(state_dict: Dict) -> Dict[str, bytes]:
    """Flatten a (possibly nested) checkpoint and convert every tensor to raw bytes."""
    return {name: to_bytes(t) for name, t in flatten_state_dict(state_dict).items()}


def _sidecar_path(filepath: str) -> str:
    """Get the sidecar .merkle.json path for a checkpoint file."""
    return str(Path(filepath).with_suffix(".merkle.json"))


def _extract_state_dict(obj: Any) -> Dict:
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        return obj.state_dict()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Expected model with state_dict() or dict, got {type(obj)}")


def _tree_for(
    state_dict: Dict,
    chunk_size: int,
    model_name: str,
    parallel: bool,
) -> ModelMerkleTree:
    flat = flatten_state_dict(state_dict)
    if not flat:
        raise ValueError("checkpoint contains no tensors")
    return build_model_merkle_tree(
        flat, chunk_size=chunk_size, model_name=model_name, parallel=parallel,
    )


def _load(filepath: str, map_location: Any, weights_only: bool) -> Dict:
    torch = _import_torch()
    return torch.load(filepath, map_location=map_location, weights_only=weights_only)


def sign_checkpoint(
    filepath: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: Optional[str] = None,
    parallel: bool = True,
    map_location: Any = "cpu",
    weights_only: bool = True,
    sidecar: bool = True,
) -> ModelMerkleTree:
    """
    Build a Merkle manifest for an existing checkpoint without modifying it.

    Args:
        filepath: Path to a checkpoint written by ``torch.save``
        chunk_size: Chunk size for the Merkle tree
        model_name: Model name (default: filename stem)
        parallel: Use parallel hashing
        map_location: torch.load map_location (default: "cpu")
        weights_only: torch.load weights_only (default: True)
        sidecar: Write ``<file>.merkle.json`` next to the checkpoint

    Returns:
        The constructed ModelMerkleTree
    """
    state_dict = _load(filepath, map_location, weights_only)
    if model_name is None:
        model_name = Path(filepath).stem
    tree = _tree_for(state_dict, chunk_size, model_name, parallel)
    if sidecar:
        tree.to_json(_sidecar_path(filepath))
    return tree


def merkle_save(
    obj: Any,
    filepath: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: Optional[str] = None,
    parallel: bool = True,
    **torch_save_kwargs,
) -> ModelMerkleTree:
    """
    Save a PyTorch model/state_dict with ``torch.save`` and write its Merkle manifest.

    To create a manifest for a checkpoint that already exists, use
    :func:`sign_checkpoint`, which does not rewrite the file.

    Args:
        obj: Model, state_dict, or nested checkpoint dict
        filepath: Path to save checkpoint
        chunk_size: Chunk size for Merkle tree
        model_name: Model name (default: filename stem)
        parallel: Use parallel hashing
        **torch_save_kwargs: Extra args forwarded to torch.save

    Returns:
        The constructed ModelMerkleTree
    """
    torch = _import_torch()
    state_dict = _extract_state_dict(obj)
    torch.save(state_dict, filepath, **torch_save_kwargs)

    if model_name is None:
        model_name = Path(filepath).stem
    tree = _tree_for(state_dict, chunk_size, model_name, parallel)
    tree.to_json(_sidecar_path(filepath))
    return tree


def _load_manifest(manifest: str) -> Optional[ModelMerkleTree]:
    try:
        return ModelMerkleTree.from_json(manifest)
    except FileNotFoundError:
        return None


def merkle_load(
    filepath: str,
    verify: bool = True,
    manifest: Optional[str] = None,
    map_location: Any = None,
    weights_only: bool = True,
    parallel: bool = True,
    expected_root: Optional[str] = None,
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
        expected_root: Trusted model root; if given, the loaded weights must hash to it

    Returns:
        Tuple of (state_dict, verification_details).
        verification_details is None if verify=False; its "verified" key holds the verdict.
    """
    state_dict = _load(filepath, map_location, weights_only)
    if not verify:
        return state_dict, None

    stored_tree = _load_manifest(manifest or _sidecar_path(filepath))
    if stored_tree is None and expected_root is None:
        return state_dict, {
            "verified": False,
            "error": f"Manifest not found: {manifest or _sidecar_path(filepath)}",
        }

    chunk_size = _infer_chunk_size(stored_tree) if stored_tree else DEFAULT_CHUNK_SIZE
    name = stored_tree.model_name if stored_tree else Path(filepath).stem
    current_tree = _tree_for(state_dict, chunk_size, name, parallel)
    is_valid, details = check_roots(current_tree, stored_tree, expected_root)
    details["verified"] = is_valid
    return state_dict, details


def verify_checkpoint(
    filepath: str,
    manifest: Optional[str] = None,
    expected_root: Optional[str] = None,
    map_location: Any = None,
    weights_only: bool = True,
    parallel: bool = True,
    chunk_size: Optional[int] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify a PyTorch checkpoint.

    The checkpoint is always loaded and re-hashed. With ``expected_root`` the
    computed root must equal it (a manifest is then optional); otherwise it
    must equal the manifest's root.

    Args:
        filepath: Path to checkpoint file
        manifest: Path to .merkle.json (default: auto-detect sidecar)
        expected_root: Trusted model root obtained out of band
        map_location: torch.load map_location
        weights_only: torch.load weights_only (default: True)
        parallel: Use parallel hashing
        chunk_size: Chunk size override (default: from manifest, else 16KB)

    Returns:
        Tuple of (is_valid, details_dict)
    """
    manifest = manifest or _sidecar_path(filepath)
    stored_tree = _load_manifest(manifest)
    if stored_tree is None and expected_root is None:
        return False, {"error": f"Manifest not found: {manifest}"}

    chunk_size = _resolve_chunk_size(chunk_size, stored_tree, expected_root)
    name = stored_tree.model_name if stored_tree else Path(filepath).stem
    state_dict = _load(filepath, map_location, weights_only)
    current_tree = _tree_for(state_dict, chunk_size, name, parallel)
    return check_roots(current_tree, stored_tree, expected_root)


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
        Comparison dict with layer/param/chunk-level changes
    """
    sd_a = _load(filepath_a, map_location, weights_only)
    sd_b = _load(filepath_b, map_location, weights_only)
    tree_a = _tree_for(sd_a, chunk_size, Path(filepath_a).stem, parallel)
    tree_b = _tree_for(sd_b, chunk_size, Path(filepath_b).stem, parallel)
    return compare_model_trees(tree_a, tree_b, detailed=True)
