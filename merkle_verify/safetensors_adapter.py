"""
Safetensors integration for hanfei-fa.

Provides sign/verify/diff operations on .safetensors files
using per-tensor Merkle trees with metadata stored in __metadata__
or sidecar .merkle.json files.

Requires: pip install safetensors
"""

import json
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path

from merkle_verify.merkle_tree import (
    ModelMerkleTree,
    build_model_merkle_tree,
)
from merkle_verify.comparison import compare_model_trees
from merkle_verify.chunking import DEFAULT_CHUNK_SIZE


def _import_safetensors():
    """Import safetensors with a clear error message."""
    try:
        from safetensors import safe_open
        return safe_open
    except ImportError:
        raise ImportError(
            "safetensors package required. "
            "Install with: pip install hanfei-fa[safetensors]"
        )


def _read_tensor_bytes(filepath: str, framework: str = "np") -> Dict[str, bytes]:
    """
    Read all tensors from a safetensors file as raw bytes.

    Uses safe_open for memory-efficient per-tensor access.
    """
    safe_open = _import_safetensors()

    blobs = {}
    with safe_open(filepath, framework=framework) as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if framework == "np":
                blobs[key] = tensor.tobytes()
            else:
                # PyTorch / TensorFlow / etc.
                import numpy as np
                blobs[key] = np.array(tensor).tobytes()
    return blobs


def _read_metadata(filepath: str) -> Dict[str, str]:
    """Read __metadata__ from a safetensors file."""
    safe_open = _import_safetensors()
    with safe_open(filepath, framework="np") as f:
        return f.metadata() or {}


def _sidecar_path(filepath: str) -> str:
    """Get the sidecar .merkle.json path for a safetensors file."""
    return str(Path(filepath).with_suffix(".merkle.json"))


def sign(
    filepath: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: Optional[str] = None,
    sidecar: bool = True,
    parallel: bool = True,
) -> ModelMerkleTree:
    """
    Build a Merkle tree for a safetensors file and store the manifest.

    Args:
        filepath: Path to .safetensors file
        chunk_size: Chunk size in bytes (default: 16KB)
        model_name: Model name (default: filename stem)
        sidecar: If True, write .merkle.json sidecar file.
        parallel: Use parallel hashing

    Returns:
        The constructed ModelMerkleTree
    """
    if model_name is None:
        model_name = Path(filepath).stem

    blobs = _read_tensor_bytes(filepath)
    tree = build_model_merkle_tree(
        blobs,
        chunk_size=chunk_size,
        model_name=model_name,
        parallel=parallel,
    )

    if sidecar:
        tree.to_json(_sidecar_path(filepath))

    return tree


def verify(
    filepath: str,
    manifest: Optional[str] = None,
    expected_root: Optional[str] = None,
    parallel: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify a safetensors file against a stored Merkle manifest.

    Args:
        filepath: Path to .safetensors file
        manifest: Path to .merkle.json manifest (default: auto-detect sidecar)
        expected_root: If provided, only compare root hash (O(1) check)
        parallel: Use parallel hashing

    Returns:
        Tuple of (is_valid, details_dict)
    """
    details: Dict[str, Any] = {"filepath": filepath}

    # Load stored tree
    if manifest is None:
        manifest = _sidecar_path(filepath)

    try:
        stored_tree = ModelMerkleTree.from_json(manifest)
    except FileNotFoundError:
        return False, {**details, "error": f"Manifest not found: {manifest}"}

    # Quick O(1) root-only check
    if expected_root is not None:
        match = stored_tree.model_root == expected_root
        return match, {**details, "mode": "root_only", "stored_root": stored_tree.model_root}

    # Full verification: re-hash tensors and compare
    blobs = _read_tensor_bytes(filepath)
    current_tree = build_model_merkle_tree(
        blobs,
        chunk_size=_infer_chunk_size(stored_tree),
        model_name=stored_tree.model_name,
        parallel=parallel,
    )

    root_match = current_tree.model_root == stored_tree.model_root
    details["mode"] = "full"
    details["stored_root"] = stored_tree.model_root
    details["computed_root"] = current_tree.model_root
    details["root_match"] = root_match

    if not root_match:
        result = compare_model_trees(current_tree, stored_tree, detailed=True)
        details["changed_layers"] = result["changed_layers"]
        details["changed_params"] = result["changed_params"]
        details["change_percentage"] = result["change_percentage"]

    return root_match, details


def verify_tensor(
    filepath: str,
    tensor_name: str,
    manifest: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify a single tensor without loading others.

    Args:
        filepath: Path to .safetensors file
        tensor_name: Name of tensor to verify
        manifest: Path to manifest (default: auto-detect sidecar)

    Returns:
        Tuple of (is_valid, details_dict)
    """
    if manifest is None:
        manifest = _sidecar_path(filepath)

    stored_tree = ModelMerkleTree.from_json(manifest)

    # Find the tensor in stored tree
    stored_param_tree = None
    for layer_tree in stored_tree.layer_trees.values():
        if tensor_name in layer_tree.param_trees:
            stored_param_tree = layer_tree.param_trees[tensor_name]
            break

    if stored_param_tree is None:
        return False, {"error": f"Tensor '{tensor_name}' not found in manifest"}

    # Read just this one tensor
    safe_open = _import_safetensors()
    with safe_open(filepath, framework="np") as f:
        if tensor_name not in f.keys():
            return False, {"error": f"Tensor '{tensor_name}' not found in file"}
        tensor_bytes = f.get_tensor(tensor_name).tobytes()

    # Build tree for this tensor and compare root
    from merkle_verify.chunking import chunk_bytes
    from merkle_verify.hasher import compute_hash
    from merkle_verify.merkle_tree import MerkleTree

    chunks = chunk_bytes(tensor_bytes, stored_param_tree.chunk_size)
    chunk_hashes = [compute_hash(c) for c in chunks]
    current_param_tree = MerkleTree(chunk_hashes, stored_param_tree.chunk_size)

    match = current_param_tree.root_hash == stored_param_tree.root_hash
    details = {
        "tensor_name": tensor_name,
        "stored_root": stored_param_tree.root_hash,
        "computed_root": current_param_tree.root_hash,
        "num_chunks": current_param_tree.num_chunks,
    }

    return match, details


def diff(
    filepath_a: str,
    filepath_b: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    parallel: bool = True,
) -> Dict[str, Any]:
    """
    Compare two safetensors files and return detailed diff.

    Args:
        filepath_a: Path to first .safetensors file
        filepath_b: Path to second .safetensors file
        chunk_size: Chunk size for tree construction
        parallel: Use parallel hashing

    Returns:
        ComparisonResult with layer/param/chunk-level changes
    """
    blobs_a = _read_tensor_bytes(filepath_a)
    blobs_b = _read_tensor_bytes(filepath_b)

    tree_a = build_model_merkle_tree(
        blobs_a, chunk_size=chunk_size,
        model_name=Path(filepath_a).stem, parallel=parallel,
    )
    tree_b = build_model_merkle_tree(
        blobs_b, chunk_size=chunk_size,
        model_name=Path(filepath_b).stem, parallel=parallel,
    )

    return compare_model_trees(tree_a, tree_b, detailed=True)


def sign_sharded(
    directory: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: Optional[str] = None,
    parallel: bool = True,
) -> ModelMerkleTree:
    """
    Sign a sharded safetensors model (multiple .safetensors files + index.json).

    Args:
        directory: Directory containing model.safetensors.index.json and shard files
        chunk_size: Chunk size in bytes
        model_name: Model name (default: directory name)
        parallel: Use parallel hashing

    Returns:
        ModelMerkleTree covering all shards
    """
    import glob as globmod

    dirpath = Path(directory)
    index_path = dirpath / "model.safetensors.index.json"

    if not index_path.exists():
        # Try single-file model
        single = dirpath / "model.safetensors"
        if single.exists():
            return sign(str(single), chunk_size=chunk_size,
                        model_name=model_name, parallel=parallel)
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json in {directory}"
        )

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    shard_files = sorted(set(weight_map.values()))

    if model_name is None:
        model_name = dirpath.name

    # Read all tensors across shards
    all_blobs: Dict[str, bytes] = {}
    for shard_name in shard_files:
        shard_path = str(dirpath / shard_name)
        shard_blobs = _read_tensor_bytes(shard_path)
        all_blobs.update(shard_blobs)

    tree = build_model_merkle_tree(
        all_blobs,
        chunk_size=chunk_size,
        model_name=model_name,
        parallel=parallel,
    )

    # Write sidecar next to the index
    tree.to_json(str(dirpath / "model.merkle.json"))
    return tree


def from_hf_repo(
    repo_id: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    parallel: bool = True,
) -> ModelMerkleTree:
    """
    Build a Merkle tree from a HuggingFace Hub model (cached locally).

    Works with both single-file and sharded safetensors models.
    Requires the model to be already downloaded (uses local cache).

    Args:
        repo_id: HuggingFace repo ID (e.g., "bert-base-uncased")
        revision: Git revision (default: latest)
        cache_dir: HF cache directory (default: ~/.cache/huggingface/hub)
        chunk_size: Chunk size in bytes
        parallel: Use parallel hashing

    Returns:
        ModelMerkleTree

    Example:
        >>> tree = from_hf_repo("bert-base-uncased")
        >>> print(tree.model_root)
    """
    try:
        from huggingface_hub import scan_cache_dir, HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub required for HF repo integration. "
            "Install with: pip install huggingface_hub"
        )

    snapshot_dir = _resolve_hf_snapshot(repo_id, revision, cache_dir)

    # Check for sharded model
    index_path = Path(snapshot_dir) / "model.safetensors.index.json"
    single_path = Path(snapshot_dir) / "model.safetensors"

    if index_path.exists():
        return sign_sharded(
            snapshot_dir, chunk_size=chunk_size,
            model_name=repo_id, parallel=parallel,
        )
    elif single_path.exists():
        return sign(
            str(single_path), chunk_size=chunk_size,
            model_name=repo_id, sidecar=True, parallel=parallel,
        )
    else:
        raise FileNotFoundError(
            f"No safetensors files found in {snapshot_dir}. "
            f"Download first: huggingface-cli download {repo_id}"
        )


def _resolve_hf_snapshot(
    repo_id: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Resolve a HuggingFace repo to its local snapshot directory."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    else:
        cache_dir = Path(cache_dir)

    # HF cache layout: models--{owner}--{name}/snapshots/{commit}/
    repo_dir_name = "models--" + repo_id.replace("/", "--")
    repo_path = cache_dir / repo_dir_name

    if not repo_path.exists():
        raise FileNotFoundError(
            f"Model {repo_id} not found in cache at {repo_path}. "
            f"Download first: huggingface-cli download {repo_id}"
        )

    snapshots_dir = repo_path / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"No snapshots in {repo_path}")

    if revision:
        # Try exact match
        snap = snapshots_dir / revision
        if snap.exists():
            return str(snap)

    # Try reading refs/main
    refs_main = repo_path / "refs" / "main"
    if refs_main.exists():
        commit = refs_main.read_text().strip()
        snap = snapshots_dir / commit
        if snap.exists():
            return str(snap)

    # Fallback: most recent snapshot
    snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    if snapshots:
        return str(snapshots[-1])

    raise FileNotFoundError(f"No snapshots found for {repo_id}")


def _infer_chunk_size(tree: ModelMerkleTree) -> int:
    """Infer chunk size from a stored tree."""
    for layer_tree in tree.layer_trees.values():
        for param_tree in layer_tree.param_trees.values():
            return param_tree.chunk_size
    return DEFAULT_CHUNK_SIZE
