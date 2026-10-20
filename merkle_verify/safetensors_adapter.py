"""
Safetensors integration for merkle-weight-verify.

Provides sign/verify/diff operations on .safetensors files (single-file and
sharded) using per-tensor Merkle trees, with the manifest stored in a sidecar
``.merkle.json`` file.

Tensors are read straight from their byte ranges in the file by a built-in
reader (``merkle_verify._safetensors_io``), so:

* no third-party package is required (the ``safetensors`` package is optional),
* every dtype works, including BF16 and F8,
* memory use is one chunk per worker, not the size of the model.

The bytes hashed are identical to ``safe_open(..).get_tensor(name).tobytes()``,
so manifests are interchangeable with those produced by version 0.2.0.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from merkle_verify._chunk_hashing import hash_file_ranges, leaf_hasher
from merkle_verify._safetensors_io import (
    TensorInfo,
    read_header,
    read_tensor_bytes,
)
from merkle_verify import format as fmt
from merkle_verify.chunking import DEFAULT_CHUNK_SIZE
from merkle_verify.comparison import check_roots, compare_model_trees, plan_rehash
from merkle_verify.format import DEFAULT_FORMAT_VERSION
from merkle_verify.hasher import resolve_algorithm
from merkle_verify.merkle_tree import (
    DEFAULT_NUM_WORKERS,
    MerkleTree,
    ModelMerkleTree,
    assemble_model_tree,
)

INDEX_FILENAME = "model.safetensors.index.json"
SINGLE_FILENAME = "model.safetensors"
SHARDED_MANIFEST_FILENAME = "model.merkle.json"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _read_tensor_bytes(filepath: str, framework: str = "np") -> Dict[str, bytes]:
    """Read all tensors from a safetensors file as raw bytes.

    Kept for backward compatibility; the ``framework`` argument is ignored.
    Prefer the streaming builders, which never hold the whole model in memory.
    """
    tensors, _ = read_header(filepath)
    return {name: read_tensor_bytes(filepath, info) for name, info in tensors.items()}


def _read_metadata(filepath: str) -> Dict[str, str]:
    """Read ``__metadata__`` from a safetensors file."""
    return read_header(filepath)[1]


def _sidecar_path(filepath: str) -> str:
    """Get the sidecar .merkle.json path for a safetensors file."""
    return str(Path(filepath).with_suffix(".merkle.json"))


def _tensor_tree(
    filepath: str,
    info: TensorInfo,
    chunk_size: int,
    format_version: int = DEFAULT_FORMAT_VERSION,
    algorithm=None,
) -> MerkleTree:
    """Build one tensor's chunk tree by streaming its byte range."""
    alg = resolve_algorithm(algorithm)
    hashes = hash_file_ranges([(info.name, filepath, info.offset, info.nbytes)], chunk_size,
                              leaf_hasher(format_version, alg))[info.name]
    return _tree(hashes, info, chunk_size, format_version, alg)


def _tree(hashes, info: TensorInfo, chunk_size: int, format_version: int, alg) -> MerkleTree:
    return MerkleTree(
        hashes, chunk_size, version=format_version, algorithm=alg,
        num_bytes=info.nbytes, dtype=info.dtype, shape=info.shape,
    )


def _collect_tensors(files: Iterable[str]) -> List[Tuple[str, TensorInfo]]:
    """List (file, tensor) pairs across files, rejecting duplicate names."""
    seen: Dict[str, str] = {}
    out: List[Tuple[str, TensorInfo]] = []
    for path in files:
        tensors, _ = read_header(path)
        for name, info in tensors.items():
            if name in seen:
                raise ValueError(
                    f"tensor {name!r} appears in both {seen[name]} and {path}"
                )
            seen[name] = path
            out.append((path, info))
    return out


def _build_tree_from_files(
    files: Iterable[str],
    chunk_size: int,
    model_name: str,
    parallel: bool,
    num_workers: Optional[int] = None,
    format_version: int = DEFAULT_FORMAT_VERSION,
    algorithm=None,
) -> ModelMerkleTree:
    """Stream every tensor of the given safetensors files into one model tree."""
    entries = _collect_tensors(files)
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS
    fmt.check_format_version(format_version)
    alg = resolve_algorithm(algorithm)
    leaves = hash_file_ranges(
        [(info.name, path, info.offset, info.nbytes) for path, info in entries],
        chunk_size, leaf_hasher(format_version, alg),
        workers=num_workers if parallel else 1,
    )
    param_trees = {
        info.name: _tree(leaves[info.name], info, chunk_size, format_version, alg)
        for _, info in entries
    }
    return assemble_model_tree(param_trees, model_name=model_name)


def _infer_chunk_size(tree: ModelMerkleTree) -> int:
    """Infer chunk size from a stored tree."""
    for layer_tree in tree.layer_trees.values():
        for param_tree in layer_tree.param_trees.values():
            return param_tree.chunk_size
    return DEFAULT_CHUNK_SIZE


# ---------------------------------------------------------------------------
# Shared verification logic
# ---------------------------------------------------------------------------

def _load_manifest(manifest: str):
    """Returns (tree or None, error or None). A missing file is not an error here."""
    try:
        return ModelMerkleTree.from_json(manifest), None
    except FileNotFoundError:
        return None, None
    except (ValueError, KeyError, TypeError) as e:
        return None, f"invalid manifest {manifest}: {e}"


def _verify_against(
    build_current,
    manifest: str,
    expected_root: Optional[str],
    chunk_size: Optional[int],
    details: Dict[str, Any],
    format_version: Optional[int] = None,
    algorithm=None,
) -> Tuple[bool, Dict[str, Any]]:
    """Re-hash the weights and compare with a manifest and/or an expected root.

    ``build_current(plan, model_name)`` must return a freshly computed
    ModelMerkleTree using ``plan``'s format, algorithm and chunk size.
    The weights are always re-hashed.
    """
    stored_tree, error = _load_manifest(manifest)
    if error is not None and expected_root is None:
        return False, {**details, "error": error}
    if stored_tree is None and expected_root is None:
        return False, {**details, "error": f"Manifest not found: {manifest}"}

    plan = plan_rehash(stored_tree, expected_root, format_version, algorithm, chunk_size)
    name = stored_tree.model_name if stored_tree else ""
    current_tree = build_current(plan, name)
    is_valid, result = check_roots(current_tree, plan.stored, expected_root)
    details.update(result)
    details["format_version"] = plan.format_version
    details["algorithm"] = plan.algorithm.value
    if plan.note:
        details["manifest_ignored"] = plan.note
    if error is not None:
        details["manifest_ignored"] = error
    return is_valid, details


# ---------------------------------------------------------------------------
# Single file
# ---------------------------------------------------------------------------

def build_tree(
    filepath: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: Optional[str] = None,
    parallel: bool = True,
    *,
    format_version: int = DEFAULT_FORMAT_VERSION,
    algorithm=None,
) -> ModelMerkleTree:
    """Build a Merkle tree for a safetensors file without writing anything."""
    if model_name is None:
        model_name = Path(filepath).stem
    return _build_tree_from_files([filepath], chunk_size, model_name, parallel,
                                  format_version=format_version, algorithm=algorithm)


def sign(
    filepath: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: Optional[str] = None,
    sidecar: bool = True,
    parallel: bool = True,
    *,
    format_version: int = DEFAULT_FORMAT_VERSION,
    algorithm=None,
    include_chunks: bool = True,
) -> ModelMerkleTree:
    """
    Build a Merkle tree for a safetensors file and store the manifest.

    The weights file itself is only read, never modified.

    Args:
        filepath: Path to .safetensors file
        chunk_size: Chunk size in bytes (default: 16KB)
        model_name: Model name (default: filename stem)
        sidecar: If True, write .merkle.json sidecar file.
        parallel: Use parallel hashing
        format_version: Manifest format, 2 (default) or 1
        algorithm: Hash algorithm (default: SHA-256)
        include_chunks: False writes a compact v2 manifest without chunk hashes

    Returns:
        The constructed ModelMerkleTree
    """
    tree = build_tree(filepath, chunk_size=chunk_size, model_name=model_name, parallel=parallel,
                      format_version=format_version, algorithm=algorithm)
    if sidecar:
        tree.to_json(_sidecar_path(filepath), include_chunks=include_chunks)
    return tree


def verify(
    filepath: str,
    manifest: Optional[str] = None,
    expected_root: Optional[str] = None,
    parallel: bool = True,
    chunk_size: Optional[int] = None,
    *,
    format_version: Optional[int] = None,
    algorithm=None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify a safetensors file.

    The file is always re-hashed. What it is compared against:

    * ``expected_root`` given: the freshly computed root must equal it. A
      manifest is optional in this mode; if present it supplies the chunk size
      and is used to localise any difference.
    * otherwise: the computed root must equal the manifest's root.

    Args:
        filepath: Path to .safetensors file
        manifest: Path to .merkle.json manifest (default: auto-detect sidecar)
        expected_root: Trusted model root obtained out of band, optionally
            prefixed with its algorithm (``"sha256:..."``)
        parallel: Use parallel hashing
        chunk_size: Chunk size (default: from manifest, else 16KB)
        format_version, algorithm: Only needed with ``expected_root`` and no
            manifest (default: format 2, SHA-256 or the root's prefix)

    Returns:
        Tuple of (is_valid, details_dict)
    """
    if manifest is None:
        manifest = _sidecar_path(filepath)

    def build_current(plan, name: str) -> ModelMerkleTree:
        return build_tree(filepath, chunk_size=plan.chunk_size, model_name=name,
                          parallel=parallel, format_version=plan.format_version,
                          algorithm=plan.algorithm)

    return _verify_against(
        build_current, manifest, expected_root, chunk_size, {"filepath": filepath},
        format_version=format_version, algorithm=algorithm,
    )


def verify_tensor(
    filepath: str,
    tensor_name: str,
    manifest: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify a single tensor without reading the others.

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

    stored_param_tree = None
    for layer_tree in stored_tree.layer_trees.values():
        if tensor_name in layer_tree.param_trees:
            stored_param_tree = layer_tree.param_trees[tensor_name]
            break

    if stored_param_tree is None:
        return False, {"error": f"Tensor '{tensor_name}' not found in manifest"}

    tensors, _ = read_header(filepath)
    if tensor_name not in tensors:
        return False, {"error": f"Tensor '{tensor_name}' not found in file"}

    current = _tensor_tree(filepath, tensors[tensor_name], stored_param_tree.chunk_size,
                           stored_param_tree.version, stored_param_tree.algorithm)
    stored_root = stored_param_tree.tensor_root(tensor_name)
    computed_root = current.tensor_root(tensor_name)
    details = {
        "tensor_name": tensor_name,
        "stored_root": stored_root,
        "computed_root": computed_root,
        "num_chunks": current.num_chunks,
    }
    return computed_root == stored_root, details


def diff(
    filepath_a: str,
    filepath_b: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    parallel: bool = True,
    *,
    format_version: int = DEFAULT_FORMAT_VERSION,
    algorithm=None,
) -> Dict[str, Any]:
    """
    Compare two safetensors files and return a detailed diff.

    Args:
        filepath_a: Path to first .safetensors file
        filepath_b: Path to second .safetensors file
        chunk_size: Chunk size for tree construction
        parallel: Use parallel hashing
        format_version, algorithm: Hash format used for both trees

    Returns:
        Comparison dict with layer/param/chunk-level changes
    """
    kw = dict(chunk_size=chunk_size, parallel=parallel,
              format_version=format_version, algorithm=algorithm)
    tree_a = build_tree(filepath_a, **kw)
    tree_b = build_tree(filepath_b, **kw)
    return compare_model_trees(tree_a, tree_b, detailed=True)


# ---------------------------------------------------------------------------
# Sharded models (model.safetensors.index.json + shards)
# ---------------------------------------------------------------------------

def _shard_files(directory: str) -> List[str]:
    """Resolve the weight files of a model directory.

    Uses ``model.safetensors.index.json`` when present and checks that every
    tensor it lists actually exists in the shards; falls back to a single
    ``model.safetensors``.
    """
    dirpath = Path(directory)
    index_path = dirpath / INDEX_FILENAME
    if not index_path.exists():
        single = dirpath / SINGLE_FILENAME
        if single.exists():
            return [str(single)]
        raise FileNotFoundError(f"No {SINGLE_FILENAME} or {INDEX_FILENAME} in {directory}")

    with open(index_path) as f:
        index = json.load(f)
    weight_map: Dict[str, str] = index.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"{index_path}: empty or missing weight_map")

    files = [str(dirpath / shard) for shard in sorted(set(weight_map.values()))]
    missing_files = [p for p in files if not os.path.exists(p)]
    if missing_files:
        raise FileNotFoundError(f"shards listed in {INDEX_FILENAME} are missing: {missing_files}")

    present: Dict[str, str] = {}
    for path in files:
        tensors, _ = read_header(path)
        for name in tensors:
            shard = os.path.basename(path)
            if name in present:
                raise ValueError(f"tensor {name!r} appears in both {present[name]} and {shard}")
            present[name] = shard
    misplaced = [
        name for name, shard in weight_map.items() if present.get(name) != shard
    ]
    if misplaced:
        raise ValueError(
            f"{len(misplaced)} tensor(s) listed in {INDEX_FILENAME} are not in their "
            f"declared shard (first: {misplaced[0]!r})"
        )
    return files


def build_sharded_tree(
    directory: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: Optional[str] = None,
    parallel: bool = True,
    *,
    format_version: int = DEFAULT_FORMAT_VERSION,
    algorithm=None,
) -> ModelMerkleTree:
    """Build one Merkle tree over all shards of a model directory (read-only)."""
    if model_name is None:
        model_name = Path(directory).resolve().name
    return _build_tree_from_files(_shard_files(directory), chunk_size, model_name, parallel,
                                  format_version=format_version, algorithm=algorithm)


def sign_sharded(
    directory: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: Optional[str] = None,
    parallel: bool = True,
    sidecar: bool = True,
    *,
    format_version: int = DEFAULT_FORMAT_VERSION,
    algorithm=None,
    include_chunks: bool = True,
) -> ModelMerkleTree:
    """
    Sign a sharded safetensors model (multiple .safetensors files + index.json).

    Args:
        directory: Directory containing model.safetensors.index.json and shard files
        chunk_size: Chunk size in bytes
        model_name: Model name (default: directory name)
        parallel: Use parallel hashing
        sidecar: Write ``model.merkle.json`` into the directory

    Returns:
        ModelMerkleTree covering all shards
    """
    tree = build_sharded_tree(directory, chunk_size, model_name, parallel,
                              format_version=format_version, algorithm=algorithm)
    if sidecar:
        tree.to_json(str(Path(directory) / SHARDED_MANIFEST_FILENAME),
                     include_chunks=include_chunks)
    return tree


def verify_sharded(
    directory: str,
    manifest: Optional[str] = None,
    expected_root: Optional[str] = None,
    parallel: bool = True,
    chunk_size: Optional[int] = None,
    *,
    format_version: Optional[int] = None,
    algorithm=None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify a sharded (or single-file) safetensors model directory.

    Same comparison rules as :func:`verify`; the default manifest is
    ``<directory>/model.merkle.json``.
    """
    if manifest is None:
        manifest = str(Path(directory) / SHARDED_MANIFEST_FILENAME)

    def build_current(plan, name: str) -> ModelMerkleTree:
        return build_sharded_tree(directory, chunk_size=plan.chunk_size, model_name=name or None,
                                  parallel=parallel, format_version=plan.format_version,
                                  algorithm=plan.algorithm)

    return _verify_against(
        build_current, manifest, expected_root, chunk_size, {"directory": directory},
        format_version=format_version, algorithm=algorithm,
    )


# ---------------------------------------------------------------------------
# HuggingFace Hub cache
# ---------------------------------------------------------------------------

def _default_hf_cache() -> Path:
    """Locate the HF hub cache the same way huggingface_hub does."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "huggingface" / "hub"


def from_hf_repo(
    repo_id: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    parallel: bool = True,
    manifest_path: Optional[str] = None,
    *,
    format_version: int = DEFAULT_FORMAT_VERSION,
    algorithm=None,
) -> ModelMerkleTree:
    """
    Build a Merkle tree from a HuggingFace Hub model in the local cache.

    Works with both single-file and sharded safetensors models. The model
    must already be downloaded. Nothing is written into the HF cache; pass
    ``manifest_path`` to save the manifest somewhere.

    Args:
        repo_id: HuggingFace repo ID (e.g., "bert-base-uncased")
        revision: Commit hash, branch or tag (default: main)
        cache_dir: HF hub cache directory (default: $HF_HUB_CACHE, $HF_HOME/hub,
            or ~/.cache/huggingface/hub)
        chunk_size: Chunk size in bytes
        parallel: Use parallel hashing
        manifest_path: If given, write the manifest JSON to this path

    Returns:
        ModelMerkleTree

    Example:
        >>> tree = from_hf_repo("bert-base-uncased")
        >>> print(tree.model_root)
    """
    snapshot_dir = _resolve_hf_snapshot(repo_id, revision, cache_dir)
    try:
        tree = build_sharded_tree(
            snapshot_dir, chunk_size=chunk_size, model_name=repo_id, parallel=parallel,
            format_version=format_version, algorithm=algorithm,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No safetensors weights found in {snapshot_dir}. "
            f"Download first: huggingface-cli download {repo_id}"
        ) from None
    if manifest_path:
        tree.to_json(manifest_path)
    return tree


def _resolve_hf_snapshot(
    repo_id: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Resolve a HuggingFace repo to its local snapshot directory.

    A ``revision`` that cannot be found is an error; it never silently falls
    back to a different snapshot.
    """
    cache = Path(cache_dir) if cache_dir else _default_hf_cache()

    # HF cache layout: models--{owner}--{name}/{refs,snapshots}/
    repo_path = cache / ("models--" + repo_id.replace("/", "--"))
    if not repo_path.exists():
        raise FileNotFoundError(
            f"Model {repo_id} not found in cache at {repo_path}. "
            f"Download first: huggingface-cli download {repo_id}"
        )

    snapshots_dir = repo_path / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"No snapshots in {repo_path}")

    ref = revision or "main"
    snap = snapshots_dir / ref  # a full commit hash
    if snap.is_dir():
        return str(snap)

    ref_file = repo_path / "refs" / ref  # a branch or tag name
    if ref_file.exists():
        snap = snapshots_dir / ref_file.read_text().strip()
        if snap.is_dir():
            return str(snap)

    if revision is not None:
        raise FileNotFoundError(f"Revision {revision!r} of {repo_id} is not in the local cache")

    snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    if len(snapshots) == 1:
        return str(snapshots[0])
    raise FileNotFoundError(
        f"{repo_id}: no refs/main and {len(snapshots)} snapshots in cache; "
        "pass revision= explicitly"
    )
