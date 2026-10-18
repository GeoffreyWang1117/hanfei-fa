"""
merkle-verify CLI: command-line tool for model weight integrity verification.

Usage:
    merkle-verify hash <file>                    Compute Merkle root hash of a file
    merkle-verify sign <file|dir>                Build tree + write .merkle.json manifest
    merkle-verify verify <file|dir>              Re-hash and check against the manifest
    merkle-verify verify <file|dir> --expected-root ROOT
                                                 Re-hash and check against a trusted root
    merkle-verify diff <a> <b>                   Compare two files or two manifests
    merkle-verify info <manifest>                Show info about a .merkle.json manifest
    merkle-verify hf-sign <repo_id>              Hash a HuggingFace model from the local cache

``sign`` and ``verify`` never modify the weights. A directory is treated as a
(possibly sharded) safetensors model: model.safetensors.index.json + shards,
or a single model.safetensors.

Exit codes: 0 = OK, 1 = verification failed, 2 = error (bad input, missing file).
"""

import argparse
import json
import sys
import time
from pathlib import Path

from merkle_verify import __version__
from merkle_verify.chunking import DEFAULT_CHUNK_SIZE
from merkle_verify.merkle_tree import MerkleTree, ModelMerkleTree, build_file_merkle_tree

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_ERROR = 2

_TORCH_SUFFIXES = (".pt", ".pth")


class CLIError(Exception):
    """An error that should be reported without a traceback."""


def _kind(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        return "dir"
    suffix = p.suffix.lower()
    if suffix == ".safetensors":
        return "safetensors"
    if suffix in _TORCH_SUFFIXES:
        return "torch"
    if suffix == ".json":
        return "json"
    return "generic"


def _need_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise CLIError("torch not installed. Run: pip install merkle-weight-verify[torch]")


def _print_model_summary(tree: ModelMerkleTree, elapsed: float, verbose: bool) -> None:
    print(f"root: {tree.model_root}")
    if verbose:
        n_params = sum(len(lt.param_trees) for lt in tree.layer_trees.values())
        print(f"  tensors: {n_params}")
        print(f"  layers: {len(tree.layer_trees)}")
        print(f"  time: {elapsed:.3f}s")


def cmd_hash(args):
    """Compute Merkle root hash of a file."""
    t0 = time.time()
    tree = build_file_merkle_tree(args.file, chunk_size=args.chunk_size)
    elapsed = time.time() - t0

    print(f"{tree.root_hash}  {args.file}")
    if args.verbose:
        print(f"  chunks: {tree.num_chunks}")
        print(f"  chunk_size: {tree.chunk_size}")
        print(f"  time: {elapsed:.3f}s")
    return EXIT_OK


def cmd_sign(args):
    """Build a Merkle tree for a model file or directory and write the manifest."""
    path = args.file
    kind = _kind(path)
    t0 = time.time()

    if kind == "dir":
        from merkle_verify.safetensors_adapter import sign_sharded, SHARDED_MANIFEST_FILENAME
        tree = sign_sharded(path, chunk_size=args.chunk_size)
        manifest = str(Path(path) / SHARDED_MANIFEST_FILENAME)
    elif kind == "safetensors":
        from merkle_verify.safetensors_adapter import sign, _sidecar_path
        tree = sign(path, chunk_size=args.chunk_size)
        manifest = _sidecar_path(path)
    elif kind == "torch":
        _need_torch()
        from merkle_verify.pytorch_adapter import sign_checkpoint, _sidecar_path
        tree = sign_checkpoint(path, chunk_size=args.chunk_size)
        manifest = _sidecar_path(path)
    else:
        file_tree = build_file_merkle_tree(path, chunk_size=args.chunk_size)
        manifest = str(Path(path).with_suffix(".merkle.json"))
        with open(manifest, "w") as f:
            json.dump(file_tree.to_dict(), f, indent=2)
        print(f"root: {file_tree.root_hash}")
        print(f"manifest: {manifest}")
        if args.verbose:
            print(f"  chunks: {file_tree.num_chunks}")
            print(f"  time: {time.time() - t0:.3f}s")
        return EXIT_OK

    _print_model_summary(tree, time.time() - t0, args.verbose)
    print(f"manifest: {manifest}")
    return EXIT_OK


def _verify_generic(path: str, manifest_path: str, expected_root, chunk_size_arg):
    try:
        with open(manifest_path) as f:
            stored = json.load(f)
    except FileNotFoundError:
        stored = None
        if expected_root is None:
            return False, {"error": f"manifest not found: {manifest_path}"}

    if stored is not None and "root_hash" not in stored:
        raise CLIError(f"{manifest_path}: not a single-file manifest")
    chunk_size = stored.get("chunk_size", DEFAULT_CHUNK_SIZE) if stored else chunk_size_arg
    tree = build_file_merkle_tree(path, chunk_size=chunk_size)
    reference = expected_root if expected_root is not None else stored["root_hash"]
    details = {"computed_root": tree.root_hash}
    if expected_root is not None:
        details["expected_root"] = expected_root
    else:
        details["stored_root"] = stored["root_hash"]
    if tree.root_hash != reference and stored and "chunk_hashes" in stored:
        changed, _ = tree.diff_tree(MerkleTree.from_dict(stored))
        details["changed_chunks"] = changed
    return tree.root_hash == reference, details


def cmd_verify(args):
    """Re-hash a file or model directory and check it."""
    path = args.file
    kind = _kind(path)
    expected = args.expected_root
    chunk_size = args.chunk_size if args.chunk_size_given else None
    t0 = time.time()

    if kind == "dir":
        from merkle_verify.safetensors_adapter import verify_sharded
        is_valid, details = verify_sharded(
            path, manifest=args.manifest, expected_root=expected, chunk_size=chunk_size)
    elif kind == "safetensors":
        from merkle_verify.safetensors_adapter import verify
        is_valid, details = verify(
            path, manifest=args.manifest, expected_root=expected, chunk_size=chunk_size)
    elif kind == "torch":
        _need_torch()
        from merkle_verify.pytorch_adapter import verify_checkpoint
        is_valid, details = verify_checkpoint(
            path, manifest=args.manifest, expected_root=expected, chunk_size=chunk_size)
    else:
        manifest_path = args.manifest or str(Path(path).with_suffix(".merkle.json"))
        is_valid, details = _verify_generic(path, manifest_path, expected, args.chunk_size)
    elapsed = time.time() - t0

    if is_valid:
        print(f"OK: {path}")
    else:
        print(f"FAIL: {path}")
        if "error" in details:
            print(f"  error: {details['error']}")
        else:
            reference = details.get("expected_root", details.get("stored_root", "?"))
            label = "expected" if "expected_root" in details else "stored"
            print(f"  {label}: {reference}")
            print(f"  computed: {details.get('computed_root', '?')}")
            if details.get("changed_params"):
                print(f"  changed params: {details['changed_params']}")
            if details.get("changed_chunks"):
                changed = details["changed_chunks"]
                more = "..." if len(changed) > 20 else ""
                print(f"  changed chunks: {changed[:20]}{more}")
    if details.get("manifest_mismatch"):
        print("  warning: the manifest's root differs from --expected-root")

    if args.verbose:
        print(f"  time: {elapsed:.3f}s")
    return EXIT_OK if is_valid else EXIT_FAIL


def _load_json_manifest(path: str):
    with open(path) as f:
        data = json.load(f)
    if "layer_trees" in data:
        return ModelMerkleTree.from_dict(data)
    if "chunk_hashes" in data:
        return MerkleTree.from_dict(data)
    raise CLIError(f"{path}: not a merkle-verify manifest")


def _print_chunk_diff(tree_a: MerkleTree, tree_b: MerkleTree, elapsed: float, verbose: bool):
    changed, comparisons = tree_a.diff_tree(tree_b)
    total = max(tree_a.num_chunks, tree_b.num_chunks)
    pct = (len(changed) / total * 100) if total > 0 else 0
    print(f"{len(changed)}/{total} chunks changed ({pct:.1f}%), {comparisons} hash comparisons")
    if verbose:
        print(f"  time: {elapsed:.3f}s")
        if changed:
            print(f"  changed indices: {changed[:20]}{'...' if len(changed) > 20 else ''}")


def cmd_diff(args):
    """Compare two files, model directories, or manifests."""
    from merkle_verify.comparison import compare_model_trees

    kind_a, kind_b = _kind(args.file_a), _kind(args.file_b)
    if kind_a != kind_b:
        raise CLIError(f"cannot diff a {kind_a} input against a {kind_b} input")
    t0 = time.time()

    if kind_a == "dir":
        from merkle_verify.safetensors_adapter import build_sharded_tree
        result = compare_model_trees(
            build_sharded_tree(args.file_a, chunk_size=args.chunk_size),
            build_sharded_tree(args.file_b, chunk_size=args.chunk_size),
        )
    elif kind_a == "safetensors":
        from merkle_verify.safetensors_adapter import diff
        result = diff(args.file_a, args.file_b, chunk_size=args.chunk_size)
    elif kind_a == "torch":
        _need_torch()
        from merkle_verify.pytorch_adapter import diff_checkpoints
        result = diff_checkpoints(args.file_a, args.file_b, chunk_size=args.chunk_size)
    elif kind_a == "json":
        tree_a = _load_json_manifest(args.file_a)
        tree_b = _load_json_manifest(args.file_b)
        if type(tree_a) is not type(tree_b):
            raise CLIError("cannot diff a model manifest against a single-file manifest")
        if isinstance(tree_a, MerkleTree):
            _print_chunk_diff(tree_a, tree_b, time.time() - t0, args.verbose)
            return EXIT_OK
        result = compare_model_trees(tree_a, tree_b, detailed=True)
    else:
        tree_a = build_file_merkle_tree(args.file_a, chunk_size=args.chunk_size)
        tree_b = build_file_merkle_tree(args.file_b, chunk_size=args.chunk_size)
        _print_chunk_diff(tree_a, tree_b, time.time() - t0, args.verbose)
        return EXIT_OK
    elapsed = time.time() - t0

    total_params = len(result["changed_params"]) + len(result["unchanged_params"])
    print(f"changed layers: {result['changed_layers']}")
    print(f"changed params: {len(result['changed_params'])} / {total_params}")
    print(f"change: {result['change_percentage']:.1f}%")
    print(f"hash comparisons: {result['hash_comparisons']}")
    if args.verbose:
        print(f"  time: {elapsed:.3f}s")
        if result["changed_params"]:
            print(f"  changed: {result['changed_params'][:10]}")
    return EXIT_OK


def cmd_info(args):
    """Show info about a .merkle.json manifest."""
    tree = _load_json_manifest(args.manifest)
    if isinstance(tree, MerkleTree):
        print(f"root: {tree.root_hash}")
        print(f"chunks: {tree.num_chunks}")
        print(f"chunk_size: {tree.chunk_size}")
        return EXIT_OK

    n_params = sum(len(lt.param_trees) for lt in tree.layer_trees.values())
    n_chunks = sum(
        pt.num_chunks
        for lt in tree.layer_trees.values()
        for pt in lt.param_trees.values()
    )
    print(f"model: {tree.model_name}")
    print(f"root: {tree.model_root}")
    print(f"layers: {len(tree.layer_trees)}")
    print(f"params: {n_params}")
    print(f"chunks: {n_chunks}")
    if args.verbose:
        for lname, lt in sorted(tree.layer_trees.items()):
            print(f"  {lname}: {len(lt.param_trees)} params, root={lt.layer_root[:16]}...")
    return EXIT_OK


def cmd_hf_sign(args):
    """Hash a HuggingFace Hub model from the local cache (read-only)."""
    from merkle_verify.safetensors_adapter import from_hf_repo

    t0 = time.time()
    tree = from_hf_repo(
        args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        chunk_size=args.chunk_size,
        manifest_path=args.output,
    )
    elapsed = time.time() - t0
    n_params = sum(len(lt.param_trees) for lt in tree.layer_trees.values())
    print(f"root: {tree.model_root}")
    print(f"model: {tree.model_name}")
    print(f"layers: {len(tree.layer_trees)}, params: {n_params}")
    if args.output:
        print(f"manifest: {args.output}")
    if args.verbose:
        print(f"  time: {elapsed:.2f}s")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="merkle-verify",
        description="Model weight integrity verification via hierarchical Merkle trees",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help=f"Chunk size in bytes (default: {DEFAULT_CHUNK_SIZE}; "
             "verify uses the manifest's chunk size unless this is given)"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_hash = sub.add_parser("hash", help="Compute Merkle root hash")
    p_hash.add_argument("file", help="File to hash")

    p_sign = sub.add_parser("sign", help="Build tree + write .merkle.json (weights are not modified)")
    p_sign.add_argument("file", help="File or model directory to sign")

    p_verify = sub.add_parser("verify", help="Re-hash and verify against a manifest or root")
    p_verify.add_argument("file", help="File or model directory to verify")
    p_verify.add_argument("--manifest", help="Path to .merkle.json (default: auto)")
    p_verify.add_argument("--expected-root", help="Trusted root to verify against")

    p_diff = sub.add_parser("diff", help="Compare two files, directories or manifests")
    p_diff.add_argument("file_a", help="First file")
    p_diff.add_argument("file_b", help="Second file")

    p_info = sub.add_parser("info", help="Show manifest info")
    p_info.add_argument("manifest", help=".merkle.json file")

    p_hf = sub.add_parser("hf-sign", help="Hash a HuggingFace Hub model from the local cache")
    p_hf.add_argument("repo_id", help="HF repo ID (e.g. bert-base-uncased)")
    p_hf.add_argument("--revision", help="Commit hash, branch or tag (default: main)")
    p_hf.add_argument("--cache-dir", help="HF hub cache directory")
    p_hf.add_argument("--output", "-o", help="Write the manifest to this path")
    return parser


COMMANDS = {
    "hash": cmd_hash,
    "sign": cmd_sign,
    "verify": cmd_verify,
    "diff": cmd_diff,
    "info": cmd_info,
    "hf-sign": cmd_hf_sign,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.chunk_size_given = args.chunk_size is not None
    if args.chunk_size is None:
        args.chunk_size = DEFAULT_CHUNK_SIZE
    elif args.chunk_size <= 0:
        print("error: --chunk-size must be positive", file=sys.stderr)
        return EXIT_ERROR

    try:
        return COMMANDS[args.command](args)
    except (CLIError, FileNotFoundError, IsADirectoryError, ValueError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
