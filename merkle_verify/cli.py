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
    merkle-verify hf-check <repo_id>             Verify a cached HuggingFace model
    merkle-verify repair <file|dir> --from SRC   Re-fetch only the damaged chunks from a good copy

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
from merkle_verify.format import DEFAULT_FORMAT_VERSION, FORMAT_V1, SUPPORTED_FORMATS, parse_root
from merkle_verify.hasher import HashAlgorithm, resolve_algorithm
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


def _fmt_kwargs(args) -> dict:
    """format_version / algorithm for building new trees."""
    return {
        "format_version": args.format_version or DEFAULT_FORMAT_VERSION,
        "algorithm": args.algorithm,
    }


def _print_model_summary(tree: ModelMerkleTree, elapsed: float, verbose: bool) -> None:
    print(f"root: {tree.model_root}")
    if verbose:
        print(f"  format: v{tree.format_version}, {tree.algorithm.value}")
        n_params = sum(len(lt.param_trees) for lt in tree.layer_trees.values())
        print(f"  tensors: {n_params}")
        print(f"  layers: {len(tree.layer_trees)}")
        print(f"  time: {elapsed:.3f}s")


def cmd_hash(args):
    """Compute Merkle root hash of a file."""
    t0 = time.time()
    tree = build_file_merkle_tree(args.file, chunk_size=args.chunk_size, **_fmt_kwargs(args))
    elapsed = time.time() - t0

    print(f"{tree.root_hash}  {args.file}")
    if args.verbose:
        print(f"  format: v{tree.version}, {tree.algorithm.value}")
        print(f"  chunks: {tree.num_chunks}")
        print(f"  chunk_size: {tree.chunk_size}")
        print(f"  time: {elapsed:.3f}s")
    return EXIT_OK


def cmd_sign(args):
    """Build a Merkle tree for a model file or directory and write the manifest."""
    path = args.file
    kind = _kind(path)
    t0 = time.time()
    kw = dict(chunk_size=args.chunk_size, **_fmt_kwargs(args))
    if args.compact:
        if kw["format_version"] == FORMAT_V1:
            raise CLIError("--compact needs format version 2")
        kw["include_chunks"] = False

    if kind == "dir":
        from merkle_verify.safetensors_adapter import sign_sharded, SHARDED_MANIFEST_FILENAME
        tree = sign_sharded(path, **kw)
        manifest = str(Path(path) / SHARDED_MANIFEST_FILENAME)
    elif kind == "safetensors":
        from merkle_verify.safetensors_adapter import sign, _sidecar_path
        tree = sign(path, **kw)
        manifest = _sidecar_path(path)
    elif kind == "torch":
        _need_torch()
        from merkle_verify.pytorch_adapter import sign_checkpoint, _sidecar_path
        tree = sign_checkpoint(path, **kw)
        manifest = _sidecar_path(path)
    else:
        kw.pop("include_chunks", None)
        file_tree = build_file_merkle_tree(path, **kw)
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


def _verify_generic(path: str, manifest_path: str, expected_root, args):
    try:
        with open(manifest_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = None
        if expected_root is None:
            return False, {"error": f"manifest not found: {manifest_path}"}

    stored = None
    if data is not None:
        if "root_hash" not in data:
            raise CLIError(f"{manifest_path}: not a single-file manifest")
        try:
            stored = MerkleTree.from_dict(data, algorithm=args.algorithm)
        except (ValueError, KeyError, TypeError) as e:
            if expected_root is None:
                return False, {"error": f"invalid manifest {manifest_path}: {e}"}

    if expected_root is None:
        version, alg, chunk_size = stored.version, stored.algorithm, stored.chunk_size
    else:
        prefix = parse_root(expected_root)[0]
        version = args.format_version or (stored.version if stored else DEFAULT_FORMAT_VERSION)
        alg = resolve_algorithm(args.algorithm or prefix or (stored.algorithm if stored else None))
        chunk_size = args.chunk_size if args.chunk_size_given else (
            stored.chunk_size if stored else DEFAULT_CHUNK_SIZE)

    tree = build_file_merkle_tree(path, chunk_size=chunk_size, format_version=version, algorithm=alg)
    reference = parse_root(expected_root)[1] if expected_root is not None else stored.root_hash
    details = {"computed_root": tree.root_hash}
    if expected_root is not None:
        details["expected_root"] = expected_root
    else:
        details["stored_root"] = stored.root_hash
    comparable = stored is not None and stored.has_chunk_hashes and \
        (stored.version, stored.algorithm, stored.chunk_size) == (version, alg, chunk_size)
    if tree.root_hash != reference and comparable:
        changed, _ = tree.diff_tree(stored)
        details["changed_chunks"] = changed
    return tree.root_hash == reference, details


def cmd_verify(args):
    """Re-hash a file or model directory and check it."""
    path = args.file
    kind = _kind(path)
    expected = args.expected_root
    kw = dict(
        manifest=args.manifest, expected_root=expected,
        chunk_size=args.chunk_size if args.chunk_size_given else None,
        format_version=args.format_version, algorithm=args.algorithm,
    )
    t0 = time.time()

    if kind == "dir":
        from merkle_verify.safetensors_adapter import verify_sharded
        is_valid, details = verify_sharded(path, **kw)
    elif kind == "safetensors":
        from merkle_verify.safetensors_adapter import verify
        is_valid, details = verify(path, **kw)
    elif kind == "torch":
        _need_torch()
        from merkle_verify.pytorch_adapter import verify_checkpoint
        is_valid, details = verify_checkpoint(path, **kw)
    else:
        manifest_path = args.manifest or str(Path(path).with_suffix(".merkle.json"))
        is_valid, details = _verify_generic(path, manifest_path, expected, args)
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
    if details.get("manifest_ignored"):
        print(f"  note: {details['manifest_ignored']}")

    if args.verbose:
        print(f"  time: {elapsed:.3f}s")
    return EXIT_OK if is_valid else EXIT_FAIL


def _load_json_manifest(path: str):
    with open(path) as f:
        data = json.load(f)
    if "layer_trees" in data or "tensors" in data:
        return ModelMerkleTree.from_dict(data)
    if "root_hash" in data:
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
    kw = dict(chunk_size=args.chunk_size, **_fmt_kwargs(args))

    if kind_a == "dir":
        from merkle_verify.safetensors_adapter import build_sharded_tree
        result = compare_model_trees(
            build_sharded_tree(args.file_a, **kw),
            build_sharded_tree(args.file_b, **kw),
        )
    elif kind_a == "safetensors":
        from merkle_verify.safetensors_adapter import diff
        result = diff(args.file_a, args.file_b, **kw)
    elif kind_a == "torch":
        _need_torch()
        from merkle_verify.pytorch_adapter import diff_checkpoints
        result = diff_checkpoints(args.file_a, args.file_b, **kw)
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
        tree_a = build_file_merkle_tree(args.file_a, **kw)
        tree_b = build_file_merkle_tree(args.file_b, **kw)
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


def _format_label(version: int, algorithm: HashAlgorithm) -> str:
    if version == FORMAT_V1:
        return f"v1 (algorithm not recorded; assumed {algorithm.value})"
    return f"v{version}, {algorithm.value}"


def cmd_info(args):
    """Show info about a .merkle.json manifest."""
    tree = _load_json_manifest(args.manifest)
    if isinstance(tree, MerkleTree):
        print(f"format: {_format_label(tree.version, tree.algorithm)}")
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
    print(f"format: {_format_label(tree.format_version, tree.algorithm)}")
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
        **_fmt_kwargs(args),
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


def cmd_hf_check(args):
    """Verify a model in the HuggingFace cache against a published manifest or a trusted root."""
    from merkle_verify.safetensors_adapter import (
        SHARDED_MANIFEST_FILENAME, _resolve_hf_snapshot, verify_sharded,
    )

    snapshot = _resolve_hf_snapshot(args.repo_id, args.revision, args.cache_dir)
    commit = Path(snapshot).name
    manifest = args.manifest
    published = Path(snapshot) / SHARDED_MANIFEST_FILENAME
    if manifest is None and published.exists():
        manifest = str(published)
        print(f"manifest: {SHARDED_MANIFEST_FILENAME} published in {args.repo_id}@{commit[:12]}")
    if manifest is None and args.expected_root is None:
        raise CLIError(
            f"{args.repo_id}@{commit[:12]} publishes no {SHARDED_MANIFEST_FILENAME}; "
            "pass --manifest or --expected-root"
        )

    t0 = time.time()
    is_valid, details = verify_sharded(
        snapshot, manifest=manifest or str(published), expected_root=args.expected_root)
    if is_valid:
        print(f"OK: {args.repo_id}@{commit[:12]}")
    else:
        print(f"FAIL: {args.repo_id}@{commit[:12]}")
        if "error" in details:
            print(f"  error: {details['error']}")
        if details.get("changed_params"):
            print(f"  changed params: {details['changed_params'][:10]}")
        if manifest is not None and "error" not in details:
            print(f"  repair: merkle-verify repair {snapshot} --from hf://{args.repo_id}@{commit}"
                  + ("" if manifest == str(published) else f" --manifest {manifest}"))
    if args.verbose:
        print(f"  snapshot: {snapshot}")
        print(f"  time: {time.time() - t0:.3f}s")
    return EXIT_OK if is_valid else EXIT_FAIL


def cmd_repair(args):
    """Re-fetch only damaged chunks from a good copy, checking each against the manifest."""
    from merkle_verify.repair import RepairError, repair

    t0 = time.time()
    try:
        report = repair(
            args.file, args.source, manifest=args.manifest, expected_root=args.expected_root,
            output=args.output, dry_run=args.dry_run,
            max_request_bytes=int(args.max_request_mb * 1024 * 1024),
        )
    except RepairError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_FAIL
    print(report.summary())
    if args.verbose:
        for name, idx in report.bad_chunks[:20]:
            print(f"  {name} chunk {idx if idx >= 0 else '(whole tensor)'}")
        if len(report.bad_chunks) > 20:
            print(f"  ... {len(report.bad_chunks) - 20} more")
        print(f"  time: {time.time() - t0:.3f}s")
    if args.dry_run:
        print("dry run: nothing was written")
        return EXIT_OK
    target = args.output or args.file
    if report.verified:
        print(f"OK: {target}")
        return EXIT_OK
    print(f"FAIL: {target} does not reproduce the manifest root after repair")
    return EXIT_FAIL


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
    parser.add_argument(
        "--format-version", type=int, choices=SUPPORTED_FORMATS, default=None,
        help=f"Manifest format for new trees (default: {DEFAULT_FORMAT_VERSION}; "
             "verify follows the manifest)"
    )
    parser.add_argument(
        "--algorithm", choices=[a.value for a in HashAlgorithm], default=None,
        help="Hash algorithm for new trees (default: sha256; verify follows the manifest)"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_hash = sub.add_parser("hash", help="Compute Merkle root hash")
    p_hash.add_argument("file", help="File to hash")

    p_sign = sub.add_parser("sign", help="Build tree + write .merkle.json (weights are not modified)")
    p_sign.add_argument("file", help="File or model directory to sign")
    p_sign.add_argument("--compact", action="store_true",
                        help="Omit per-chunk hashes (v2): smaller manifest, tensor-level diffs only")

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
    p_hfc = sub.add_parser("hf-check", help="Verify a cached HuggingFace model")
    p_hfc.add_argument("repo_id", help="HF repo ID (e.g. Qwen/Qwen2.5-0.5B)")
    p_hfc.add_argument("--revision", help="Commit hash, branch or tag (default: main)")
    p_hfc.add_argument("--cache-dir", help="HF hub cache directory")
    p_hfc.add_argument("--manifest", help="Trusted manifest (default: model.merkle.json published in the repo)")
    p_hfc.add_argument("--expected-root", help="Trusted root to verify against")

    p_rep = sub.add_parser("repair", help="Re-fetch only damaged chunks from a good copy")
    p_rep.add_argument("file", help="Damaged file or model directory")
    p_rep.add_argument("--from", dest="source", required=True,
                       help="Good copy: path, directory, http(s):// URL, or hf://org/name[@revision]")
    p_rep.add_argument("--manifest", help="Trusted manifest (default: the sidecar)")
    p_rep.add_argument("--expected-root", help="Refuse a manifest whose root differs from this")
    p_rep.add_argument("--output", "-o", help="Write the repaired copy here (files only)")
    p_rep.add_argument("--dry-run", action="store_true", help="Only report what would be fetched")
    p_rep.add_argument("--max-request-mb", type=float, default=8.0,
                       help="Largest merged range request (default: 8)")
    return parser


COMMANDS = {
    "hash": cmd_hash,
    "sign": cmd_sign,
    "verify": cmd_verify,
    "diff": cmd_diff,
    "info": cmd_info,
    "hf-sign": cmd_hf_sign,
    "hf-check": cmd_hf_check,
    "repair": cmd_repair,
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
