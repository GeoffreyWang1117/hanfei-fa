"""
merkle-verify CLI: command-line tool for model weight integrity verification.

Usage:
    merkle-verify hash <file>              Compute Merkle root hash of a file
    merkle-verify sign <file>              Build tree + write .merkle.json sidecar
    merkle-verify verify <file>            Verify file against its .merkle.json
    merkle-verify diff <file_a> <file_b>   Compare two files (layer/param/chunk diff)
    merkle-verify info <manifest>          Show info about a .merkle.json manifest
"""

import argparse
import json
import sys
import time
from pathlib import Path

from merkle_verify.merkle_tree import build_file_merkle_tree, ModelMerkleTree
from merkle_verify.chunking import DEFAULT_CHUNK_SIZE


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


def cmd_sign(args):
    """Build Merkle tree for a safetensors/model file and write sidecar."""
    filepath = args.file
    suffix = Path(filepath).suffix.lower()

    if suffix == ".safetensors":
        try:
            from merkle_verify.safetensors_adapter import sign
            t0 = time.time()
            tree = sign(filepath, chunk_size=args.chunk_size, parallel=True)
            elapsed = time.time() - t0
            sidecar = str(Path(filepath).with_suffix(".merkle.json"))
            print(f"root: {tree.model_root}")
            print(f"manifest: {sidecar}")
            if args.verbose:
                n_params = sum(
                    len(lt.param_trees) for lt in tree.layer_trees.values()
                )
                print(f"  tensors: {n_params}")
                print(f"  layers: {len(tree.layer_trees)}")
                print(f"  time: {elapsed:.3f}s")
        except ImportError:
            print("Error: safetensors not installed. Run: pip install hanfei-fa[safetensors]",
                  file=sys.stderr)
            sys.exit(1)
    elif suffix in (".pt", ".pth"):
        try:
            from merkle_verify.pytorch_adapter import merkle_save
            import torch
            t0 = time.time()
            sd = torch.load(filepath, map_location="cpu", weights_only=True)
            tree = merkle_save(sd, filepath, chunk_size=args.chunk_size)
            elapsed = time.time() - t0
            sidecar = str(Path(filepath).with_suffix(".merkle.json"))
            print(f"root: {tree.model_root}")
            print(f"manifest: {sidecar}")
            if args.verbose:
                print(f"  time: {elapsed:.3f}s")
        except ImportError:
            print("Error: torch not installed. Run: pip install hanfei-fa[torch]",
                  file=sys.stderr)
            sys.exit(1)
    else:
        # Generic file: just compute hash and write sidecar
        t0 = time.time()
        tree = build_file_merkle_tree(filepath, chunk_size=args.chunk_size)
        elapsed = time.time() - t0
        sidecar = str(Path(filepath).with_suffix(".merkle.json"))
        data = {
            "root_hash": tree.root_hash,
            "num_chunks": tree.num_chunks,
            "chunk_size": tree.chunk_size,
            "chunk_hashes": tree.chunk_hashes,
        }
        with open(sidecar, "w") as f:
            json.dump(data, f, indent=2)
        print(f"root: {tree.root_hash}")
        print(f"manifest: {sidecar}")
        if args.verbose:
            print(f"  chunks: {tree.num_chunks}")
            print(f"  time: {elapsed:.3f}s")


def cmd_verify(args):
    """Verify a file against its Merkle manifest."""
    filepath = args.file
    suffix = Path(filepath).suffix.lower()

    if suffix == ".safetensors":
        try:
            from merkle_verify.safetensors_adapter import verify
            t0 = time.time()
            is_valid, details = verify(filepath, manifest=args.manifest)
            elapsed = time.time() - t0
        except ImportError:
            print("Error: safetensors not installed.", file=sys.stderr)
            sys.exit(1)
    elif suffix in (".pt", ".pth"):
        try:
            from merkle_verify.pytorch_adapter import verify_checkpoint
            t0 = time.time()
            is_valid, details = verify_checkpoint(filepath, manifest=args.manifest)
            elapsed = time.time() - t0
        except ImportError:
            print("Error: torch not installed.", file=sys.stderr)
            sys.exit(1)
    else:
        # Generic file
        manifest_path = args.manifest or str(Path(filepath).with_suffix(".merkle.json"))
        try:
            with open(manifest_path) as f:
                stored = json.load(f)
        except FileNotFoundError:
            print(f"FAIL: manifest not found: {manifest_path}", file=sys.stderr)
            sys.exit(1)

        t0 = time.time()
        tree = build_file_merkle_tree(filepath, chunk_size=stored.get("chunk_size", DEFAULT_CHUNK_SIZE))
        elapsed = time.time() - t0
        is_valid = tree.root_hash == stored["root_hash"]
        details = {
            "stored_root": stored["root_hash"],
            "computed_root": tree.root_hash,
        }

    if is_valid:
        print(f"OK: {filepath}")
    else:
        print(f"FAIL: {filepath}")
        if "error" in details:
            print(f"  error: {details['error']}")
        elif "changed_params" in details:
            print(f"  changed params: {details['changed_params']}")
        else:
            print(f"  stored:   {details.get('stored_root', '?')}")
            print(f"  computed: {details.get('computed_root', '?')}")

    if args.verbose:
        print(f"  time: {elapsed:.3f}s")

    sys.exit(0 if is_valid else 1)


def cmd_diff(args):
    """Compare two files."""
    suffix_a = Path(args.file_a).suffix.lower()

    if suffix_a == ".safetensors":
        try:
            from merkle_verify.safetensors_adapter import diff
            t0 = time.time()
            result = diff(args.file_a, args.file_b, chunk_size=args.chunk_size)
            elapsed = time.time() - t0
        except ImportError:
            print("Error: safetensors not installed.", file=sys.stderr)
            sys.exit(1)
    elif suffix_a in (".pt", ".pth"):
        try:
            from merkle_verify.pytorch_adapter import diff_checkpoints
            t0 = time.time()
            result = diff_checkpoints(args.file_a, args.file_b, chunk_size=args.chunk_size)
            elapsed = time.time() - t0
        except ImportError:
            print("Error: torch not installed.", file=sys.stderr)
            sys.exit(1)
    elif suffix_a == ".json":
        # Compare two .merkle.json manifests
        t0 = time.time()
        tree_a = ModelMerkleTree.from_json(args.file_a)
        tree_b = ModelMerkleTree.from_json(args.file_b)
        from merkle_verify.comparison import compare_model_trees
        result = compare_model_trees(tree_a, tree_b, detailed=True)
        elapsed = time.time() - t0
    else:
        # Generic binary diff
        t0 = time.time()
        tree_a = build_file_merkle_tree(args.file_a, chunk_size=args.chunk_size)
        tree_b = build_file_merkle_tree(args.file_b, chunk_size=args.chunk_size)
        changed, comparisons = tree_a.diff_tree(tree_b)
        elapsed = time.time() - t0
        total = max(tree_a.num_chunks, tree_b.num_chunks)
        pct = (len(changed) / total * 100) if total > 0 else 0
        print(f"{len(changed)}/{total} chunks changed ({pct:.1f}%), {comparisons} hash comparisons")
        if args.verbose:
            print(f"  time: {elapsed:.3f}s")
            if changed:
                print(f"  changed indices: {changed[:20]}{'...' if len(changed) > 20 else ''}")
        return

    # Model-aware diff result (dict)
    print(f"changed layers: {result['changed_layers']}")
    print(f"changed params: {len(result['changed_params'])} / {len(result['changed_params']) + len(result['unchanged_params'])}")
    print(f"change: {result['change_percentage']:.1f}%")
    print(f"hash comparisons: {result['hash_comparisons']}")
    if args.verbose:
        print(f"  time: {elapsed:.3f}s")
        if result['changed_params']:
            print(f"  changed: {result['changed_params'][:10]}")


def cmd_info(args):
    """Show info about a .merkle.json manifest."""
    try:
        tree = ModelMerkleTree.from_json(args.manifest)
    except Exception:
        # Try as flat tree json
        with open(args.manifest) as f:
            data = json.load(f)
        print(f"root: {data.get('root_hash', data.get('model_root', '?'))}")
        print(f"chunks: {data.get('num_chunks', '?')}")
        print(f"chunk_size: {data.get('chunk_size', '?')}")
        return

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


def cmd_hf_sign(args):
    """Sign a HuggingFace Hub model from local cache."""
    try:
        from merkle_verify.safetensors_adapter import from_hf_repo
    except ImportError:
        print("Error: safetensors not installed.", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    try:
        tree = from_hf_repo(
            args.repo_id,
            revision=getattr(args, 'revision', None),
            cache_dir=getattr(args, 'cache_dir', None),
            chunk_size=args.chunk_size,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - t0
    n_params = sum(len(lt.param_trees) for lt in tree.layer_trees.values())
    print(f"root: {tree.model_root}")
    print(f"model: {tree.model_name}")
    print(f"layers: {len(tree.layer_trees)}, params: {n_params}")
    if args.verbose:
        print(f"  time: {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        prog="merkle-verify",
        description="Model weight integrity verification via hierarchical Merkle trees",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in bytes (default: {DEFAULT_CHUNK_SIZE})"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_hash = sub.add_parser("hash", help="Compute Merkle root hash")
    p_hash.add_argument("file", help="File to hash")

    p_sign = sub.add_parser("sign", help="Build tree + write .merkle.json")
    p_sign.add_argument("file", help="File to sign")

    p_verify = sub.add_parser("verify", help="Verify file against manifest")
    p_verify.add_argument("file", help="File to verify")
    p_verify.add_argument("--manifest", help="Path to .merkle.json (default: auto)")

    p_diff = sub.add_parser("diff", help="Compare two files")
    p_diff.add_argument("file_a", help="First file")
    p_diff.add_argument("file_b", help="Second file")

    p_info = sub.add_parser("info", help="Show manifest info")
    p_info.add_argument("manifest", help=".merkle.json file")

    p_hf = sub.add_parser("hf-sign", help="Sign a HuggingFace Hub model from local cache")
    p_hf.add_argument("repo_id", help="HF repo ID (e.g. bert-base-uncased)")
    p_hf.add_argument("--revision", help="Git revision")
    p_hf.add_argument("--cache-dir", help="HF cache directory")

    args = parser.parse_args()

    commands = {
        "hash": cmd_hash,
        "sign": cmd_sign,
        "verify": cmd_verify,
        "diff": cmd_diff,
        "info": cmd_info,
        "hf-sign": cmd_hf_sign,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
