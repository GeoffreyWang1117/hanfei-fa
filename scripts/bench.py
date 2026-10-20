#!/usr/bin/env python3
"""Reproducible benchmark: tree build time and diff / sync cost on real models.

For each safetensors model (a file or a model directory):

1. Build the tree from disk with format v1 and v2 (best of --repeat runs,
   page cache warm after the first run).
2. Simulate typical updates in memory and diff against the base tree:
   - one-tensor:   one MLP weight matrix rewritten
   - lora-merged:  every attention q/v projection rewritten (a merged LoRA update)
   - sparse-1pct:  1% of all chunks rewritten at random
   - full:         every tensor rewritten
   Reported: changed tensors, hash comparisons, bytes an incremental sync
   would transfer (changed chunks) and the saving against a full download.

Usage:
    python scripts/bench.py PATH [PATH ...] [--chunk-size 16384] [--json out.json]

Nothing is written next to the models.
"""

import argparse
import json
import platform
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from merkle_verify import RawTensor, __version__, build_model_merkle_tree, compare_model_trees  # noqa: E402
from merkle_verify._safetensors_io import read_header, read_tensor_bytes  # noqa: E402
from merkle_verify.safetensors_adapter import _shard_files, build_sharded_tree, build_tree  # noqa: E402

QV = re.compile(r"(q_proj|v_proj|\.q\.|\.v\.|c_attn)")
MLP = re.compile(r"(mlp|ffn|fc|down_proj|up_proj)")


def load_tensors(path: Path):
    files = _shard_files(str(path)) if path.is_dir() else [str(path)]
    tensors = {}
    for f in files:
        header, _ = read_header(f)
        for name, info in header.items():
            tensors[name] = RawTensor(read_tensor_bytes(f, info), info.dtype, info.shape)
    return tensors


def build_from_disk(path: Path, chunk_size: int, version: int):
    if path.is_dir():
        return build_sharded_tree(str(path), chunk_size=chunk_size, format_version=version)
    return build_tree(str(path), chunk_size=chunk_size, format_version=version)


def scenario(tensors, name: str, chunk_size: int, rng: random.Random):
    names = sorted(tensors)
    if name == "one-tensor":
        mlp = [n for n in names if MLP.search(n) and len(tensors[n].shape) == 2] or names
        targets = {mlp[len(mlp) // 2]: "dense"}
    elif name == "lora-merged":
        targets = {n: "dense" for n in names if QV.search(n) and len(tensors[n].shape) == 2}
    elif name == "full":
        targets = {n: "dense" for n in names}
    elif name == "sparse-1pct":
        targets = {n: "sparse" for n in names}
    else:
        raise ValueError(name)

    out = dict(tensors)
    for n, kind in targets.items():
        t = tensors[n]
        raw = bytearray(t.data)
        if kind == "dense":
            # touch one byte in every chunk: every chunk of the tensor changes
            for off in range(0, len(raw), chunk_size):
                raw[off] ^= 0x5A
        else:
            n_chunks = max(1, (len(raw) + chunk_size - 1) // chunk_size)
            for c in range(n_chunks):
                if rng.random() < 0.01 and raw:
                    raw[min(c * chunk_size, len(raw) - 1)] ^= 0x5A
        out[n] = RawTensor(bytes(raw), t.dtype, t.shape)
    return out, len(targets)


def label(path: Path) -> str:
    """HF cache paths (models--org--name/snapshots/...) are shown as org/name."""
    for part in path.resolve().parts:
        if part.startswith("models--"):
            return part[len("models--"):].replace("--", "/")
    return path.name if path.is_file() else path.resolve().name


def best_of(fn, repeat):
    best, result = None, None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        best = dt if best is None or dt < best else best
    return best, result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--chunk-size", type=int, default=16384)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write raw results here")
    args = ap.parse_args()

    results = {
        "version": __version__,
        "python": platform.python_version(),
        "machine": f"{platform.processor() or platform.machine()} ({platform.system()})",
        "chunk_size": args.chunk_size,
        "models": [],
    }
    for p in args.paths:
        path = Path(p)
        tensors = load_tensors(path)
        size = sum(len(t.data) for t in tensors.values())
        model = {"path": label(path),
                 "bytes": size, "tensors": len(tensors), "build": {}, "scenarios": {}}
        for version in (1, 2):
            secs, tree = best_of(lambda: build_from_disk(path, args.chunk_size, version), args.repeat)
            chunks = sum(t.num_chunks for _, t in tree.iter_params())
            model["build"][f"v{version}"] = {"seconds": round(secs, 3),
                                             "gb_per_s": round(size / secs / 1e9, 2),
                                             "chunks": chunks}
        base = build_model_merkle_tree(tensors, chunk_size=args.chunk_size)
        rng = random.Random(args.seed)
        for name in ("one-tensor", "lora-merged", "sparse-1pct", "full"):
            modified, n_targets = scenario(tensors, name, args.chunk_size, rng)
            new = build_model_merkle_tree(modified, chunk_size=args.chunk_size)
            t0 = time.perf_counter()
            diff = compare_model_trees(base, new, detailed=True)
            dt = time.perf_counter() - t0
            changed_chunks = sum(len(v) for v in diff["chunk_changes"].values())
            sync = min(changed_chunks * args.chunk_size, size)
            model["scenarios"][name] = {
                "changed_tensors": len(diff["changed_params"]),
                "changed_chunks": changed_chunks,
                "hash_comparisons": diff["hash_comparisons"],
                "diff_ms": round(dt * 1000, 2),
                "sync_bytes": sync,
                "saving_pct": round(100 * (1 - sync / size), 1),
            }
        results["models"].append(model)

    print(f"merkle-weight-verify {results['version']}, Python {results['python']}, "
          f"{results['machine']}, chunk size {args.chunk_size}\n")
    print("| Model | Size | Tensors | Build v1 | Build v2 |")
    print("|---|---|---|---|---|")
    for m in results["models"]:
        b1, b2 = m["build"]["v1"], m["build"]["v2"]
        print(f"| {m['path']} | {m['bytes'] / 1e6:.0f} MB | {m['tensors']} | "
              f"{b1['seconds']}s ({b1['gb_per_s']} GB/s) | {b2['seconds']}s ({b2['gb_per_s']} GB/s) |")
    print()
    print("| Model | Update | Changed tensors | Hash comparisons | Diff | Sync | Saving |")
    print("|---|---|---|---|---|---|---|")
    for m in results["models"]:
        for name, s in m["scenarios"].items():
            print(f"| {m['path']} | {name} | {s['changed_tensors']} | {s['hash_comparisons']} | "
                  f"{s['diff_ms']} ms | {s['sync_bytes'] / 1e6:.1f} MB | {s['saving_pct']}% |")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
