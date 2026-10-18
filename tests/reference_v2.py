"""
Independent reference implementation of manifest format v2 (docs/MANIFEST_V2.md).

Written from the specification only, using hashlib and struct; it does not
import merkle_verify. The library must agree with it on every root, and the
committed vectors (tests/vectors/format_v2.json) are produced by it.

Regenerate the vectors with:
    python tests/reference_v2.py > tests/vectors/format_v2.json
"""

import hashlib
import json
import struct
import sys


def H(alg, data):
    if alg == "sha256":
        return hashlib.sha256(data).digest()
    if alg == "sha512":
        return hashlib.sha512(data).digest()
    if alg == "sha3_256":
        return hashlib.sha3_256(data).digest()
    if alg == "blake2b":
        return hashlib.blake2b(data, digest_size=64).digest()
    if alg == "blake3":
        import blake3
        return blake3.blake3(data).digest()
    raise ValueError(alg)


def u32(n):
    return struct.pack(">I", n)


def u64(n):
    return struct.pack(">Q", n)


def s(text):
    b = text.encode("utf-8")
    return u32(len(b)) + b


def mth(alg, leaves):
    """RFC 6962 recursive definition (largest power of two strictly below n)."""
    n = len(leaves)
    if n == 1:
        return leaves[0]
    k = 1
    while k * 2 < n:
        k *= 2
    return H(alg, b"\x01" + mth(alg, leaves[:k]) + mth(alg, leaves[k:]))


def chunks_of(data, c):
    if not data:
        return [b""]
    return [data[i:i + c] for i in range(0, len(data), c)]


def chunk_root(alg, data, c):
    return mth(alg, [H(alg, b"\x00" + ch) for ch in chunks_of(data, c)])


def meta(dtype, shape):
    if dtype is None:
        return b"\x00"
    return b"\x01" + s(dtype) + u32(len(shape)) + b"".join(u64(d) for d in shape)


def tensor_root(alg, name, data, c, dtype, shape):
    return H(alg, b"\x02" + s(name) + meta(dtype, shape) + u64(len(data)) + u64(c)
             + chunk_root(alg, data, c))


def model_root(alg, tensors, c):
    """tensors: {name: (data, dtype, shape)}"""
    names = sorted(tensors, key=lambda n: n.encode("utf-8"))
    roots = [tensor_root(alg, n, tensors[n][0], c, tensors[n][1], tensors[n][2]) for n in names]
    return H(alg, b"\x04" + u64(len(roots)) + mth(alg, roots))


# ---------------------------------------------------------------------------
# Vector cases (inputs are generated deterministically, so the file stays small)
# ---------------------------------------------------------------------------

def pattern(seed, n):
    return bytes((seed * 131 + i * 7) % 256 for i in range(n))


CASES = [
    {
        "id": "llama-like",
        "chunk_size": 64,
        "tensors": {
            "model.embed_tokens.weight": [1, 640, "BF16", [10, 32]],
            "model.layers.0.self_attn.q_proj.weight": [2, 512, "BF16", [16, 16]],
            "model.layers.0.input_layernorm.weight": [3, 32, "BF16", [16]],
            "model.layers.1.mlp.down_proj.weight": [4, 200, "F32", [5, 10]],
            "model.norm.weight": [5, 64, "F32", [16]],
            "lm_head.weight": [6, 65, "U8", [65]],
        },
    },
    {"id": "single-tensor", "chunk_size": 16384, "tensors": {"w": [7, 100, "F16", [50]]}},
    {"id": "empty-and-scalar", "chunk_size": 16,
     "tensors": {"a.empty": [8, 0, "F32", [0, 4]], "a.scalar": [9, 4, "F32", []]}},
    {"id": "opaque-blobs", "chunk_size": 32,
     "tensors": {"blob.one": [10, 33, None, None], "blob.two": [11, 31, None, None]}},
    {"id": "unicode-names", "chunk_size": 8,
     "tensors": {"é.weight": [12, 9, "I8", [9]], "z.weight": [13, 9, "I8", [9]],
                 "Z.weight": [14, 9, "I8", [9]]}},
    {"id": "exact-chunk-multiple", "chunk_size": 32,
     "tensors": {"x.w": [15, 128, "U8", [128]]}},
]


def build_vectors(algorithms=("sha256", "sha512", "sha3_256", "blake2b")):
    out = {
        "spec": "docs/MANIFEST_V2.md",
        "generated_by": "tests/reference_v2.py (independent of merkle_verify)",
        "cases": [],
        "chunk_trees": [],
    }
    for case in CASES:
        c = case["chunk_size"]
        tensors = {name: (pattern(seed, n), dt, None if sh is None else tuple(sh))
                   for name, (seed, n, dt, sh) in case["tensors"].items()}
        entry = {"id": case["id"], "chunk_size": c, "tensors": case["tensors"], "roots": {}}
        for alg in algorithms:
            entry["roots"][alg] = {
                "model_root": model_root(alg, tensors, c).hex(),
                "tensor_roots": {n: tensor_root(alg, n, d, c, dt, sh).hex()
                                 for n, (d, dt, sh) in tensors.items()},
                "chunk_roots": {n: chunk_root(alg, d, c).hex() for n, (d, _, _) in tensors.items()},
            }
        out["cases"].append(entry)
    for n_chunks in (1, 2, 3, 4, 5, 7, 8, 9, 16, 17):
        data = pattern(99, n_chunks * 4 - 1)  # last chunk is short
        out["chunk_trees"].append({
            "seed_len": [99, len(data)], "chunk_size": 4,
            "root_sha256": chunk_root("sha256", data, 4).hex(),
        })
    return out


if __name__ == "__main__":
    json.dump(build_vectors(), sys.stdout, indent=2, sort_keys=True, ensure_ascii=True)
    sys.stdout.write("\n")
