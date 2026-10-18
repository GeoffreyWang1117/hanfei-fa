# Manifest format v2

Status: **draft for review** (targets merkle-weight-verify 0.3.0).
Format v1 is the format written by 0.2.x. Readers keep accepting v1 manifests
and verify them with the v1 rules; writers produce v2 by default.

This document defines how roots are computed, precisely enough for an
independent implementation (the Rust port `hanfei-fa` is one) to reproduce
every root. `tests/vectors/format_v2.json` holds test vectors computed by a
reference implementation written from this document
(`tests/reference_v2.py`), not by the library itself.

## 1. Why a new format

The v1 construction has four weaknesses, each reproducible with v0.2.x:

| v1 property | Consequence |
|---|---|
| A leaf is `H(chunk)` and a node is `H(hex(left) ‖ hex(right))`, with no domain separation | A chunk whose bytes are the ASCII text `hex(A) ‖ hex(B)` hashes to the parent of `A` and `B` (second preimage across tree levels). |
| A level with an odd number of nodes pairs its last node with itself | `[A, B, C]` and `[A, B, C, C]` have the same root, so the root does not determine the number of chunks. |
| Layer and model roots hash the concatenated child roots in sorted-name order | Renaming a tensor without changing the sort order keeps the root; layer grouping is a heuristic that is baked into the root. |
| The tensor's dtype and shape are not hashed; the manifest does not record the hash algorithm | Reinterpreting bytes as another dtype or shape keeps the root; a BLAKE3 manifest checked with SHA-256 reports tampering. |

v2 fixes all four and adds inclusion proofs for single tensors and single
chunks against the model root.

## 2. Notation

- `H(x)`: the manifest's hash function applied to the byte string `x`,
  returning the raw digest (not hex).
- `‖`: byte concatenation.
- `u8(n)`, `u32(n)`, `u64(n)`: `n` as an unsigned big-endian integer of 1, 4, 8 bytes.
- `str(s)`: `u32(len(b)) ‖ b` where `b` is the UTF-8 encoding of `s`.
- Digests are written in manifests and APIs as lowercase hexadecimal.

Hash functions (`algorithm` field):

| name | function | digest |
|---|---|---|
| `sha256` (default) | SHA-256 | 32 bytes |
| `sha512` | SHA-512 | 64 bytes |
| `sha3_256` | SHA3-256 | 32 bytes |
| `blake2b` | BLAKE2b, 64-byte digest, no key | 64 bytes |
| `blake3` | BLAKE3, default mode | 32 bytes |

Domain tags, one byte prefixed to every hash input:

| tag | used for |
|---|---|
| `0x00` | chunk leaf |
| `0x01` | interior node of any Merkle tree |
| `0x02` | tensor root |
| `0x03` | layer digest (index only, not part of the model root) |
| `0x04` | model root |

## 3. Merkle tree hash

`MTH(L)` for a non-empty list `L = [x_0, ..., x_{n-1}]` of digests is the
RFC 6962 (§2.1) Merkle Tree Hash over already-hashed leaves:

```
MTH([x])  = x
MTH(L)    = H(0x01 ‖ MTH(L[0:k]) ‖ MTH(L[k:n]))   where k is the largest power of two < n
```

Equivalently, build the tree bottom-up: pair adjacent nodes left to right;
when a level has an odd number of nodes, carry its last node up unchanged.
No node is ever paired with itself.

## 4. Roots

### 4.1 Chunks

A tensor's bytes `d` (length `n`) are split into chunks of `c` bytes, the
last one possibly shorter: `d[0:c], d[c:2c], ...`. A tensor with `n = 0` has
exactly one chunk, the empty string. So the number of chunks is
`max(1, ceil(n / c))`.

```
leaf_i     = H(0x00 ‖ chunk_i)
chunk_root = MTH([leaf_0, ..., leaf_{m-1}])
```

`chunk_root` equals the RFC 6962 Merkle Tree Hash of the chunk sequence.

### 4.2 Tensor bytes

A tensor is hashed as its elements in C (row-major) order, each element in
little-endian byte order, which is exactly the byte range safetensors stores.
Hashing never decodes values, so every dtype is supported.

### 4.3 Tensor root

```
meta        = u8(0)                                         for opaque byte blobs
            | u8(1) ‖ str(dtype) ‖ u32(ndim) ‖ u64(dim_0) ‖ ... ‖ u64(dim_{ndim-1})
tensor_root = H(0x02 ‖ str(name) ‖ meta ‖ u64(n) ‖ u64(c) ‖ chunk_root)
```

`dtype` uses the safetensors dtype codes: `F64 F32 F16 BF16 I64 I32 I16 I8
U64 U32 U16 U8 BOOL F8_E4M3 F8_E5M2 C64 C128`. A scalar has `ndim = 0`.
`name` is the full tensor name, e.g. `model.layers.0.self_attn.q_proj.weight`.

### 4.4 Model root

Sort the tensors by name, comparing the UTF-8 encodings bytewise. With
`P >= 1` tensors and tensor roots `t_0, ..., t_{P-1}` in that order:

```
model_root = H(0x04 ‖ u64(P) ‖ MTH([t_0, ..., t_{P-1}]))
```

Tensor names must be unique. Layer grouping does not enter the model root.

### 4.5 Layer digests (index, not committed)

For fast diffs, implementations group tensors into layers (by name, e.g.
`model.layers.7`) and compare per-layer digests before descending. With the
tensors of one layer sorted by name:

```
layer_digest = H(0x03 ‖ str(layer_name) ‖ u64(m) ‖ t_0 ‖ ... ‖ t_{m-1})
```

Layer digests are recomputed by the reader from the tensor roots and are never
trusted from a file. Two implementations that group differently still agree
on every model root.

### 4.6 Single files

`merkle-verify hash/sign` on a file that is not a model use the chunk root of
§4.1 over the whole file (`n` = file size).

## 5. Proofs

### 5.1 Chunk inclusion

For leaf index `i` of a tree with `m` leaves, the audit path is the RFC 9162
(§2.1.3.1) inclusion path: the sibling digests from the leaf up to the root,
skipping levels where the node is carried up. A verifier checks it with the
RFC 9162 §2.1.3.2 algorithm using `i` and `m`, not with left/right flags taken
from the proof.

### 5.2 Tensor inclusion

The tensor-level tree of §4.4 is an MTH over tensor roots, so the same audit
path proves that tensor `j` (in sorted order, of `P`) is committed by the
model root: recompute `t_j` from its name, metadata and `chunk_root`, fold the
path to `MTH`, then check `H(0x04 ‖ u64(P) ‖ MTH) == model_root`.

A chunk proof plus a tensor proof show that one chunk's bytes belong to a model
without revealing any other tensor.

## 6. Manifest JSON

```json
{
  "format": "merkle-weight-verify",
  "format_version": 2,
  "algorithm": "sha256",
  "model_name": "Llama-3.1-8B",
  "model_root": "<hex>",
  "num_tensors": 291,
  "created_by": "merkle-weight-verify 0.3.0",
  "tensors": {
    "model.embed_tokens.weight": {
      "dtype": "BF16",
      "shape": [128256, 4096],
      "num_bytes": 1050673152,
      "chunk_size": 16384,
      "num_chunks": 64128,
      "chunk_root": "<hex>",
      "tensor_root": "<hex>",
      "chunk_hashes": ["<hex>", "..."]
    }
  }
}
```

- `dtype` and `shape` are `null` for opaque byte blobs.
- `chunk_hashes` is optional. Without it a manifest is *compact*: it still
  verifies a model and localises changes to tensors, but not to chunks.
- On load, a reader recomputes `chunk_root` from `chunk_hashes` (when present),
  every `tensor_root`, and the `model_root`, and rejects the manifest if any
  stored value disagrees. A manifest is never trusted for a value it can derive.
- `model_name` and `created_by` are labels; they do not enter any root.

A manifest without `format_version` is v1 (the 0.2.x layout with
`layer_trees`) and is verified with the v1 rules.

## 7. Compatibility

- v1 roots are unchanged and pinned by `tests/vectors/format_v1.json`
  (generated by the 0.2.0 release).
- Diffing a v1 tree against a v2 tree, or trees with different algorithms or
  chunk sizes, is an error rather than a report that everything changed.
- `MerkleTree`, `ModelMerkleTree`, `build_model_merkle_tree` keep their
  signatures; new keyword arguments `format_version=` and `algorithm=`
  select the format. The default is v2 and SHA-256.
