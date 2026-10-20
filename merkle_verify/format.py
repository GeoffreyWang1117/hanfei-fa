"""
Hashing rules for the manifest formats (see docs/MANIFEST_V2.md).

Format v1 (0.2.x) is defined by the legacy code paths in ``merkle_tree.py``
and is kept bit-for-bit. Format v2 is defined here:

* chunk leaves ``H(0x00 || chunk)``, interior nodes ``H(0x01 || left || right)``
  over raw digests, and RFC 6962 tree shape (an odd node is carried up, never
  paired with itself);
* each tensor root binds the tensor's name, dtype, shape, byte length and
  chunk size;
* the model root is an RFC 6962 tree over the tensor roots sorted by name,
  wrapped with the tensor count.
"""

import struct
from typing import List, Optional, Sequence, Tuple, Union

from merkle_verify.hasher import HashAlgorithm, compute_digest, hasher_factory, resolve_algorithm

FORMAT_V1 = 1
FORMAT_V2 = 2
SUPPORTED_FORMATS = (FORMAT_V1, FORMAT_V2)
DEFAULT_FORMAT_VERSION = FORMAT_V2

FORMAT_NAME = "merkle-weight-verify"

TAG_LEAF = b"\x00"
TAG_NODE = b"\x01"
TAG_TENSOR = b"\x02"
TAG_LAYER = b"\x03"
TAG_MODEL = b"\x04"

Algorithm = Optional[Union[HashAlgorithm, str]]


def check_format_version(version: int) -> int:
    if version not in SUPPORTED_FORMATS:
        raise ValueError(
            f"unsupported manifest format version {version!r}; "
            f"this release reads {list(SUPPORTED_FORMATS)}"
        )
    return version


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------

def _pack(code: str, n: int, bits: int) -> bytes:
    if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n < (1 << bits):
        raise ValueError(f"{n!r} does not fit in an unsigned {bits}-bit integer")
    return struct.pack(code, n)


def u8(n: int) -> bytes:
    return _pack(">B", n, 8)


def u32(n: int) -> bytes:
    return _pack(">I", n, 32)


def u64(n: int) -> bytes:
    return _pack(">Q", n, 64)


def enc_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return u32(len(b)) + b


def encode_meta(dtype: Optional[str], shape: Optional[Sequence[int]]) -> bytes:
    """``u8(0)`` for opaque blobs, else ``u8(1) || str(dtype) || u32(ndim) || u64(dim)*``."""
    if dtype is None and shape is None:
        return u8(0)
    if dtype is None or shape is None:
        raise ValueError("dtype and shape must be given together")
    out = [u8(1), enc_str(dtype), u32(len(shape))]
    out.extend(u64(int(d)) for d in shape)
    return b"".join(out)


def name_sort_key(name: str) -> bytes:
    """Tensors are ordered by the bytes of their UTF-8 names."""
    return name.encode("utf-8")


# ---------------------------------------------------------------------------
# Node hashing
# ---------------------------------------------------------------------------

def hash_leaf(chunk: bytes, algorithm: Algorithm = None) -> str:
    """v2 chunk leaf: ``H(0x00 || chunk)``, hex."""
    h = hasher_factory(algorithm)()
    h.update(TAG_LEAF)
    h.update(chunk)
    return h.hexdigest()


def hash_node(left: str, right: str, algorithm: Algorithm = None) -> str:
    """v2 interior node: ``H(0x01 || raw(left) || raw(right))``, hex."""
    return compute_digest(TAG_NODE + bytes.fromhex(left) + bytes.fromhex(right), algorithm).hex()


def _largest_power_of_two_below(n: int) -> int:
    k = 1
    while k << 1 < n:
        k <<= 1
    return k


def mth(leaves: Sequence[str], algorithm: Algorithm = None) -> str:
    """RFC 6962 Merkle Tree Hash over already-hashed leaves (hex in, hex out)."""
    if not leaves:
        raise ValueError("MTH of an empty list is not defined in this format")
    alg = resolve_algorithm(algorithm)
    level = list(leaves)
    while len(level) > 1:
        nxt = [hash_node(level[i], level[i + 1], alg) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])  # carried up unchanged
        level = nxt
    return level[0]


def inclusion_path(leaves: Sequence[str], index: int, algorithm: Algorithm = None) -> List[str]:
    """RFC 9162 inclusion path (sibling digests, leaf to root) for ``leaves[index]``."""
    n = len(leaves)
    if not 0 <= index < n:
        raise ValueError(f"index {index} out of range for {n} leaves")
    alg = resolve_algorithm(algorithm)
    path: List[str] = []
    level = list(leaves)
    i = index
    while len(level) > 1:
        if i % 2 == 1:
            path.append(level[i - 1])
        elif i + 1 < len(level):
            path.append(level[i + 1])
        # else: the node is carried up, no sibling at this level
        nxt = [hash_node(level[j], level[j + 1], alg) for j in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
        i //= 2
    return path


def root_from_inclusion_path(
    leaf: str, index: int, size: int, path: Sequence[str], algorithm: Algorithm = None
) -> Optional[str]:
    """Fold an inclusion path (RFC 9162 §2.1.3.2). Returns the root, or None if the
    path length is inconsistent with ``index`` and ``size``."""
    if not 0 <= index < size:
        return None
    alg = resolve_algorithm(algorithm)
    fn, sn = index, size - 1
    r = leaf
    for p in path:
        if sn == 0:
            return None
        if fn & 1 or fn == sn:
            r = hash_node(p, r, alg)
            if not fn & 1:
                while not fn & 1 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            r = hash_node(r, p, alg)
        fn >>= 1
        sn >>= 1
    if sn != 0:
        return None
    return r


# ---------------------------------------------------------------------------
# Tensor, layer and model roots
# ---------------------------------------------------------------------------

def tensor_root(
    name: str,
    chunk_root: str,
    num_bytes: int,
    chunk_size: int,
    dtype: Optional[str] = None,
    shape: Optional[Sequence[int]] = None,
    algorithm: Algorithm = None,
) -> str:
    """``H(0x02 || str(name) || meta || u64(num_bytes) || u64(chunk_size) || chunk_root)``."""
    data = b"".join([
        TAG_TENSOR,
        enc_str(name),
        encode_meta(dtype, shape),
        u64(num_bytes),
        u64(chunk_size),
        bytes.fromhex(chunk_root),
    ])
    return compute_digest(data, algorithm).hex()


def layer_digest(layer_name: str, tensor_roots: Sequence[str], algorithm: Algorithm = None) -> str:
    """Index digest for one layer; ``tensor_roots`` in name order. Not part of the model root."""
    data = b"".join(
        [TAG_LAYER, enc_str(layer_name), u64(len(tensor_roots))]
        + [bytes.fromhex(t) for t in tensor_roots]
    )
    return compute_digest(data, algorithm).hex()


def wrap_model_root(tensor_tree_root: str, num_tensors: int, algorithm: Algorithm = None) -> str:
    return compute_digest(
        TAG_MODEL + u64(num_tensors) + bytes.fromhex(tensor_tree_root), algorithm
    ).hex()


def model_root(tensor_roots: Sequence[str], algorithm: Algorithm = None) -> str:
    """``H(0x04 || u64(P) || MTH(tensor roots in name order))``."""
    if not tensor_roots:
        raise ValueError("a model needs at least one tensor")
    return wrap_model_root(mth(tensor_roots, algorithm), len(tensor_roots), algorithm)


def parse_root(root: str) -> Tuple[Optional[str], str]:
    """Split ``"sha256:ab12.."`` into ``("sha256", "ab12..")``; a bare hex root has no algorithm."""
    if ":" in root:
        alg, hexpart = root.split(":", 1)
        return resolve_algorithm(alg).value, hexpart.lower()
    return None, root.lower()
