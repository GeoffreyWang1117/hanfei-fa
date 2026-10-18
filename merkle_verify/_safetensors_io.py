"""
Minimal, dependency-free reader for the safetensors container format.

Format (https://github.com/huggingface/safetensors#format):

    [8 bytes]  N, little-endian unsigned 64-bit header length
    [N bytes]  UTF-8 JSON header:
               {"<name>": {"dtype": "F32", "shape": [..], "data_offsets": [start, end]},
                ...,
                "__metadata__": {"<str>": "<str>"}}   (optional)
    [rest]     byte buffer; offsets are relative to the start of this buffer

Tensors are read straight from their byte ranges, so hashing never decodes
them. That means every dtype (BF16, F8_*, ...) works, nothing beyond the
standard library is needed, and memory use is one chunk at a time.
"""

import json
import os
import struct
from dataclasses import dataclass
from typing import BinaryIO, Dict, Iterator, Tuple

# safetensors itself rejects headers above 100 MB
MAX_HEADER_BYTES = 100 * 1024 * 1024

# Bits per element of the safetensors dtypes, to check declared shapes against byte ranges.
DTYPE_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8,
    "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
    "I32": 32, "U32": 32, "F32": 32,
    "I64": 64, "U64": 64, "F64": 64, "C64": 64, "C128": 128,
    "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6,
}


@dataclass(frozen=True)
class TensorInfo:
    """Location and metadata of one tensor inside a safetensors file."""

    name: str
    dtype: str
    shape: Tuple[int, ...]
    offset: int  # absolute byte offset in the file
    nbytes: int


def read_header(path: str) -> Tuple[Dict[str, TensorInfo], Dict[str, str]]:
    """Parse a safetensors header.

    Returns:
        (tensors, metadata) where ``tensors`` maps name to :class:`TensorInfo`
        and ``metadata`` is the optional ``__metadata__`` dict.

    Raises:
        ValueError: if the file is not a well-formed safetensors file.
    """
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) < 8:
            raise ValueError(f"{path}: not a safetensors file (shorter than 8 bytes)")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len > MAX_HEADER_BYTES or 8 + header_len > file_size:
            raise ValueError(f"{path}: not a safetensors file (bad header length {header_len})")
        raw = f.read(header_len)

    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise ValueError(f"{path}: safetensors header is not valid JSON ({e})") from None
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header must be a JSON object")

    metadata = header.pop("__metadata__", None) or {}
    data_start = 8 + header_len
    data_len = file_size - data_start

    tensors: Dict[str, TensorInfo] = {}
    for name, entry in header.items():
        try:
            dtype = str(entry["dtype"])
            shape = tuple(int(d) for d in entry["shape"])
            start, end = (int(x) for x in entry["data_offsets"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{path}: malformed header entry for tensor {name!r}") from None
        if not (0 <= start <= end <= data_len):
            raise ValueError(
                f"{path}: tensor {name!r} has data_offsets [{start}, {end}] "
                f"outside the {data_len}-byte data buffer (truncated file?)"
            )
        if any(d < 0 for d in shape):
            raise ValueError(f"{path}: tensor {name!r} has a negative dimension")
        bits = DTYPE_BITS.get(dtype)
        if bits is not None:
            numel = 1
            for d in shape:
                numel *= d
            if (numel * bits + 7) // 8 != end - start:
                raise ValueError(
                    f"{path}: tensor {name!r} is declared {dtype}{list(shape)} "
                    f"but occupies {end - start} bytes"
                )
        tensors[name] = TensorInfo(name, dtype, shape, data_start + start, end - start)
    return tensors, metadata


def _read_exact(f: BinaryIO, n: int) -> bytes:
    parts = []
    remaining = n
    while remaining > 0:
        block = f.read(remaining)
        if not block:
            raise ValueError("unexpected end of file while reading tensor data")
        parts.append(block)
        remaining -= len(block)
    return b"".join(parts)


def iter_tensor_chunks(f: BinaryIO, info: TensorInfo, chunk_size: int) -> Iterator[bytes]:
    """Yield a tensor's bytes in ``chunk_size`` pieces (the last may be shorter)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    f.seek(info.offset)
    remaining = info.nbytes
    while remaining > 0:
        n = min(chunk_size, remaining)
        yield _read_exact(f, n)
        remaining -= n


def read_tensor_bytes(path: str, info: TensorInfo) -> bytes:
    """Read one tensor's raw bytes."""
    with open(path, "rb") as f:
        f.seek(info.offset)
        return _read_exact(f, info.nbytes)
