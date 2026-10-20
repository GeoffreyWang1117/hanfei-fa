"""Shared test helpers (not collected by pytest)."""

import json
import struct


def write_safetensors(path, tensors, metadata=None):
    """Write a safetensors file from {name: (dtype, shape, raw_bytes)} without any library."""
    header = {}
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    if metadata:
        header["__metadata__"] = metadata
    hjson = json.dumps(header).encode()
    hjson += b" " * (-len(hjson) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for _, _, raw in tensors.values():
            f.write(raw)


def flip_last_byte(path):
    data = bytearray(open(path, "rb").read())
    data[-1] ^= 0xFF
    open(path, "wb").write(bytes(data))


SAMPLE = {
    "layer.0.weight": ("F32", (8, 8), bytes(range(256))),
    "layer.0.bias": ("F32", (8,), bytes(range(32))),
    "layer.1.weight": ("BF16", (16, 4), bytes(range(128))),
}
