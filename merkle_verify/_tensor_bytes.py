"""
Framework-agnostic conversion of tensor-like values to raw bytes.

Nothing here imports torch or numpy at module load time. A value is converted
according to what it is:

* ``bytes`` / ``bytearray`` / ``memoryview`` -- used as-is, no tensor metadata.
* ``torch.Tensor`` (including ``nn.Parameter``) -- detached, moved to CPU, made
  contiguous and reinterpreted as ``uint8``. This works for every dtype torch
  can view as bytes, including ``bfloat16`` and the float8 types, which numpy
  cannot represent.
* numpy arrays -- made C-contiguous and little-endian.

The byte layout matches what safetensors writes to disk (C order, little
endian), so a state_dict and the ``.safetensors`` file saved from it hash to
the same parameter roots.
"""

import sys
from typing import Any, Dict, Mapping, Optional, Tuple

#: (dtype code, shape) using safetensors dtype codes, e.g. ("BF16", (4096, 4096)).
TensorMeta = Tuple[str, Tuple[int, ...]]

# torch dtype (str(dtype) without the "torch." prefix) -> safetensors dtype code
_TORCH_DTYPE_CODES = {
    "float64": "F64",
    "float32": "F32",
    "float16": "F16",
    "bfloat16": "BF16",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "uint64": "U64",
    "uint32": "U32",
    "uint16": "U16",
    "uint8": "U8",
    "bool": "BOOL",
    "float8_e4m3fn": "F8_E4M3",
    "float8_e5m2": "F8_E5M2",
    "complex64": "C64",
    "complex128": "C128",
}

# numpy (kind, itemsize) -> safetensors dtype code
_NUMPY_DTYPE_CODES = {
    ("f", 8): "F64",
    ("f", 4): "F32",
    ("f", 2): "F16",
    ("i", 8): "I64",
    ("i", 4): "I32",
    ("i", 2): "I16",
    ("i", 1): "I8",
    ("u", 8): "U64",
    ("u", 4): "U32",
    ("u", 2): "U16",
    ("u", 1): "U8",
    ("b", 1): "BOOL",
    ("c", 8): "C64",
    ("c", 16): "C128",
}

_BYTES_TYPES = (bytes, bytearray, memoryview)


def _is_torch_tensor(value: Any) -> bool:
    return (
        type(value).__module__.split(".")[0] == "torch"
        and hasattr(value, "detach")
        and hasattr(value, "numel")
    )


def _is_numpy_array(value: Any) -> bool:
    return (
        type(value).__module__.split(".")[0] == "numpy"
        and hasattr(value, "tobytes")
        and hasattr(value, "dtype")
    )


def is_tensor_like(value: Any) -> bool:
    """True if :func:`to_bytes` can convert ``value``."""
    return isinstance(value, _BYTES_TYPES) or _is_torch_tensor(value) or _is_numpy_array(value)


def _torch_to_bytes(value: Any) -> Tuple[bytes, TensorMeta]:
    torch = sys.modules["torch"]
    t = value.detach()
    if getattr(t, "is_sparse", False):
        raise TypeError("sparse tensors are not supported; call .to_dense() first")
    if getattr(t, "is_quantized", False):
        raise TypeError("quantized tensors are not supported; hash t.int_repr() instead")
    if sys.byteorder != "little":
        raise NotImplementedError("tensor hashing requires a little-endian host")

    dtype_name = str(t.dtype).replace("torch.", "")
    code = _TORCH_DTYPE_CODES.get(dtype_name)
    if code is None:
        raise TypeError(f"unsupported torch dtype: {t.dtype}")

    t = t.cpu().contiguous()
    shape = tuple(int(d) for d in t.shape)
    if t.numel() == 0:
        return b"", (code, shape)

    raw = t.reshape(-1).view(torch.uint8)
    try:
        data = raw.numpy().tobytes()
    except (RuntimeError, ImportError):
        # torch built or installed without numpy
        data = bytes(raw.tolist())
    return data, (code, shape)


def _numpy_to_bytes(value: Any) -> Tuple[bytes, TensorMeta]:
    import numpy as np

    a = np.ascontiguousarray(value)
    code = _NUMPY_DTYPE_CODES.get((a.dtype.kind, a.dtype.itemsize))
    if code is None:
        raise TypeError(f"unsupported numpy dtype: {a.dtype}")
    if a.dtype.byteorder not in ("<", "|") and not (
        a.dtype.byteorder == "=" and sys.byteorder == "little"
    ):
        a = a.astype(a.dtype.newbyteorder("<"))
    return a.tobytes(), (code, tuple(int(d) for d in a.shape))


def to_bytes_and_meta(value: Any) -> Tuple[bytes, Optional[TensorMeta]]:
    """Convert a tensor-like value to ``(raw_bytes, meta)``.

    ``meta`` is ``None`` for plain byte strings.
    """
    if isinstance(value, bytes):
        return value, None
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value), None
    if _is_torch_tensor(value):
        return _torch_to_bytes(value)
    if _is_numpy_array(value):
        return _numpy_to_bytes(value)
    raise TypeError(
        f"cannot hash value of type {type(value).__name__}; "
        "expected bytes, a torch.Tensor or a numpy array"
    )


def to_bytes(value: Any) -> bytes:
    """Convert a tensor-like value to its raw little-endian bytes."""
    return to_bytes_and_meta(value)[0]


def flatten_state_dict(obj: Mapping, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested mappings of tensors into a single ``{dotted.name: tensor}`` dict.

    Training checkpoints often look like ``{"model": state_dict, "optimizer": ...,
    "epoch": 3}``. Every tensor anywhere in the structure is kept under its
    dotted path; non-tensor leaves (ints, strings, ...) are dropped because they
    are not weights.
    """
    flat: Dict[str, Any] = {}
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            nested = flatten_state_dict(value, prefix=f"{name}.")
            for nested_name in nested:
                if nested_name in flat:
                    raise ValueError(f"duplicate tensor name after flattening: {nested_name}")
            flat.update(nested)
        elif is_tensor_like(value):
            if name in flat:
                raise ValueError(f"duplicate tensor name after flattening: {name}")
            flat[name] = value
    return flat
