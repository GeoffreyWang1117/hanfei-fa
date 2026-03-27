<div align="center">

# hanfei-fa 法

**Verify ML model weights. Know exactly what changed.**

<br>

> **法不阿贵，绳不挠曲。**
> *The law does not favor the noble; the plumb line does not bend for the crooked.*
> — 韩非子

</div>

---

Hierarchical Merkle tree verification for ML model weights. Answers three questions no other tool can:

1. **"Is this model exactly what I expect?"** — O(1) root hash comparison
2. **"Which layers changed after fine-tuning?"** — O(k log C) tree-walk diff with layer/tensor/chunk granularity
3. **"How much bandwidth can I save with incremental sync?"** — estimates show 50-70% savings for typical fine-tuning

Zero runtime dependencies. Pure Python standard library. Optional integrations with safetensors, PyTorch, HuggingFace Hub, and BLAKE3.

## Why this exists

Every existing tool hashes model files as opaque blobs:

| Tool | Granularity | Diff capability | Knows model structure? |
|------|------------|-----------------|----------------------|
| HuggingFace Hub | Whole-file SHA256 | No | No |
| HuggingFace Xet | Byte-level CDC chunks | Implicit (dedup) | No |
| Sigstore Model Signing | Whole-file SHA256 | No | No |
| DVC | Whole-file MD5 | No | No |
| PyTorch `torch.save` | None (CRC disabled) | No | No |
| safetensors | None (Issue [#220](https://github.com/huggingface/safetensors/issues/220) closed "not planned") | No | No |
| **hanfei-fa** | **Chunk → Tensor → Layer → Model** | **O(k log C) tree-walk** | **Yes** |

`hanfei-fa` is the only tool that understands model structure. When you fine-tune 2 of 12 transformer layers, it tells you *which* 2 layers changed, *which* tensors within them, and *which* chunks — without scanning the unchanged 80%.

## Install

```bash
pip install hanfei-fa                    # core (zero deps)
pip install hanfei-fa[safetensors]       # + safetensors support
pip install hanfei-fa[huggingface]       # + HuggingFace Hub integration
pip install hanfei-fa[torch]             # + PyTorch checkpoint support
pip install hanfei-fa[fast]              # + BLAKE3 (5-10x faster hashing)
pip install hanfei-fa[all]               # everything
```

## Quick Start

### Sign and verify a safetensors model

```python
from merkle_verify.safetensors_adapter import sign, verify

# Sign: builds Merkle tree, writes .merkle.json sidecar
tree = sign("model.safetensors")
print(tree.model_root)  # e14b10a8ce78b70...

# Verify: re-hashes and compares against manifest
is_valid, details = verify("model.safetensors")
# True — all tensors intact
```

### Diff two model versions

```python
from merkle_verify.safetensors_adapter import diff

result = diff("base_model.safetensors", "finetuned_model.safetensors")
print(result["changed_layers"])   # ['blocks.4', 'blocks.5']
print(result["changed_params"])   # ['blocks.4.attn.weight', ...]
print(result["change_percentage"])  # 33.2%
print(result["hash_comparisons"])   # 2066 (vs 21811 total chunks)
```

### Verify a single tensor (without loading the full model)

```python
from merkle_verify.safetensors_adapter import verify_tensor

is_valid, details = verify_tensor("model.safetensors", "blocks.0.attn.weight")
# Loads and hashes only this one tensor — O(tensor_size), not O(model_size)
```

### Sign a HuggingFace Hub model from local cache

```python
from merkle_verify.safetensors_adapter import from_hf_repo

tree = from_hf_repo("bert-base-uncased")
# Automatically finds cached safetensors, handles sharded models
print(f"{tree.model_root}")   # golden fingerprint
print(f"{len(tree.layer_trees)} layers, verified")
```

### PyTorch checkpoints

```python
from merkle_verify.pytorch_adapter import merkle_save, merkle_load

# Save with integrity manifest
merkle_save(model, "checkpoint.pt")

# Load with automatic verification
state_dict, details = merkle_load("checkpoint.pt")
assert details["verified"]  # weights match manifest
```

### Use BLAKE3 for faster hashing

```python
from merkle_verify import set_default_algorithm, HashAlgorithm

set_default_algorithm(HashAlgorithm.BLAKE3)  # 5-10x faster than SHA-256
# All subsequent operations use BLAKE3 automatically
```

### Stream-hash a large file (constant memory)

```python
from merkle_verify import build_file_merkle_tree

tree = build_file_merkle_tree("70b-model.safetensors")
# O(chunk_size) memory, not O(file_size). Works on multi-GB files.
```

## CLI

```bash
merkle-verify hash model.safetensors          # Merkle root hash
merkle-verify sign model.safetensors          # Build tree + write .merkle.json
merkle-verify verify model.safetensors        # Check against manifest (exit 0/1)
merkle-verify diff base.safetensors ft.safetensors  # Layer-aware diff
merkle-verify info model.merkle.json          # Show manifest details
merkle-verify hf-sign bert-base-uncased       # Sign from HF cache
```

## How it works

A 4-level hierarchical Merkle tree mirrors the structure of a neural network:

```
                Model Root
               /          \
        Layer 0            Layer 1         ...  Layer N
       /       \          /       \
  attn.weight  attn.bias  mlp.weight  mlp.bias
   /  |  \       |         /  |  \      |
  c0  c1  c2    c0       c0  c1  c2    c0       ← 16KB chunks
```

**Verification**: Compare root hashes — O(1).

**Diff**: Walk both trees in parallel. If a subtree's hash matches, skip it entirely. Only descend into subtrees that differ. Complexity: O(k log C) where k = changed chunks, C = total chunks.

**Pruning in practice**: Fine-tuning 1 of 60 ResNet parameters? The diff performs 264 hash comparisons instead of scanning all 2,953 chunks — a 91% reduction.

## Performance

Tested on real models:

| Model | Params | Build time | Diff (1% change) | Hash comparisons |
|-------|--------|-----------|-------------------|-----------------|
| ResNet-18 | 11.7M | 0.03s | 0.1ms | 264 / 2,953 |
| BERT-base | 110M | 1.2s | — | — |
| GPT-2 scale (340MB) | — | 0.8s | 1.0ms | 2,066 / 21,811 |
| Streaming 512MB | — | 0.9s | — | — |

## Supported hash algorithms

| Algorithm | Output | Speed | Install |
|-----------|--------|-------|---------|
| SHA-256 (default) | 64 hex | Baseline | Built-in |
| SHA-512 | 128 hex | ~Same | Built-in |
| SHA3-256 | 64 hex | ~Same | Built-in |
| BLAKE2b | 128 hex | ~Same | Built-in |
| BLAKE3 | 64 hex | **5-10x faster** | `pip install hanfei-fa[fast]` |

## Part of the HanFei (韩非) series

This project is part of a family of open-source tools for verifiable AI:

| Project | Role | Language | Install |
|---------|------|----------|---------|
| [**hanfei-shu 术**](https://github.com/GeoffreyWang1117/hanfei-shu) | GPU-accelerated MSM for ZK proofs | Rust + CUDA | `cargo add hanfei-shu` |
| **hanfei-fa 法** (this) | Model weight integrity verification | Python | `pip install hanfei-fa` |

The names come from Han Feizi's (韩非子) political philosophy:
- **法 (fa)** — Law: objective, deterministic verification. A hash doesn't lie.
- **术 (shu)** — Technique: the computational machinery that makes proofs fast.

## Contributing

Contributions are welcome and appreciated. This project grows through community involvement.

**How to contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Make your changes with tests
4. Submit a pull request

All PRs are reviewed and merged on a regular basis. We especially welcome:
- New model architecture support in `_extract_layer_name()`
- Chunking strategy improvements (content-defined chunking, etc.)
- Performance optimizations
- Documentation and examples
- Integration with other ML frameworks (JAX, TensorFlow, ONNX)

**Found this useful?** Please consider:
- Giving a star on [GitHub](https://github.com/GeoffreyWang1117/hanfei-fa)
- Citing the project if you use it in your work:

```bibtex
@software{hanfei_fa,
  author = {Geoffrey Wang},
  title = {hanfei-fa: ML Model Weight Integrity Verification via Hierarchical Merkle Trees},
  year = {2026},
  url = {https://github.com/GeoffreyWang1117/hanfei-fa},
}
```

## License

Apache-2.0 — Copyright 2026 Geoffrey Wang
