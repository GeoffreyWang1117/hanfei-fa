# merkle-weight-verify

[![PyPI](https://img.shields.io/pypi/v/merkle-weight-verify.svg)](https://pypi.org/project/merkle-weight-verify/)
[![CI](https://github.com/GeoffreyWang1117/hanfei-fa/actions/workflows/ci.yml/badge.svg)](https://github.com/GeoffreyWang1117/hanfei-fa/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Hierarchical Merkle trees for verifying ML model weights: one root hash for the
whole model, and a tensor- and chunk-level diff when it does not match.

- **"Are these exactly the weights I expect?"** Re-hash the files and compare one root.
- **"What changed between two versions?"** A tree-walk diff names the layers,
  tensors and 16 KB chunks that differ, in O(k log C) hash comparisons for k
  changed chunks out of C.
- **"How much do I need to re-download?"** Estimate incremental-sync size from the diff.

The core library uses only the Python standard library. Safetensors files
(single-file or sharded) are read by a built-in parser, so they need no extra
packages and every dtype works, including bfloat16 and float8. PyTorch
checkpoints need `torch`.

The same package is also published as [`hanfei-fa`](https://pypi.org/project/hanfei-fa/)
(part of the HanFei series, alongside the Rust port
[`hanfei-fa`](https://crates.io/crates/hanfei-fa) on crates.io). Both names
install the same `merkle_verify` module, so install only one of them.

## Install

```bash
pip install merkle-weight-verify            # core + safetensors
pip install "merkle-weight-verify[torch]"   # + PyTorch checkpoints
pip install "merkle-weight-verify[fast]"    # + BLAKE3 hash backend
```

## Command line

```bash
# Sign: build the tree and write a manifest next to the weights (weights are never modified)
merkle-verify sign model.safetensors          # -> model.merkle.json
merkle-verify sign ./Llama-3.1-8B/            # sharded: index.json + shards -> model.merkle.json
merkle-verify sign checkpoint.pt              # PyTorch checkpoint (read-only)

# Verify: re-hash the weights and compare
merkle-verify verify model.safetensors                         # against the sidecar manifest
merkle-verify verify ./Llama-3.1-8B/ --expected-root 9f2c...   # against a root you trust

# Diff two versions (files, model directories, or manifests)
merkle-verify diff base.safetensors finetuned.safetensors

# Hash a model already in the HuggingFace cache (nothing is written to the cache)
merkle-verify hf-sign Qwen/Qwen2.5-0.5B --revision main -o qwen.merkle.json
```

Exit codes: `0` verified, `1` verification failed, `2` error (bad input, missing file).

## Python API

### Safetensors

```python
from merkle_verify.safetensors_adapter import sign, verify, diff, verify_sharded

tree = sign("model.safetensors")            # writes model.merkle.json
print(tree.model_root)

ok, details = verify("model.safetensors")   # re-hashes, compares with the manifest
ok, details = verify("model.safetensors", expected_root=trusted_root)
if not ok:
    print(details["changed_params"])        # e.g. ['model.layers.7.mlp.down_proj.weight']

changes = diff("base.safetensors", "finetuned.safetensors")
print(changes["changed_layers"], changes["change_percentage"])

ok, details = verify_sharded("./Llama-3.1-8B/", expected_root=trusted_root)
```

### PyTorch state dicts and checkpoints

```python
from merkle_verify import build_model_merkle_tree
from merkle_verify.pytorch_adapter import sign_checkpoint, verify_checkpoint

tree = build_model_merkle_tree(model.state_dict())   # tensors accepted directly, any dtype
sign_checkpoint("checkpoint.pt")                     # nested checkpoints are flattened
ok, details = verify_checkpoint("checkpoint.pt", expected_root=tree.model_root)
```

A tensor hashes to its contiguous little-endian bytes, which is exactly what
safetensors stores, so `build_model_merkle_tree(state_dict)` and
`sign("model.safetensors")` give the same root for the same weights.

### Low-level trees and proofs

```python
from merkle_verify import MerkleTree, compute_hash, chunk_bytes

hashes = [compute_hash(c) for c in chunk_bytes(data, chunk_size=16384)]
tree = MerkleTree(hashes)
changed_indices, comparisons = tree.diff_tree(other_tree)

proof = tree.get_proof(chunk_index=42)
assert proof.verify()
```

### Sync estimate

```python
from merkle_verify import estimate_sync_savings

savings = estimate_sync_savings(old_tree, new_tree)
print(f"incremental sync would transfer {100 - savings['savings_percentage']:.1f}% of the bytes")
```

The estimate counts changed chunks times the chunk size, so it is an upper
bound on the bytes that differ.

## Security notes

**Where the trusted root comes from matters.** A `.merkle.json` sidecar is not
authenticated: anyone who can replace the weights can also regenerate the
sidecar. Verifying against the sidecar detects corruption and accidental
mismatches. To detect deliberate tampering, verify against a root obtained
through a channel the attacker does not control (`expected_root=` /
`--expected-root`), or sign the manifest, for example with
[Sigstore model signing](https://github.com/sigstore/model-transparency).
Verification always re-hashes the weights; a root read from a manifest is
never trusted on its own.

**Limitations of the current hash format (manifest format v1).** The model root
commits to the bytes of every tensor, but:

- leaf and internal nodes are not domain-separated, and a level with an odd
  number of nodes duplicates its last node, as in Bitcoin's original Merkle
  tree, so a chunk tree's root does not uniquely determine its number of chunks;
- tensor names are bound only through their sorted order, and dtype and shape
  are not bound at all, so reinterpreting a tensor's bytes under another dtype
  or shape keeps the root unchanged;
- manifests do not record the hash algorithm.

A versioned manifest format v2 that fixes all of these is planned for 0.3.
Version 1 manifests will keep verifying.

## Performance

ResNet18 (11.7M parameters, 60 tensors), single workstation CPU, SHA-256, 16 KB chunks:

| Operation | Time | Detail |
|-----------|------|--------|
| Build tree | 0.034s | All 60 tensors, serial or 4 threads |
| Compare (1 tensor changed) | 0.1ms | 264 hash comparisons vs 2953 total chunks |
| `ModelMerkleTree.verify()` | 0.1ms | Recomputes layer and model roots from stored tensor roots; does not re-read weights |

Larger-model figures (GPT-2 through 70B, and the 50-68% sync savings for
partial fine-tuning) come from the earlier research prototype this package was
extracted from. They have not yet been reproduced with this package; a
benchmark script that regenerates them is planned.

## Tests

```bash
pip install pytest              # core tests
pip install torch torchvision safetensors numpy blake3   # optional: integration tests
python -m pytest tests/
```

Tests that need an optional package are skipped when it is missing.
`tests/vectors/format_v1.json` pins the roots produced by the 0.2.0 release;
every later version must reproduce them.

## Roadmap

- **0.3: manifest format v2.** Domain-separated, length-binding tree hashing;
  tensor name, dtype and shape bound into each tensor's root; algorithm and
  format version recorded in the manifest; per-tensor inclusion proofs against
  the model root; shared test vectors with the Rust port.
- **0.4: ecosystem.** `hf-check` against published manifests, chunk-level
  repair from a healthy replica, Sigstore signing guide.
- Later: property-based tests, typed API, manifest specification document.

## License

Apache-2.0. Copyright 2026 Geoffrey Wang.
