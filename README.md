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
- **"Prove this one tensor is part of that model."** Inclusion proofs for a
  single tensor or a single chunk against the model root.
- **"Fix my corrupted download without fetching 70 GB again."** Re-fetch only
  the damaged chunks from any mirror, checking each one before it is written.

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

# Check a cached HF model against a manifest published with it, or a trusted root
merkle-verify hf-check org/model --expected-root sha256:9f2c...

# Repair: re-fetch only damaged chunks, from a local copy, a URL, or the HF Hub
merkle-verify repair ./Llama-3.1-8B/ --from hf://meta-llama/Llama-3.1-8B@<commit>
merkle-verify repair model.safetensors --from /mnt/backup/model.safetensors --dry-run

# Options: hash algorithm, compact manifests, the 0.2.x format
merkle-verify --algorithm blake3 sign model.safetensors
merkle-verify sign --compact model.safetensors      # no per-chunk hashes: small manifest
merkle-verify --format-version 1 sign model.safetensors
merkle-verify info model.merkle.json                # format, algorithm, root, counts
```

`verify` always uses the format, algorithm and chunk size recorded in the
manifest. A trusted root can carry its algorithm: `--expected-root sha512:9f2c...`.

Exit codes: `0` verified (or repaired and verified), `1` verification or repair failed,
`2` error (bad input, missing file).

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

### Proofs

```python
tree = sign("model.safetensors", sidecar=False)

# This tensor (name, dtype, shape, bytes) is committed by the model root
proof = tree.get_tensor_proof("model.layers.7.mlp.down_proj.weight")
assert proof.verify()

# These 16 KB are chunk 3 of that tensor, in the model with this root
proof = tree.get_chunk_proof("model.layers.7.mlp.down_proj.weight", 3)
assert proof.verify(chunk_data)
```

A verifier needs only the model root and the proof; no other tensor is revealed.

### Low-level trees

```python
from merkle_verify import MerkleTree, RawTensor, build_model_merkle_tree

tree = MerkleTree.from_bytes(data, chunk_size=16384)   # hashes leaves with the format's rule
changed_indices, comparisons = tree.diff_tree(other_tree)

# Raw buffers with explicit metadata (safetensors dtype codes)
model = build_model_merkle_tree({"w": RawTensor(raw, "BF16", (4096, 4096))})
```

### Repair

```python
from merkle_verify.repair import repair

report = repair("model.safetensors", "hf://org/model@<commit>")   # sidecar manifest by default
print(report.summary())   # e.g. 5/33522 chunks damaged; fetched 68608 bytes (0.01% of 548105171) in 5 request(s)
assert report.verified    # the repaired file reproduces the manifest root
```

Only the manifest has to be trusted. A chunk from the source is written only if
it hashes to the manifest's leaf, so a damaged or malicious mirror cannot make
things worse. Adjacent chunks are merged into one Range request and requests
run concurrently. A damaged safetensors header is re-fetched and checked
against the manifest's tensor list; a truncated file is completed. With a
`--compact` manifest a damaged tensor is re-fetched whole.

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
never trusted on its own, and a manifest whose stored roots disagree with its
own contents is rejected. [docs/SIGNING.md](docs/SIGNING.md) shows how to
sign a manifest with Sigstore and verify it before trusting it. A manifest
published in the same HF repository as the weights detects corrupted
downloads, not a compromised repository.

**What a root commits to (format v2, the default since 0.3).** Every tensor's
name, dtype, shape and bytes, the chunk size, and the number of tensors. Trees
are RFC 6962 Merkle trees with domain-separated leaves and nodes, so no chunk
can pass for an interior node and no tree is the duplicate-padded version of
another. Layer grouping is only an index for fast diffs and does not affect
the root. The manifest records the hash algorithm. The byte-level
specification is [docs/MANIFEST_V2.md](docs/MANIFEST_V2.md).

**Format v1** (written by 0.2.x) lacks all of this: no domain separation,
odd levels duplicate their last node, names are bound only through their sort
order, dtype and shape are not bound, and the algorithm is not recorded.
v1 manifests still load and verify with the v1 rules, but new manifests should
use v2.

## Performance

Measured with `scripts/bench.py` on a Ryzen 9 5950X, SHA-256, 16 KB chunks.
Full table and method: [docs/benchmarks.md](docs/benchmarks.md).

| Model | Size | Build (v2) | Merged LoRA update: diff | Sync saving |
|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B (bf16) | 988 MB | 0.51 s | 1.7 ms, 48 tensors changed | 95.5% |
| gpt2-medium | 1520 MB | 0.83 s | 9.1 ms, 24 tensors changed | 80.1% |

Updates confined to a few tensors save 80-99% of the transfer; a full
fine-tune rewrites everything and saves nothing.

## Tests

```bash
pip install pytest              # core tests
pip install torch torchvision safetensors numpy blake3   # optional: integration tests
python -m pytest tests/
```

Tests that need an optional package are skipped when it is missing.
`tests/vectors/format_v1.json` pins the roots produced by the 0.2.0 release.
`tests/vectors/format_v2.json` is generated by `tests/reference_v2.py`, an
independent implementation written from the specification; the library must
match it for every algorithm.

## Roadmap

- **0.3: manifest format v2** (this release). See [CHANGELOG.md](CHANGELOG.md).
- **Next:** chunk-level repair, `hf-check` and the signing guide (in review);
  the v2 format in the Rust port (in review).
- Later: property-based tests, typed API.

## License

Apache-2.0. Copyright 2026 Geoffrey Wang.
