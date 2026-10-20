# Changelog

All notable changes to this project are documented here. Versions follow
[semantic versioning](https://semver.org/); while the major version is 0, a
minor version may change the manifest format.

## [0.3.0] - unreleased

Manifest format v2 becomes the default. Roots change for newly signed
models; v1 manifests written by 0.2.x keep verifying with the v1 rules.
Specification: `docs/MANIFEST_V2.md`.

### Changed

- New trees use format v2: RFC 6962 trees with domain-separated leaves
  (`0x00`) and nodes (`0x01`) over raw digests; an odd node is carried up
  instead of being paired with itself.
- Each tensor root binds the tensor's name, dtype, shape, byte length and
  chunk size. The model root is a Merkle tree over tensor roots sorted by
  name, wrapped with the tensor count. Layer grouping no longer enters the
  root; layer digests remain as a diff index.
- Manifests record `format`, `format_version`, `algorithm`, `num_tensors` and
  `created_by`. Every tree remembers its algorithm, so changing the process
  default no longer breaks proofs or verification.
- Loading a v2 manifest re-derives every chunk root, tensor root and the model
  root and rejects the manifest if any stored value disagrees.
- Diffing trees of different formats, algorithms or chunk sizes is an error
  instead of reporting everything as changed.
- `MerkleTree.from_file` on an empty file yields one empty chunk in v2.

### Added

- Inclusion proofs against the model root: `ModelMerkleTree.get_tensor_proof()`
  and `get_chunk_proof()`, verified with the RFC 9162 algorithm.
- Compact manifests without per-chunk hashes (`include_chunks=False`,
  `merkle-verify sign --compact`).
- `RawTensor(data, dtype, shape)` input for raw buffers, `MerkleTree.from_bytes()`,
  `hash_leaf()` / `hash_node()`, and `format_version=` / `algorithm=` keyword
  arguments on every builder; CLI `--format-version`, `--algorithm`.
- Expected roots may carry their algorithm: `"sha512:<hex>"`.
- `ModelMerkleTree.total_params` is exact when shapes are known.
- `scripts/bench.py` and `docs/benchmarks.md`: build, diff and sync
  measurements on real models.
- `tests/reference_v2.py`: an independent implementation of the specification
  that generates `tests/vectors/format_v2.json`.

## [0.2.1] - 2026-10-18

Every root produced by 0.2.0 is reproduced exactly (pinned by
`tests/vectors/format_v1.json`); existing manifests keep verifying.

### Fixed

- **`verify(..., expected_root=...)` and `verify_checkpoint(..., expected_root=...)`
  did not read the weights.** They compared the trusted root with the root
  stored in the sidecar manifest, so a tampered file next to an untouched
  manifest verified as valid. They now re-hash the weights and compare the
  freshly computed root with `expected_root`. The manifest becomes optional in
  this mode. The details `mode` is now `"expected_root"` instead of
  `"root_only"`.
- **`merkle-verify sign checkpoint.pt` rewrote the checkpoint** through
  `torch.save`. Signing is now read-only (`pytorch_adapter.sign_checkpoint`).
- **bfloat16 (and float8) tensors could not be signed.** The safetensors adapter
  decoded tensors through numpy, which has no bfloat16; the PyTorch adapter
  called `.numpy()`, which fails the same way. Both now hash raw bytes.
- **Tensors in nested checkpoints were ignored.** `{"model": ..., "optimizer": ...}`
  checkpoints either failed or silently skipped nested tensors, so tampering
  with them went undetected. Checkpoints are now flattened to dotted names.
- Tensors with zero elements crashed tree construction; they are now a single
  empty chunk.
- `from_hf_repo` / `hf-sign` wrote a manifest into the HuggingFace cache
  snapshot. Nothing is written unless `manifest_path=` / `--output` is given.
- `from_hf_repo` with a `revision` that was not cached silently used another
  snapshot. It now raises. Branch and tag names are resolved through `refs/`,
  and `HF_HUB_CACHE` / `HF_HOME` are honoured.
- `from_hf_repo` required `huggingface_hub` without using it.
- `merkle-verify diff` and `info` failed on single-file (non-model) manifests.

### Added

- Built-in safetensors reader: the safetensors adapter needs no third-party
  package, streams each tensor chunk by chunk (memory is one chunk per worker,
  not the model size) and validates header offsets.
- `verify_sharded()` and `build_sharded_tree()`; the CLI `sign`, `verify` and
  `diff` accept a model directory (`model.safetensors.index.json` + shards).
  Tensors missing from their declared shard, or present in two shards, are errors.
- `build_model_merkle_tree()` accepts `torch.Tensor` and numpy values directly.
- `merkle-verify verify --expected-root`, `merkle-verify --version`,
  `hf-sign --output`.
- `assemble_model_tree()` and `check_roots()` public helpers.
- Golden-vector tests for the v1 format and regression tests for every fix above.

### Changed

- CLI exit codes: `2` for errors (malformed input, missing files), `1` only for
  failed verification.
- `verify` uses the manifest's chunk size unless `--chunk-size` is given.

## [0.2.0] - 2026-03-26

Initial release on PyPI as `merkle-weight-verify` and `hanfei-fa`.
