# Benchmarks

Reproduce with:

```bash
python scripts/bench.py path/to/model.safetensors [more models ...] --json results.json
```

The script reads each model once, builds its tree from disk with format v1
and v2 (best of three runs, page cache warm), then simulates four updates in
memory and diffs them against the base tree. "Sync" is the number of changed
chunks times the chunk size: what an incremental transfer would move.

## 2026-10-18

merkle-weight-verify 0.3.0.dev0, Python 3.14.6, AMD Ryzen 9 5950X (16 cores),
Linux, SHA-256, 16 KB chunks, 4 hashing threads (the default). Models from
the HuggingFace cache.

| Model | Size | Tensors | Build v1 | Build v2 |
|---|---|---|---|---|
| gpt2 | 548 MB | 160 | 0.24 s (2.26 GB/s) | 0.19 s (2.89 GB/s) |
| Qwen/Qwen2.5-0.5B (bf16) | 988 MB | 290 | 0.39 s (2.56 GB/s) | 0.34 s (2.92 GB/s) |
| gpt2-medium | 1520 MB | 316 | 0.57 s (2.67 GB/s) | 0.46 s (3.34 GB/s) |

| Model | Update | Changed tensors | Hash comparisons | Diff time | Sync | Saving |
|---|---|---|---|---|---|---|
| gpt2 | one MLP matrix | 1 | 1,179 | 0.5 ms | 9.4 MB | 98.3% |
| gpt2 | merged LoRA on attention (c_attn) | 12 | 10,527 | 2.4 ms | 84.9 MB | 84.5% |
| gpt2 | 1% of chunks at random | 59 | 4,682 | 1.6 ms | 5.5 MB | 99.0% |
| gpt2 | full fine-tune | 160 | 67,059 | 13 ms | 548 MB | 0% |
| Qwen2.5-0.5B | one MLP matrix | 1 | 1,100 | 0.8 ms | 8.7 MB | 99.1% |
| Qwen2.5-0.5B | merged LoRA on q/v | 48 | 5,641 | 1.7 ms | 44.0 MB | 95.5% |
| Qwen2.5-0.5B | 1% of chunks at random | 101 | 8,368 | 2.8 ms | 9.8 MB | 99.0% |
| Qwen2.5-0.5B | full fine-tune | 290 | 120,863 | 27 ms | 988 MB | 0% |
| gpt2-medium | one MLP matrix | 1 | 2,087 | 1.3 ms | 16.8 MB | 98.9% |
| gpt2-medium | merged LoRA on attention (c_attn) | 24 | 37,179 | 9.1 ms | 302 MB | 80.1% |
| gpt2-medium | 1% of chunks at random | 123 | 12,678 | 4.3 ms | 15.1 MB | 99.0% |
| gpt2-medium | full fine-tune | 316 | 185,801 | 41 ms | 1520 MB | 0% |

Reading the numbers:

- v2 is faster than v1 because interior nodes hash 65 raw bytes instead of
  128 hex characters.
- How much an incremental sync saves depends only on which tensors an update
  touches. Updates confined to a few tensors save 80-99%. A full fine-tune
  rewrites every chunk, and there is nothing to save.
- GPT-2 fuses q, k and v into `c_attn`, so a merged LoRA update rewrites the
  whole fused matrix. That is why its saving is lower than Qwen's.

## Scaling and large models (2026-10-18)

Same machine. `scripts/bench.py` numbers above are with the page cache warm.

**Threads.** SmolLM3-3B, 6.15 GB in 2 shards, 375,433 chunks, warm cache, SHA-256:

| Hashing threads | 1 | 2 | 4 (default) | 8 | 16 |
|---|---|---|---|---|---|
| Throughput | 1.49 GB/s | 2.44 GB/s | 3.11 GB/s | 2.53 GB/s | 2.26 GB/s |

hashlib releases the GIL while it hashes a chunk, but the per-chunk Python
work does not, so more than four threads only add contention. Leaves are
hashed without copying (memoryview slices, the v2 tag fed to the hasher
separately) from 4 MB reads, and tensors are split into 64 MB segments so a
large embedding matrix is spread over all threads. Before these changes the
same model built at 1.93 GB/s with four threads. At 8 threads SHA-512 and
BLAKE2b reach about 2.0 GB/s and BLAKE3 2.4 GB/s, so the `[fast]` extra
(BLAKE3) gives no real speedup at 16 KB chunks.

**Large models.** Four threads, SHA-256, 16 KB chunks:

| Model | Size | Shards | Chunks | Build | Peak memory | Full manifest | Load manifest |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B-FP8 | 30.9 GB | 66 | 1,884,612 | 15.0 s | 472 MB | 144 MB | 3.3 s |
| Olmo-3.1-32B (bf16) | 64.5 GB | 14 | 3,934,885 | 36.9 s | 0.99 GB | 299 MB | 7.0 s |

- A model larger than the free page cache is bound by the disk. Reading
  Olmo's shards with four parallel `cat` processes takes 32.2 s (2.00 GB/s);
  building its tree takes 36.9 s (1.75 GB/s), 88% of that ceiling, so more
  parallelism would not help.
- Memory grows with the number of chunks (about 250 bytes each), not with the
  model size: a 70B bf16 model at 16 KB chunks needs about 2 GB.
- Full manifests store every chunk hash (about 76 bytes per chunk in JSON). To
  publish a manifest for a large model, use `--compact` (kilobytes) or a larger
  `--chunk-size`; keep a full manifest locally for chunk-level repair.
