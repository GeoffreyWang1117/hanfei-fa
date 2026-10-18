"""Regression tests for the defects fixed in 0.2.1."""

import hashlib
import json
import os
import struct
import subprocess
import sys

import pytest

from merkle_verify import build_model_merkle_tree, check_roots
from merkle_verify._safetensors_io import read_header
from merkle_verify._tensor_bytes import flatten_state_dict, to_bytes
from .helpers import SAMPLE, flip_last_byte, write_safetensors
from merkle_verify.safetensors_adapter import (
    _resolve_hf_snapshot,
    from_hf_repo,
    sign,
    sign_sharded,
    verify,
    verify_sharded,
    verify_tensor,
)


def run_cli(*args):
    r = subprocess.run(
        [sys.executable, "-m", "merkle_verify.cli", *args],
        capture_output=True, text=True, timeout=60,
    )
    return r.returncode, r.stdout, r.stderr


@pytest.fixture
def st_file(tmp_path):
    p = str(tmp_path / "model.safetensors")
    write_safetensors(p, SAMPLE)
    return p


# ---------------------------------------------------------------------------
# expected_root must re-hash the file (it used to only read the sidecar)
# ---------------------------------------------------------------------------

class TestExpectedRootRehashes:
    def test_tampered_file_fails_with_good_root(self, st_file):
        good = sign(st_file).model_root
        flip_last_byte(st_file)
        ok, details = verify(st_file, expected_root=good)
        assert ok is False
        assert details["mode"] == "expected_root"
        assert details["computed_root"] != good
        assert details["changed_params"]  # localised via the sidecar

    def test_no_manifest_needed(self, st_file):
        good = sign(st_file, sidecar=False).model_root
        ok, details = verify(st_file, expected_root=good)
        assert ok is True
        assert "stored_root" not in details

    def test_manifest_disagreeing_with_root_is_flagged(self, st_file):
        sign(st_file)
        ok, details = verify(st_file, expected_root="0" * 64)
        assert ok is False
        assert details["manifest_mismatch"] is True

    def test_check_roots_needs_a_reference(self, st_file):
        tree = sign(st_file, sidecar=False)
        with pytest.raises(ValueError):
            check_roots(tree)

    def test_cli_expected_root(self, st_file):
        good = sign(st_file, sidecar=False).model_root
        rc, out, _ = run_cli("verify", st_file, "--expected-root", good)
        assert rc == 0, out
        flip_last_byte(st_file)
        rc, out, _ = run_cli("verify", st_file, "--expected-root", good)
        assert rc == 1 and "FAIL" in out

    def test_cli_expected_root_generic_file(self, tmp_path):
        p = str(tmp_path / "blob.bin")
        open(p, "wb").write(os.urandom(10_000))
        rc, out, _ = run_cli("hash", p)
        root = out.split()[0]
        assert run_cli("verify", p, "--expected-root", root)[0] == 0
        flip_last_byte(p)
        assert run_cli("verify", p, "--expected-root", root)[0] == 1


# ---------------------------------------------------------------------------
# Built-in safetensors reader: no dependency, any dtype, streaming
# ---------------------------------------------------------------------------

class TestSafetensorsReader:
    def test_bf16_sign_verify(self, st_file):
        tree = sign(st_file)
        assert verify(st_file)[0] is True
        assert verify_tensor(st_file, "layer.1.weight")[0] is True
        assert tree.model_root

    def test_root_equals_in_memory_bytes(self, st_file):
        blobs = {name: raw for name, (_, _, raw) in SAMPLE.items()}
        assert sign(st_file, sidecar=False, format_version=1).model_root == \
            build_model_merkle_tree(blobs, parallel=False, format_version=1).model_root
        # v2 binds dtype and shape, which bare bytes do not carry
        assert sign(st_file, sidecar=False).model_root != \
            build_model_merkle_tree(blobs, parallel=False).model_root

    def test_streaming_matches_for_multi_chunk_tensors(self, tmp_path):
        p = str(tmp_path / "big.safetensors")
        raw = os.urandom(5 * 1024 + 17)
        write_safetensors(p, {"w": ("U8", (len(raw),), raw)})
        for cs in (1024, 4096, 1 << 20):
            assert sign(p, chunk_size=cs, sidecar=False, format_version=1).model_root == \
                build_model_merkle_tree({"w": raw}, chunk_size=cs, parallel=False,
                                        format_version=1).model_root
        np = pytest.importorskip("numpy")
        arr = np.frombuffer(raw, dtype=np.uint8)
        for cs in (1024, 4096, 1 << 20):
            assert sign(p, chunk_size=cs, sidecar=False).model_root == \
                build_model_merkle_tree({"w": arr}, chunk_size=cs, parallel=False).model_root

    def test_empty_tensor(self, tmp_path):
        p = str(tmp_path / "empty.safetensors")
        write_safetensors(p, {"layer.0.weight": ("F32", (0, 4), b""), "layer.0.bias": ("F32", (1,), b"\0" * 4)})
        tree = sign(p)
        assert verify(p)[0] is True
        assert tree.layer_trees["layer.0"].param_trees["layer.0.weight"].num_chunks == 1

    def test_metadata_is_read(self, tmp_path):
        p = str(tmp_path / "m.safetensors")
        write_safetensors(p, SAMPLE, metadata={"format": "pt"})
        assert read_header(p)[1] == {"format": "pt"}

    @pytest.mark.parametrize("payload", [
        b"",
        b"\x01\x02",
        struct.pack("<Q", 10**12) + b"{}",
        struct.pack("<Q", 4) + b"nope",
        struct.pack("<Q", 2) + b"[]",
    ])
    def test_malformed_files_raise_value_error(self, tmp_path, payload):
        p = str(tmp_path / "bad.safetensors")
        open(p, "wb").write(payload)
        with pytest.raises(ValueError):
            read_header(p)

    def test_truncated_data_raises(self, st_file):
        size = os.path.getsize(st_file)
        with open(st_file, "r+b") as f:
            f.truncate(size - 10)
        with pytest.raises(ValueError, match="outside"):
            read_header(st_file)

    def test_cli_malformed_exits_2(self, tmp_path):
        p = str(tmp_path / "bad.safetensors")
        open(p, "wb").write(b"\x00" * 3)
        rc, _, err = run_cli("sign", p)
        assert rc == 2 and "error:" in err

    def test_matches_safetensors_library(self, tmp_path):
        np = pytest.importorskip("numpy")
        st_numpy = pytest.importorskip("safetensors.numpy")
        p = str(tmp_path / "lib.safetensors")
        tensors = {
            "a.weight": np.arange(3000, dtype=np.float32).reshape(30, 100),
            "a.bias": np.arange(7, dtype=np.int64),
            "b.mask": np.array([True, False, True]),
        }
        st_numpy.save_file(tensors, p)
        blobs = {k: v.tobytes() for k, v in tensors.items()}
        assert sign(p, chunk_size=1024, sidecar=False, format_version=1).model_root == \
            build_model_merkle_tree(blobs, chunk_size=1024, parallel=False,
                                    format_version=1).model_root
        assert sign(p, chunk_size=1024, sidecar=False).model_root == \
            build_model_merkle_tree(tensors, chunk_size=1024, parallel=False).model_root


# ---------------------------------------------------------------------------
# Tensors accepted directly; bf16 safe
# ---------------------------------------------------------------------------

class TestTensorInputs:
    def test_numpy_matches_bytes(self):
        np = pytest.importorskip("numpy")
        a = np.arange(1000, dtype=np.float32)
        assert to_bytes(a) == a.tobytes()
        assert build_model_merkle_tree({"x.w": a}, parallel=False, format_version=1).model_root == \
            build_model_merkle_tree({"x.w": a.tobytes()}, parallel=False, format_version=1).model_root

    def test_big_endian_numpy_is_normalised(self):
        np = pytest.importorskip("numpy")
        a = np.arange(10, dtype=">f4")
        assert to_bytes(a) == a.astype("<f4").tobytes()

    def test_torch_state_dict_directly(self):
        torch = pytest.importorskip("torch")
        sd = {"l.0.weight": torch.randn(4, 4), "l.0.bias": torch.randn(4)}
        blobs = {k: v.numpy().tobytes() for k, v in sd.items()}
        assert build_model_merkle_tree(sd, parallel=False, format_version=1).model_root == \
            build_model_merkle_tree(blobs, parallel=False, format_version=1).model_root

    def test_torch_bf16(self):
        torch = pytest.importorskip("torch")
        t = torch.randn(8, 8, dtype=torch.bfloat16)
        raw = to_bytes(t)
        assert len(raw) == 128
        assert torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(8, 8).equal(t)

    def test_torch_state_dict_matches_safetensors_file(self, tmp_path):
        torch = pytest.importorskip("torch")
        st_torch = pytest.importorskip("safetensors.torch")
        sd = {"l.0.weight": torch.randn(64, 64, dtype=torch.bfloat16),
              "l.0.bias": torch.randn(64, dtype=torch.float16)}
        p = str(tmp_path / "m.safetensors")
        st_torch.save_file(sd, p)
        assert sign(p, sidecar=False).model_root == build_model_merkle_tree(sd, parallel=False).model_root

    def test_unsupported_type(self):
        with pytest.raises(TypeError):
            to_bytes([1, 2, 3])


# ---------------------------------------------------------------------------
# PyTorch checkpoints: read-only signing, nested checkpoints
# ---------------------------------------------------------------------------

class TestPyTorchCheckpoints:
    @pytest.fixture
    def torch(self):
        return pytest.importorskip("torch")

    def test_cli_sign_does_not_touch_checkpoint(self, torch, tmp_path):
        p = str(tmp_path / "m.pt")
        torch.save({"l.0.weight": torch.randn(16, 16)}, p)
        before = (hashlib.sha256(open(p, "rb").read()).hexdigest(), os.stat(p).st_mtime_ns)
        rc, out, err = run_cli("sign", p)
        assert rc == 0, err
        after = (hashlib.sha256(open(p, "rb").read()).hexdigest(), os.stat(p).st_mtime_ns)
        assert before == after
        assert os.path.exists(str(tmp_path / "m.merkle.json"))
        assert run_cli("verify", p)[0] == 0

    def test_nested_checkpoint(self, torch, tmp_path):
        from merkle_verify.pytorch_adapter import sign_checkpoint, verify_checkpoint
        ckpt = {"model": {"l.0.weight": torch.randn(4, 4)},
                "optimizer": {"state": {0: {"exp_avg": torch.randn(4, 4)}}},
                "epoch": 3}
        p = str(tmp_path / "ckpt.pt")
        torch.save(ckpt, p)
        tree = sign_checkpoint(p)
        names = {n for lt in tree.layer_trees.values() for n in lt.param_trees}
        assert names == {"model.l.0.weight", "optimizer.state.0.exp_avg"}

        ckpt["optimizer"]["state"][0]["exp_avg"] += 1  # tamper only the nested tensor
        torch.save(ckpt, p)
        ok, details = verify_checkpoint(p)
        assert ok is False
        assert details["changed_params"] == ["optimizer.state.0.exp_avg"]

    def test_flat_state_dict_root_unchanged(self, torch):
        sd = {"l.0.weight": torch.randn(4, 4)}
        assert flatten_state_dict(sd) == sd

    def test_bf16_checkpoint(self, torch, tmp_path):
        from merkle_verify.pytorch_adapter import sign_checkpoint, verify_checkpoint
        p = str(tmp_path / "bf16.pt")
        torch.save({"l.0.weight": torch.randn(8, 8, dtype=torch.bfloat16)}, p)
        sign_checkpoint(p)
        assert verify_checkpoint(p)[0] is True

    def test_tampered_checkpoint_with_good_root(self, torch, tmp_path):
        from merkle_verify.pytorch_adapter import sign_checkpoint, verify_checkpoint
        p = str(tmp_path / "m.pt")
        torch.save({"l.0.weight": torch.zeros(8, 8)}, p)
        good = sign_checkpoint(p).model_root
        torch.save({"l.0.weight": torch.ones(8, 8)}, p)
        ok, details = verify_checkpoint(p, expected_root=good)
        assert ok is False and details["mode"] == "expected_root"

    def test_checkpoint_without_tensors(self, torch, tmp_path):
        from merkle_verify.pytorch_adapter import sign_checkpoint
        p = str(tmp_path / "empty.pt")
        torch.save({"epoch": 1}, p)
        with pytest.raises(ValueError, match="no tensors"):
            sign_checkpoint(p)


# ---------------------------------------------------------------------------
# Sharded models
# ---------------------------------------------------------------------------

def make_sharded(directory, shards, weight_map=None):
    os.makedirs(directory, exist_ok=True)
    wm = {}
    for shard_name, tensors in shards.items():
        write_safetensors(os.path.join(directory, shard_name), tensors)
        for name in tensors:
            wm[name] = shard_name
    if weight_map is not None:
        wm = weight_map
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f)


SHARDS = {
    "model-00001-of-00002.safetensors": {
        "model.layers.0.w": ("F32", (4,), b"\x01" * 16),
        "model.layers.1.w": ("F32", (4,), b"\x02" * 16),
    },
    "model-00002-of-00002.safetensors": {
        "model.layers.2.w": ("BF16", (8,), b"\x03" * 16),
    },
}


class TestSharded:
    def test_sign_verify_roundtrip(self, tmp_path):
        d = str(tmp_path / "m")
        make_sharded(d, SHARDS)
        tree = sign_sharded(d)
        assert len(tree.layer_trees) == 3
        assert verify_sharded(d)[0] is True
        assert verify_sharded(d, expected_root=tree.model_root)[0] is True

    def test_same_root_as_single_file(self, tmp_path):
        d = str(tmp_path / "m")
        make_sharded(d, SHARDS)
        single = str(tmp_path / "single.safetensors")
        write_safetensors(single, {**SHARDS["model-00001-of-00002.safetensors"],
                                   **SHARDS["model-00002-of-00002.safetensors"]})
        assert sign_sharded(d, sidecar=False).model_root == sign(single, sidecar=False).model_root

    def test_tampered_shard_detected(self, tmp_path):
        d = str(tmp_path / "m")
        make_sharded(d, SHARDS)
        sign_sharded(d)
        flip_last_byte(os.path.join(d, "model-00002-of-00002.safetensors"))
        ok, details = verify_sharded(d)
        assert ok is False
        assert details["changed_params"] == ["model.layers.2.w"]

    def test_index_lists_missing_tensor(self, tmp_path):
        d = str(tmp_path / "m")
        wm = {"model.layers.0.w": "model-00001-of-00002.safetensors",
              "model.layers.1.w": "model-00001-of-00002.safetensors",
              "model.layers.2.w": "model-00002-of-00002.safetensors",
              "model.layers.3.w": "model-00002-of-00002.safetensors"}
        make_sharded(d, SHARDS, weight_map=wm)
        with pytest.raises(ValueError, match="not in their declared shard"):
            sign_sharded(d)

    def test_duplicate_tensor_across_shards(self, tmp_path):
        d = str(tmp_path / "m")
        shards = {
            "a.safetensors": {"model.layers.0.w": ("F32", (1,), b"\0" * 4)},
            "b.safetensors": {"model.layers.0.w": ("F32", (1,), b"\1" * 4),
                              "model.layers.1.w": ("F32", (1,), b"\2" * 4)},
        }
        make_sharded(d, shards, weight_map={"model.layers.0.w": "a.safetensors",
                                            "model.layers.1.w": "b.safetensors"})
        with pytest.raises(ValueError, match="appears in both"):
            sign_sharded(d)

    def test_cli_directory(self, tmp_path):
        d = str(tmp_path / "m")
        make_sharded(d, SHARDS)
        rc, out, err = run_cli("sign", d)
        assert rc == 0, err
        assert run_cli("verify", d)[0] == 0
        flip_last_byte(os.path.join(d, "model-00001-of-00002.safetensors"))
        assert run_cli("verify", d)[0] == 1


# ---------------------------------------------------------------------------
# HF cache: no writes into the cache, exact revisions, env vars
# ---------------------------------------------------------------------------

def make_hf_cache(root, repo_id, commits, main=None, branches=None):
    repo = os.path.join(root, "models--" + repo_id.replace("/", "--"))
    for commit in commits:
        snap = os.path.join(repo, "snapshots", commit)
        os.makedirs(snap)
        write_safetensors(os.path.join(snap, "model.safetensors"),
                          {"w": ("U8", (4,), commit[:4].encode())})
    os.makedirs(os.path.join(repo, "refs"))
    for name, commit in dict(branches or {}, **({"main": main} if main else {})).items():
        with open(os.path.join(repo, "refs", name), "w") as f:
            f.write(commit)
    return repo


class TestHFCache:
    def test_does_not_write_into_cache(self, tmp_path):
        repo = make_hf_cache(str(tmp_path), "org/m", ["aaaa1111"], main="aaaa1111")
        before = sorted(os.listdir(os.path.join(repo, "snapshots", "aaaa1111")))
        from_hf_repo("org/m", cache_dir=str(tmp_path))
        assert sorted(os.listdir(os.path.join(repo, "snapshots", "aaaa1111"))) == before

    def test_manifest_path(self, tmp_path):
        make_hf_cache(str(tmp_path), "org/m", ["aaaa1111"], main="aaaa1111")
        out = str(tmp_path / "out.json")
        tree = from_hf_repo("org/m", cache_dir=str(tmp_path), manifest_path=out)
        assert json.load(open(out))["model_root"] == tree.model_root

    def test_unknown_revision_raises(self, tmp_path):
        make_hf_cache(str(tmp_path), "org/m", ["aaaa1111"], main="aaaa1111")
        with pytest.raises(FileNotFoundError, match="not in the local cache"):
            _resolve_hf_snapshot("org/m", revision="deadbeef", cache_dir=str(tmp_path))

    def test_branch_revision(self, tmp_path):
        make_hf_cache(str(tmp_path), "org/m", ["aaaa1111", "bbbb2222"],
                      main="aaaa1111", branches={"dev": "bbbb2222"})
        snap = _resolve_hf_snapshot("org/m", revision="dev", cache_dir=str(tmp_path))
        assert snap.endswith("bbbb2222")
        assert _resolve_hf_snapshot("org/m", cache_dir=str(tmp_path)).endswith("aaaa1111")

    def test_ambiguous_without_main(self, tmp_path):
        make_hf_cache(str(tmp_path), "org/m", ["aaaa1111", "bbbb2222"])
        with pytest.raises(FileNotFoundError, match="pass revision"):
            _resolve_hf_snapshot("org/m", cache_dir=str(tmp_path))

    def test_hf_hub_cache_env(self, tmp_path, monkeypatch):
        make_hf_cache(str(tmp_path), "org/m", ["aaaa1111"], main="aaaa1111")
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        assert _resolve_hf_snapshot("org/m").endswith("aaaa1111")


# ---------------------------------------------------------------------------
# CLI: single-file manifests in diff/info
# ---------------------------------------------------------------------------

class TestCLIManifests:
    def test_diff_flat_manifests(self, tmp_path):
        a, b = str(tmp_path / "a.bin"), str(tmp_path / "b.bin")
        data = os.urandom(40_000)
        open(a, "wb").write(data)
        open(b, "wb").write(data[:-1] + bytes([data[-1] ^ 1]))
        assert run_cli("sign", a)[0] == 0 and run_cli("sign", b)[0] == 0
        rc, out, err = run_cli("diff", str(tmp_path / "a.merkle.json"), str(tmp_path / "b.merkle.json"))
        assert rc == 0, err
        assert out.startswith("1/3 chunks changed")

    def test_diff_mixed_kinds_is_an_error(self, tmp_path, st_file):
        p = str(tmp_path / "x.bin")
        open(p, "wb").write(b"x")
        assert run_cli("diff", st_file, p)[0] == 2

    def test_version(self):
        from merkle_verify import __version__
        rc, out, _ = run_cli("--version")
        assert rc == 0 and __version__ in out


class TestChunkSizeConsistency:
    def test_mismatched_chunk_size_against_manifest_is_an_error(self, st_file):
        sign(st_file, chunk_size=1024)
        with pytest.raises(ValueError, match="differs from the manifest"):
            verify(st_file, chunk_size=4096)

    def test_manifest_chunk_size_is_used_by_default(self, st_file):
        sign(st_file, chunk_size=64)
        assert verify(st_file)[0] is True

    def test_override_allowed_with_expected_root(self, st_file):
        root = sign(st_file, chunk_size=64, sidecar=False).model_root
        sign(st_file, chunk_size=1024)  # sidecar uses another chunk size
        assert verify(st_file, expected_root=root, chunk_size=64)[0] is True
