"""Tests for the CLI tool."""

import json
import os
import subprocess
import sys
import pytest


def run_cli(*args):
    """Run merkle-verify CLI and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "merkle_verify.cli", *args],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


class TestHash:
    def test_hash_file(self, tmp_path):
        data = os.urandom(50_000)
        filepath = str(tmp_path / "test.bin")
        with open(filepath, "wb") as f:
            f.write(data)

        rc, out, _ = run_cli("hash", filepath)
        assert rc == 0
        parts = out.strip().split()
        assert len(parts[0]) == 64  # SHA256 hex digest

    def test_hash_verbose(self, tmp_path):
        filepath = str(tmp_path / "test.bin")
        with open(filepath, "wb") as f:
            f.write(os.urandom(20_000))

        rc, out, _ = run_cli("-v", "hash", filepath)
        assert rc == 0
        assert "chunks:" in out
        assert "time:" in out


class TestSignVerify:
    def test_sign_and_verify_generic(self, tmp_path):
        filepath = str(tmp_path / "data.bin")
        with open(filepath, "wb") as f:
            f.write(os.urandom(40_000))

        # Sign
        rc, out, _ = run_cli("sign", filepath)
        assert rc == 0
        assert "root:" in out
        sidecar = str(tmp_path / "data.merkle.json")
        assert os.path.exists(sidecar)

        # Verify
        rc, out, _ = run_cli("verify", filepath)
        assert rc == 0
        assert "OK" in out

    def test_verify_detects_tampering(self, tmp_path):
        filepath = str(tmp_path / "data.bin")
        with open(filepath, "wb") as f:
            f.write(os.urandom(40_000))

        run_cli("sign", filepath)

        # Tamper
        with open(filepath, "wb") as f:
            f.write(os.urandom(40_000))

        rc, out, _ = run_cli("verify", filepath)
        assert rc == 1
        assert "FAIL" in out

    def test_verify_missing_manifest(self, tmp_path):
        filepath = str(tmp_path / "orphan.bin")
        with open(filepath, "wb") as f:
            f.write(b"data")

        rc, out, err = run_cli("verify", filepath)
        assert rc == 1


class TestDiff:
    def test_diff_identical(self, tmp_path):
        filepath = str(tmp_path / "a.bin")
        with open(filepath, "wb") as f:
            f.write(os.urandom(30_000))

        rc, out, _ = run_cli("diff", filepath, filepath)
        assert rc == 0
        assert "0/" in out  # 0 chunks changed

    def test_diff_different(self, tmp_path):
        fa = str(tmp_path / "a.bin")
        fb = str(tmp_path / "b.bin")
        with open(fa, "wb") as f:
            f.write(os.urandom(30_000))
        with open(fb, "wb") as f:
            f.write(os.urandom(30_000))

        rc, out, _ = run_cli("diff", fa, fb)
        assert rc == 0
        assert "chunks changed" in out

    def test_diff_json_manifests(self, tmp_path):
        """Diff two .merkle.json files directly."""
        fa = str(tmp_path / "a.bin")
        fb = str(tmp_path / "b.bin")
        with open(fa, "wb") as f:
            f.write(os.urandom(30_000))
        with open(fb, "wb") as f:
            f.write(os.urandom(30_000))

        # We need model-level manifests for JSON diff, skip if too complex
        # Just test that the CLI doesn't crash on JSON input
        from merkle_verify.merkle_tree import build_model_merkle_tree
        blobs_a = {"param.weight": os.urandom(4096)}
        blobs_b = {"param.weight": os.urandom(4096)}
        tree_a = build_model_merkle_tree(blobs_a, model_name="a", parallel=False)
        tree_b = build_model_merkle_tree(blobs_b, model_name="b", parallel=False)
        ja = str(tmp_path / "a.merkle.json")
        jb = str(tmp_path / "b.merkle.json")
        tree_a.to_json(ja)
        tree_b.to_json(jb)

        rc, out, _ = run_cli("diff", ja, jb)
        assert rc == 0
        assert "changed" in out


class TestInfo:
    def test_info(self, tmp_path):
        from merkle_verify.merkle_tree import build_model_merkle_tree
        blobs = {
            "layer.0.weight": os.urandom(4096),
            "layer.0.bias": os.urandom(256),
        }
        tree = build_model_merkle_tree(blobs, model_name="test", parallel=False)
        manifest = str(tmp_path / "test.merkle.json")
        tree.to_json(manifest)

        rc, out, _ = run_cli("info", manifest)
        assert rc == 0
        assert "model: test" in out
        assert "layers:" in out
        assert "params:" in out
