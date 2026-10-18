"""Chunk-level repair from a local or HTTP source."""

import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.parse

import pytest

from merkle_verify.merkle_tree import build_file_merkle_tree
from merkle_verify.repair import RepairError, open_source, repair
from merkle_verify._safetensors_io import read_header
from merkle_verify.safetensors_adapter import sign, sign_sharded

from .helpers import write_safetensors

CS = 64  # small chunks so tests exercise many of them


def run_cli(*args):
    r = subprocess.run([sys.executable, "-m", "merkle_verify.cli", *args],
                       capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout, r.stderr


def make_model(path, seed=0):
    rng = __import__("random").Random(seed)
    tensors = {
        "model.layers.0.w": ("U8", (1000,), bytes(rng.randrange(256) for _ in range(1000))),
        "model.layers.1.w": ("U8", (700,), bytes(rng.randrange(256) for _ in range(700))),
        "model.norm.w": ("U8", (10,), bytes(rng.randrange(256) for _ in range(10))),
    }
    write_safetensors(path, tensors)
    return tensors


def damage(path, offset, n=1):
    with open(path, "r+b") as f:
        f.seek(offset)
        b = bytearray(f.read(n))
        f.seek(offset)
        f.write(bytes(x ^ 0xFF for x in b))


@pytest.fixture
def pair(tmp_path):
    """(damaged copy, good copy, manifest path, tensor infos)"""
    good = str(tmp_path / "good" / "model.safetensors")
    os.makedirs(os.path.dirname(good))
    make_model(good)
    copy = str(tmp_path / "model.safetensors")
    shutil.copyfile(good, copy)
    sign(copy, chunk_size=CS)
    infos, _ = read_header(copy)
    return copy, good, str(tmp_path / "model.merkle.json"), infos


def same(a, b):
    return open(a, "rb").read() == open(b, "rb").read()


class TestLocalRepair:
    def test_repairs_only_damaged_chunks(self, pair):
        copy, good, _, infos = pair
        a, b = infos["model.layers.0.w"], infos["model.layers.1.w"]
        damage(copy, a.offset + 3 * CS + 5)
        damage(copy, b.offset + 10 * CS)  # last, short chunk of a 700-byte tensor
        report = repair(copy, good)
        assert sorted(report.bad_chunks) == [("model.layers.0.w", 3), ("model.layers.1.w", 10)]
        assert report.fetched_bytes == CS + (700 - 10 * CS)
        assert report.verified and same(copy, good)

    def test_nothing_to_do(self, pair):
        copy, good, _, _ = pair
        report = repair(copy, good)
        assert report.bad_chunks == [] and report.fetched_bytes == 0 and report.verified

    def test_adjacent_chunks_are_one_request(self, pair):
        copy, good, _, infos = pair
        a = infos["model.layers.0.w"]
        for k in (4, 5, 6):
            damage(copy, a.offset + k * CS)
        report = repair(copy, good)
        assert len(report.bad_chunks) == 3 and report.requests == 1 and report.verified

    def test_dry_run_writes_nothing(self, pair):
        copy, good, _, infos = pair
        damage(copy, infos["model.norm.w"].offset)
        before = open(copy, "rb").read()
        report = repair(copy, good, dry_run=True)
        assert report.bad_chunks == [("model.norm.w", 0)] and report.fetched_bytes == 10
        assert open(copy, "rb").read() == before and not report.verified

    def test_output_copy(self, pair, tmp_path):
        copy, good, manifest, infos = pair
        damage(copy, infos["model.layers.0.w"].offset)
        before = open(copy, "rb").read()
        out = str(tmp_path / "fixed.safetensors")
        report = repair(copy, good, manifest=manifest, output=out)
        assert report.verified and same(out, good)
        assert open(copy, "rb").read() == before

    def test_untrusted_source_is_never_written(self, pair, tmp_path):
        copy, good, _, infos = pair
        offset = infos["model.layers.0.w"].offset + 2 * CS
        damage(copy, offset)
        damage(good, offset + 1)  # the source is damaged too, differently
        damaged = open(copy, "rb").read()
        with pytest.raises(RepairError, match="does not match the manifest"):
            repair(copy, good)
        assert open(copy, "rb").read() == damaged

    def test_manifest_must_match_expected_root(self, pair):
        copy, good, _, _ = pair
        with pytest.raises(RepairError, match="expected root"):
            repair(copy, good, expected_root="00" * 32)

    def test_damaged_header(self, pair):
        copy, good, _, _ = pair
        damage(copy, 12)  # inside the JSON header
        report = repair(copy, good)
        assert report.header_repaired and report.verified and same(copy, good)

    def test_truncated_file(self, pair):
        copy, good, _, _ = pair
        with open(copy, "r+b") as f:
            f.truncate(os.path.getsize(copy) - 150)
        report = repair(copy, good)
        assert report.verified and same(copy, good)

    def test_compact_manifest_refetches_whole_tensor(self, pair):
        copy, good, _, infos = pair
        sign(copy, chunk_size=CS, include_chunks=False)
        damage(copy, infos["model.layers.1.w"].offset + 100)
        report = repair(copy, good)
        assert report.bad_chunks == [("model.layers.1.w", -1)]
        assert report.fetched_bytes == 700 and report.verified and same(copy, good)

    def test_v1_manifest(self, pair):
        copy, good, _, infos = pair
        sign(copy, chunk_size=CS, format_version=1)
        damage(copy, infos["model.layers.0.w"].offset + 7 * CS)
        report = repair(copy, good)
        assert report.bad_chunks == [("model.layers.0.w", 7)] and report.verified


class TestOtherLayouts:
    def test_sharded_directory(self, tmp_path):
        good_dir, bad_dir = tmp_path / "good", tmp_path / "bad"
        good_dir.mkdir()
        write_safetensors(str(good_dir / "s1.safetensors"), {"model.layers.0.w": ("U8", (300,), os.urandom(300))})
        write_safetensors(str(good_dir / "s2.safetensors"), {"model.layers.1.w": ("U8", (300,), os.urandom(300))})
        (good_dir / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"model.layers.0.w": "s1.safetensors", "model.layers.1.w": "s2.safetensors"}}))
        shutil.copytree(good_dir, bad_dir)
        sign_sharded(str(bad_dir), chunk_size=CS)
        info = read_header(str(bad_dir / "s2.safetensors"))[0]["model.layers.1.w"]
        damage(str(bad_dir / "s2.safetensors"), info.offset + CS)
        report = repair(str(bad_dir), str(good_dir))
        assert report.bad_chunks == [("model.layers.1.w", 1)] and report.verified
        assert same(bad_dir / "s2.safetensors", good_dir / "s2.safetensors")

    def test_plain_file(self, tmp_path):
        good, copy = tmp_path / "good.bin", tmp_path / "data.bin"
        good.write_bytes(os.urandom(5000))
        shutil.copyfile(good, copy)
        tree = build_file_merkle_tree(str(copy), chunk_size=512)
        (tmp_path / "data.merkle.json").write_text(json.dumps(tree.to_dict()))
        damage(str(copy), 2048)
        report = repair(str(copy), str(good))
        assert report.bad_chunks == [("file", 4)] and report.fetched_bytes == 512 and report.verified


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _serve(root, honor_range=True):
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = os.path.join(root, urllib.parse.unquote(self.path.lstrip("/")))
            if not os.path.isfile(path):
                self.send_error(404)
                return
            data = open(path, "rb").read()
            rng = self.headers.get("Range")
            requests.append(rng)
            if rng and honor_range:
                a, b = (int(x) for x in rng.split("=", 1)[1].split("-"))
                body = data[a:b + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
            else:
                body = data
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests


class TestHTTP:
    def test_repair_over_http(self, pair):
        copy, good, _, infos = pair
        server, requests = _serve(os.path.dirname(good))
        try:
            damage(copy, infos["model.layers.0.w"].offset + 9 * CS)
            report = repair(copy, f"http://127.0.0.1:{server.server_port}/")
        finally:
            server.shutdown()
        assert report.verified and same(copy, good)
        assert requests == [f"bytes={infos['model.layers.0.w'].offset + 9 * CS}-"
                            f"{infos['model.layers.0.w'].offset + 10 * CS - 1}"]

    def test_server_without_range_support_is_refused(self, pair):
        copy, good, _, infos = pair
        server, _ = _serve(os.path.dirname(good), honor_range=False)
        try:
            damage(copy, infos["model.norm.w"].offset)
            with pytest.raises(RepairError, match="Range"):
                repair(copy, f"http://127.0.0.1:{server.server_port}/model.safetensors")
        finally:
            server.shutdown()

    def test_hf_source_url_and_token(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        src = open_source("hf://Qwen/Qwen2.5-0.5B@060db64", "model.safetensors")
        assert src.url.endswith("/Qwen/Qwen2.5-0.5B/resolve/060db64/model.safetensors")
        assert "Authorization" not in src.headers
        monkeypatch.setenv("HF_TOKEN", "hf_test_value")
        src = open_source("hf://org/m", "sub dir/x.safetensors")
        assert src.url.endswith("/org/m/resolve/main/sub%20dir/x.safetensors")
        assert src.headers["Authorization"] == "Bearer hf_test_value"


class TestCLI:
    def test_repair_command(self, pair):
        copy, good, _, infos = pair
        damage(copy, infos["model.layers.1.w"].offset)
        rc, out, err = run_cli("repair", copy, "--from", good, "--dry-run")
        assert rc == 0, err
        assert "1/" in out and "would fetch 64 bytes" in out
        rc, out, err = run_cli("repair", copy, "--from", good)
        assert rc == 0, err
        assert "OK" in out and same(copy, good)

    def test_repair_command_bad_source(self, pair):
        copy, good, _, infos = pair
        offset = infos["model.layers.1.w"].offset
        damage(copy, offset)
        damage(good, offset + 1)
        rc, _, err = run_cli("repair", copy, "--from", good)
        assert rc == 1 and "does not match the manifest" in err


class TestHFCheck:
    def _cache(self, tmp_path):
        commit = "c0ffee1234567890"
        snap = tmp_path / "models--org--m" / "snapshots" / commit
        snap.mkdir(parents=True)
        (tmp_path / "models--org--m" / "refs").mkdir()
        (tmp_path / "models--org--m" / "refs" / "main").write_text(commit)
        make_model(str(snap / "model.safetensors"))
        return snap

    def test_published_manifest(self, tmp_path):
        snap = self._cache(tmp_path)
        sign_sharded(str(snap), chunk_size=CS)
        rc, out, err = run_cli("hf-check", "org/m", "--cache-dir", str(tmp_path))
        assert rc == 0, err
        assert "published in org/m@" in out and "OK: org/m@c0ffee123456" in out

    def test_damage_suggests_repair(self, tmp_path):
        snap = self._cache(tmp_path)
        sign_sharded(str(snap), chunk_size=CS)
        info = read_header(str(snap / "model.safetensors"))[0]["model.layers.1.w"]
        damage(str(snap / "model.safetensors"), info.offset + 5)
        rc, out, _ = run_cli("hf-check", "org/m", "--cache-dir", str(tmp_path))
        assert rc == 1
        assert "changed params: ['model.layers.1.w']" in out
        assert f"repair: merkle-verify repair {snap} --from hf://org/m@c0ffee1234567890" in out

    def test_needs_a_reference(self, tmp_path):
        self._cache(tmp_path)
        rc, _, err = run_cli("hf-check", "org/m", "--cache-dir", str(tmp_path))
        assert rc == 2 and "publishes no model.merkle.json" in err

    def test_expected_root(self, tmp_path):
        snap = self._cache(tmp_path)
        root = sign(str(snap / "model.safetensors"), sidecar=False).model_root
        rc, out, err = run_cli("hf-check", "org/m", "--cache-dir", str(tmp_path), "--expected-root", root)
        assert rc == 0, err
