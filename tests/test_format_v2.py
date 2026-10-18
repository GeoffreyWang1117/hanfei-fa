"""Manifest format v2: agreement with the independent reference implementation,
the weaknesses of v1 that v2 must close, proofs, manifests and compatibility."""

import io
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from merkle_verify import (
    FORMAT_V1,
    FORMAT_V2,
    MerkleTree,
    ModelMerkleTree,
    RawTensor,
    build_model_merkle_tree,
    compare_model_trees,
    compute_hash,
    hash_leaf,
    hash_pair,
    set_default_algorithm,
)
from merkle_verify import format as fmt
from merkle_verify.safetensors_adapter import sign, verify, verify_tensor

from . import reference_v2 as ref
from .helpers import SAMPLE, flip_last_byte, write_safetensors

VECTORS = json.loads((Path(__file__).parent / "vectors" / "format_v2.json").read_text())
ALGORITHMS = ["sha256", "sha512", "sha3_256", "blake2b"]


def case_inputs(case):
    out = {}
    for name, (seed, n, dtype, shape) in case["tensors"].items():
        data = ref.pattern(seed, n)
        out[name] = data if dtype is None else RawTensor(data, dtype, shape)
    return out


def run_cli(*args):
    r = subprocess.run([sys.executable, "-m", "merkle_verify.cli", *args],
                       capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout, r.stderr


# ---------------------------------------------------------------------------
# Agreement with the reference implementation
# ---------------------------------------------------------------------------

def test_vectors_are_current():
    """The committed vectors are exactly what the reference implementation produces."""
    assert ref.build_vectors() == VECTORS


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda c: c["id"])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("parallel", [False, True])
def test_model_roots_match_reference(case, algorithm, parallel):
    tree = build_model_merkle_tree(case_inputs(case), chunk_size=case["chunk_size"],
                                   parallel=parallel, num_workers=4, algorithm=algorithm)
    expected = case["roots"][algorithm]
    assert tree.model_root == expected["model_root"]
    for name, pt in tree.iter_params():
        assert pt.root_hash == expected["chunk_roots"][name]
        assert pt.tensor_root(name) == expected["tensor_roots"][name]


@pytest.mark.parametrize("case", VECTORS["chunk_trees"], ids=lambda c: str(c["seed_len"][1]))
def test_chunk_roots_match_reference(case):
    data = ref.pattern(*case["seed_len"])
    assert MerkleTree.from_bytes(data, case["chunk_size"]).root_hash == case["root_sha256"]
    assert MerkleTree.from_file(io.BytesIO(data), case["chunk_size"]).root_hash == case["root_sha256"]


def test_rfc6962_shape_for_many_sizes():
    for n in range(1, 70):
        data = os.urandom(n * 3)
        assert MerkleTree.from_bytes(data, 3).root_hash == ref.chunk_root("sha256", data, 3).hex()


def test_blake3_matches_reference():
    pytest.importorskip("blake3")
    case = VECTORS["cases"][0]
    tensors = {n: (ref.pattern(s, k), dt, None if sh is None else tuple(sh))
               for n, (s, k, dt, sh) in case["tensors"].items()}
    tree = build_model_merkle_tree(case_inputs(case), chunk_size=case["chunk_size"],
                                   algorithm="blake3", parallel=False)
    assert tree.model_root == ref.model_root("blake3", tensors, case["chunk_size"]).hex()


# ---------------------------------------------------------------------------
# The v1 weaknesses are closed
# ---------------------------------------------------------------------------

class TestV1WeaknessesClosed:
    def test_second_preimage_across_levels(self):
        a, b = hash_leaf(b"a"), hash_leaf(b"b")
        two = MerkleTree([a, b])
        forged = MerkleTree.from_bytes(bytes.fromhex(a) + bytes.fromhex(b), chunk_size=64)
        assert forged.root_hash != two.root_hash
        # in v1 the analogous forgery works
        a1, b1 = compute_hash(b"a"), compute_hash(b"b")
        assert compute_hash((a1 + b1).encode()) == hash_pair(a1, b1)

    def test_duplicated_last_chunk_changes_root(self):
        a, b, c = (hash_leaf(x) for x in (b"a", b"b", b"c"))
        assert MerkleTree([a, b, c]).root_hash != MerkleTree([a, b, c, c]).root_hash
        a1, b1, c1 = (compute_hash(x) for x in (b"a", b"b", b"c"))
        assert MerkleTree([a1, b1, c1], version=1).root_hash == \
            MerkleTree([a1, b1, c1, c1], version=1).root_hash

    def test_order_preserving_rename_changes_root(self):
        m1 = {"layer.0.weight": b"\x01" * 64, "layer.0.bias": b"\x02" * 8}
        m2 = {"layer.0.zeta": b"\x01" * 64, "layer.0.bias": b"\x02" * 8}
        assert build_model_merkle_tree(m1, parallel=False).model_root != \
            build_model_merkle_tree(m2, parallel=False).model_root
        assert build_model_merkle_tree(m1, parallel=False, format_version=1).model_root == \
            build_model_merkle_tree(m2, parallel=False, format_version=1).model_root

    @pytest.mark.parametrize("other", [
        RawTensor(b"\x00" * 64, "I32", (16,)),     # dtype reinterpretation
        RawTensor(b"\x00" * 64, "F32", (4, 4)),    # reshape
        RawTensor(b"\x00" * 64, "F32", (16, 1)),
        b"\x00" * 64,                              # opaque blob
    ])
    def test_dtype_and_shape_are_bound(self, other):
        base = {"w": RawTensor(b"\x00" * 64, "F32", (16,))}
        assert build_model_merkle_tree(base, parallel=False).model_root != \
            build_model_merkle_tree({"w": other}, parallel=False).model_root

    def test_layer_grouping_does_not_enter_the_root(self):
        tensors = {"model.layers.0.w": b"\x01" * 8, "model.layers.1.w": b"\x02" * 8}
        tree = build_model_merkle_tree(tensors, parallel=False)
        roots = [t.tensor_root(n) for n, t in sorted(tree.iter_params())]
        assert tree.model_root == fmt.model_root(roots)
        assert len(tree.layer_trees) == 2

    def test_tensor_count_is_bound(self):
        one = build_model_merkle_tree({"a": b"x"}, parallel=False)
        assert one.model_root != one.param_root("a")

    def test_algorithm_is_recorded_and_used(self):
        tree = build_model_merkle_tree({"a.w": b"x" * 100}, parallel=False, algorithm="sha512")
        set_default_algorithm("sha256")
        data = tree.to_dict()
        assert data["algorithm"] == "sha512"
        restored = ModelMerkleTree.from_dict(data)
        assert restored.model_root == tree.model_root and restored.verify()

    def test_proof_survives_default_algorithm_change(self):
        set_default_algorithm("sha3_256")
        tree = MerkleTree.from_bytes(os.urandom(1000), 100)
        set_default_algorithm("sha256")
        assert all(tree.get_proof(i).verify() for i in range(tree.num_chunks))


# ---------------------------------------------------------------------------
# Proofs
# ---------------------------------------------------------------------------

class TestProofs:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13, 31, 32, 33])
    def test_chunk_proofs(self, n):
        data = os.urandom(n * 10)
        tree = MerkleTree.from_bytes(data, 10)
        for i in range(n):
            proof = tree.get_proof(i)
            assert proof.verify()
            assert proof.verify_chunk(data[i * 10:(i + 1) * 10])
            assert not proof.verify_chunk(b"\xff" * 10)

    def test_proof_index_is_checked(self):
        tree = MerkleTree.from_bytes(os.urandom(80), 10)
        proof = tree.get_proof(3)
        proof.chunk_index = 4
        assert not proof.verify()

    def test_proof_size_is_checked(self):
        tree = MerkleTree.from_bytes(os.urandom(80), 10)
        proof = tree.get_proof(3)
        proof.num_chunks = 9
        assert not proof.verify()

    def test_tensor_and_chunk_proofs(self):
        tensors = {f"model.layers.{i}.w": RawTensor(os.urandom(100 + i), "U8", (100 + i,))
                   for i in range(7)}
        tree = build_model_merkle_tree(tensors, chunk_size=16, parallel=False)
        for name in tensors:
            assert tree.get_tensor_proof(name).verify()
        name = "model.layers.3.w"
        raw = tensors[name].data
        proof = tree.get_chunk_proof(name, 2)
        assert proof.verify(raw[32:48])
        assert not proof.verify(raw[48:64])

    def test_tensor_proof_rejects_other_metadata(self):
        tree = build_model_merkle_tree({"a": RawTensor(b"\0" * 8, "F32", (2,)), "b": b"x"},
                                       parallel=False)
        proof = tree.get_tensor_proof("a")
        proof.dtype = "I32"
        assert not proof.verify()

    def test_tensor_proof_rejects_other_root(self):
        tree = build_model_merkle_tree({"a": b"1", "b": b"2"}, parallel=False)
        proof = tree.get_tensor_proof("a")
        proof.model_root = build_model_merkle_tree({"a": b"1", "b": b"3"}, parallel=False).model_root
        assert not proof.verify()

    def test_v1_tree_has_no_tensor_proofs(self):
        tree = build_model_merkle_tree({"a": b"1"}, parallel=False, format_version=1)
        with pytest.raises(ValueError):
            tree.get_tensor_proof("a")


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

class TestDiff:
    @pytest.mark.parametrize("seed", range(20))
    def test_tree_walk_equals_linear_scan(self, seed):
        rng = random.Random(seed)
        n = rng.randint(1, 120)
        a = [hash_leaf(bytes([i % 256, i // 256])) for i in range(n)]
        b = list(a)
        changed = sorted(rng.sample(range(n), rng.randint(0, min(n, 6))))
        for i in changed:
            b[i] = hash_leaf(b"changed" + bytes([i % 256]))
        got, comparisons = MerkleTree(a).diff_tree(MerkleTree(b))
        assert got == changed
        if not changed:
            assert comparisons == 1

    def test_different_lengths(self):
        a = MerkleTree.from_bytes(b"x" * 100, 10)
        b = MerkleTree.from_bytes(b"x" * 95 + b"y" * 25, 10)
        changed, _ = a.diff_tree(b)
        assert changed == [9, 10, 11]

    def test_cross_format_is_an_error(self):
        v1 = MerkleTree([compute_hash(b"a")], version=1)
        v2 = MerkleTree([hash_leaf(b"a")])
        with pytest.raises(ValueError, match="format"):
            v1.diff_tree(v2)
        m1 = build_model_merkle_tree({"a": b"1"}, parallel=False, format_version=1)
        m2 = build_model_merkle_tree({"a": b"1"}, parallel=False)
        with pytest.raises(ValueError):
            compare_model_trees(m1, m2)

    def test_cross_algorithm_and_chunk_size_are_errors(self):
        a = MerkleTree.from_bytes(b"x" * 100, 10)
        with pytest.raises(ValueError, match="sha512"):
            a.diff_tree(MerkleTree.from_bytes(b"x" * 100, 10, algorithm="sha512"))
        with pytest.raises(ValueError, match="chunk sizes"):
            a.diff_tree(MerkleTree.from_bytes(b"x" * 100, 20))

    def test_metadata_change_shows_as_changed_param(self):
        a = build_model_merkle_tree({"l.0.w": RawTensor(b"\0" * 16, "F32", (4,))}, parallel=False)
        b = build_model_merkle_tree({"l.0.w": RawTensor(b"\0" * 16, "I32", (4,))}, parallel=False)
        result = compare_model_trees(a, b)
        assert result["changed_params"] == ["l.0.w"]


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

def _manifest(tmp_path, include_chunks=True):
    tensors = {"model.layers.0.w": RawTensor(os.urandom(300), "U8", (300,)),
               "model.layers.1.w": RawTensor(os.urandom(200), "U8", (200,))}
    tree = build_model_merkle_tree(tensors, chunk_size=64, parallel=False)
    path = tmp_path / "m.merkle.json"
    tree.to_json(str(path), include_chunks=include_chunks)
    return tree, path


class TestManifests:
    def test_roundtrip(self, tmp_path):
        tree, path = _manifest(tmp_path)
        data = json.loads(path.read_text())
        assert data["format"] == "merkle-weight-verify" and data["format_version"] == 2
        assert data["num_tensors"] == 2
        restored = ModelMerkleTree.from_json(str(path))
        assert restored.model_root == tree.model_root
        assert restored.verify()
        assert restored.get_tensor_proof("model.layers.1.w").verify()

    @pytest.mark.parametrize("mutate", [
        lambda d: d["tensors"]["model.layers.0.w"]["chunk_hashes"].__setitem__(0, "00" * 32),
        lambda d: d["tensors"]["model.layers.0.w"].__setitem__("tensor_root", "00" * 32),
        lambda d: d["tensors"]["model.layers.0.w"].__setitem__("dtype", "I8"),
        lambda d: d["tensors"]["model.layers.0.w"].__setitem__("shape", [3, 100]),
        lambda d: d["tensors"]["model.layers.0.w"].__setitem__("num_chunks", 99),
        lambda d: d.__setitem__("model_root", "00" * 32),
        lambda d: d.__setitem__("num_tensors", 3),
        lambda d: d["tensors"].__setitem__("model.layers.9.w", d["tensors"]["model.layers.0.w"]),
    ], ids=["chunk", "tensor_root", "dtype", "shape", "num_chunks", "model_root",
            "num_tensors", "extra_tensor"])
    def test_inconsistent_manifest_is_rejected(self, tmp_path, mutate):
        _, path = _manifest(tmp_path)
        data = json.loads(path.read_text())
        mutate(data)
        with pytest.raises(ValueError, match="inconsistent"):
            ModelMerkleTree.from_dict(data)

    def test_unknown_format_version_is_rejected(self, tmp_path):
        _, path = _manifest(tmp_path)
        data = json.loads(path.read_text())
        data["format_version"] = 3
        with pytest.raises(ValueError, match="unsupported"):
            ModelMerkleTree.from_dict(data)

    def test_compact_manifest(self, tmp_path):
        tree, path = _manifest(tmp_path, include_chunks=False)
        data = json.loads(path.read_text())
        assert all("chunk_hashes" not in e for e in data["tensors"].values())
        restored = ModelMerkleTree.from_json(str(path))
        assert restored.model_root == tree.model_root
        assert restored.get_tensor_proof("model.layers.0.w").verify()
        assert path.stat().st_size < 2000

    def test_v1_manifests_still_load(self, tmp_path):
        tree = build_model_merkle_tree({"a.w": b"x" * 50}, parallel=False, format_version=1)
        path = tmp_path / "v1.json"
        tree.to_json(str(path))
        restored = ModelMerkleTree.from_json(str(path))
        assert restored.format_version == FORMAT_V1 and restored.verify()


# ---------------------------------------------------------------------------
# Adapters and CLI
# ---------------------------------------------------------------------------

@pytest.fixture
def st_file(tmp_path):
    p = str(tmp_path / "model.safetensors")
    write_safetensors(p, SAMPLE)
    return p


class TestAdaptersV2:
    def test_sign_defaults_to_v2(self, st_file):
        tree = sign(st_file)
        assert tree.format_version == FORMAT_V2
        assert verify(st_file)[0] is True
        assert verify_tensor(st_file, "layer.1.weight")[0] is True

    def test_v1_sidecar_still_verifies(self, st_file):
        sign(st_file, format_version=1)
        ok, details = verify(st_file)
        assert ok is True and details["format_version"] == 1
        flip_last_byte(st_file)
        assert verify(st_file)[0] is False

    def test_manifest_algorithm_beats_process_default(self, st_file):
        sign(st_file, algorithm="sha512")
        set_default_algorithm("sha3_256")
        ok, details = verify(st_file)
        assert ok is True and details["algorithm"] == "sha512"

    def test_expected_root_with_algorithm_prefix(self, st_file):
        root = sign(st_file, sidecar=False, algorithm="blake2b").model_root
        assert verify(st_file, expected_root=f"blake2b:{root}")[0] is True
        assert verify(st_file, expected_root=root)[0] is False  # would be hashed with sha256
        with pytest.raises(ValueError, match="conflicts"):
            verify(st_file, expected_root=f"blake2b:{root}", algorithm="sha256")

    def test_conflicting_explicit_format_is_an_error(self, st_file):
        sign(st_file)
        with pytest.raises(ValueError, match="differs from the manifest"):
            verify(st_file, format_version=1)

    def test_tampered_manifest_is_a_failure_not_a_crash(self, st_file, tmp_path):
        sign(st_file)
        sidecar = tmp_path / "model.merkle.json"
        data = json.loads(sidecar.read_text())
        data["model_root"] = "00" * 32
        sidecar.write_text(json.dumps(data))
        ok, details = verify(st_file)
        assert ok is False and "invalid manifest" in details["error"]

    def test_compact_sidecar_localises_to_tensors(self, st_file):
        sign(st_file, include_chunks=False)
        flip_last_byte(st_file)
        ok, details = verify(st_file)
        assert ok is False
        assert details["changed_params"] == ["layer.1.weight"]

    def test_total_params_exact(self, st_file):
        assert sign(st_file, sidecar=False).total_params == 64 + 8 + 64


class TestCLIV2:
    def test_info_shows_format(self, st_file, tmp_path):
        assert run_cli("sign", st_file)[0] == 0
        rc, out, _ = run_cli("info", str(tmp_path / "model.merkle.json"))
        assert rc == 0 and "format: v2, sha256" in out

    def test_format_version_1(self, st_file, tmp_path):
        assert run_cli("--format-version", "1", "sign", st_file)[0] == 0
        data = json.loads((tmp_path / "model.merkle.json").read_text())
        assert "layer_trees" in data and "format_version" not in data
        assert run_cli("verify", st_file)[0] == 0

    def test_compact_and_algorithm(self, st_file, tmp_path):
        rc, _, err = run_cli("--algorithm", "sha512", "sign", "--compact", st_file)
        assert rc == 0, err
        data = json.loads((tmp_path / "model.merkle.json").read_text())
        assert data["algorithm"] == "sha512"
        assert all("chunk_hashes" not in e for e in data["tensors"].values())
        assert run_cli("verify", st_file)[0] == 0

    def test_generic_file_v2(self, tmp_path):
        p = tmp_path / "blob.bin"
        p.write_bytes(os.urandom(5000))
        assert run_cli("sign", str(p))[0] == 0
        data = json.loads((tmp_path / "blob.merkle.json").read_text())
        assert data["format_version"] == 2 and data["num_bytes"] == 5000
        assert run_cli("verify", str(p))[0] == 0
        flip_last_byte(str(p))
        rc, out, _ = run_cli("verify", str(p))
        assert rc == 1 and "changed chunks: [0]" in out

    def test_generic_v1_manifest_still_verifies(self, tmp_path):
        p = tmp_path / "old.bin"
        p.write_bytes(os.urandom(40_000))
        assert run_cli("--format-version", "1", "sign", str(p))[0] == 0
        data = json.loads((tmp_path / "old.merkle.json").read_text())
        assert "format_version" not in data
        assert run_cli("verify", str(p))[0] == 0
        flip_last_byte(str(p))
        rc, out, _ = run_cli("verify", str(p))
        assert rc == 1 and "changed chunks: [2]" in out


class TestHostileInputs:
    @pytest.mark.parametrize("field,value", [("num_bytes", -1), ("chunk_size", 0),
                                             ("shape", [-3, 100]), ("num_bytes", 2 ** 64)])
    def test_out_of_range_manifest_values_are_rejected(self, tmp_path, field, value):
        _, path = _manifest(tmp_path)
        data = json.loads(path.read_text())
        data["tensors"]["model.layers.0.w"][field] = value
        with pytest.raises(ValueError):
            ModelMerkleTree.from_dict(data)

    def test_hostile_sidecar_is_a_failed_verification(self, st_file, tmp_path):
        sign(st_file)
        sidecar = tmp_path / "model.merkle.json"
        data = json.loads(sidecar.read_text())
        data["tensors"]["layer.0.bias"]["shape"] = [-1]
        sidecar.write_text(json.dumps(data))
        ok, details = verify(st_file)
        assert ok is False and "invalid manifest" in details["error"]

    def test_shape_inconsistent_with_byte_range(self, tmp_path):
        p = str(tmp_path / "bad.safetensors")
        write_safetensors(p, {"w": ("F32", (4, 4), b"\0" * 60)})
        with pytest.raises(ValueError, match="occupies"):
            sign(p)
