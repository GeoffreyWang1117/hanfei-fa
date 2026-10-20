"""Tests for safetensors adapter."""

import os
import json
import pytest

np = pytest.importorskip("numpy", reason="numpy not installed")
safetensors = pytest.importorskip("safetensors", reason="safetensors package not installed")

from safetensors.numpy import save_file as np_save_file
from merkle_verify.safetensors_adapter import (
    sign,
    verify,
    verify_tensor,
    diff,
    _sidecar_path,
)


@pytest.fixture
def sample_tensors():
    """Create sample numpy tensors."""
    np.random.seed(42)
    return {
        "layer.0.weight": np.random.randn(64, 32).astype(np.float32),
        "layer.0.bias": np.random.randn(64).astype(np.float32),
        "layer.1.weight": np.random.randn(32, 64).astype(np.float32),
        "layer.1.bias": np.random.randn(32).astype(np.float32),
    }


@pytest.fixture
def safetensors_file(tmp_path, sample_tensors):
    """Create a temporary safetensors file."""
    filepath = str(tmp_path / "model.safetensors")
    np_save_file(sample_tensors, filepath)
    return filepath


@pytest.fixture
def signed_file(safetensors_file):
    """Create a signed safetensors file with sidecar."""
    sign(safetensors_file)
    return safetensors_file


class TestSign:
    def test_sign_creates_sidecar(self, safetensors_file):
        tree = sign(safetensors_file)
        sidecar = _sidecar_path(safetensors_file)
        assert os.path.exists(sidecar)
        assert tree.model_root is not None

    def test_sign_model_name(self, safetensors_file):
        tree = sign(safetensors_file, model_name="my-model")
        assert tree.model_name == "my-model"

    def test_sign_default_model_name(self, safetensors_file):
        tree = sign(safetensors_file)
        assert tree.model_name == "model"  # stem of "model.safetensors"

    def test_sign_no_sidecar(self, safetensors_file):
        tree = sign(safetensors_file, sidecar=False)
        assert not os.path.exists(_sidecar_path(safetensors_file))
        assert tree.model_root is not None

    def test_sign_deterministic(self, safetensors_file):
        tree1 = sign(safetensors_file, sidecar=False)
        tree2 = sign(safetensors_file, sidecar=False)
        assert tree1.model_root == tree2.model_root

    def test_sign_has_all_tensors(self, safetensors_file, sample_tensors):
        tree = sign(safetensors_file, sidecar=False)
        all_params = set()
        for layer_tree in tree.layer_trees.values():
            all_params.update(layer_tree.param_trees.keys())
        assert all_params == set(sample_tensors.keys())

    def test_sidecar_json_valid(self, safetensors_file):
        sign(safetensors_file)
        sidecar = _sidecar_path(safetensors_file)
        with open(sidecar) as f:
            data = json.load(f)
        assert "model_root" in data
        assert data["format_version"] == 2
        assert set(data["tensors"]) == {"layer.0.weight", "layer.0.bias",
                                        "layer.1.weight", "layer.1.bias"}


class TestVerify:
    def test_verify_valid(self, signed_file):
        is_valid, details = verify(signed_file)
        assert is_valid is True
        assert details["root_match"] is True

    def test_verify_root_only(self, signed_file):
        tree = sign(signed_file, sidecar=False)
        is_valid, details = verify(signed_file, expected_root=tree.model_root)
        assert is_valid is True
        assert details["mode"] == "expected_root"

    def test_verify_wrong_root(self, signed_file):
        is_valid, details = verify(signed_file, expected_root="0" * 64)
        assert is_valid is False

    def test_verify_missing_manifest(self, safetensors_file):
        is_valid, details = verify(safetensors_file, manifest="/nonexistent.json")
        assert is_valid is False
        assert "error" in details

    def test_verify_detects_tampering(self, signed_file, sample_tensors, tmp_path):
        # Create a tampered version with different weights
        tampered = sample_tensors.copy()
        tampered["layer.0.weight"] = np.zeros_like(tampered["layer.0.weight"])
        tampered_path = str(tmp_path / "tampered.safetensors")
        np_save_file(tampered, tampered_path)

        # Verify tampered file against original manifest
        is_valid, details = verify(
            tampered_path,
            manifest=_sidecar_path(signed_file),
        )
        assert is_valid is False
        assert "changed_params" in details
        assert len(details["changed_params"]) > 0


class TestVerifyTensor:
    def test_verify_single_tensor(self, signed_file):
        is_valid, details = verify_tensor(signed_file, "layer.0.weight")
        assert is_valid is True
        assert details["tensor_name"] == "layer.0.weight"

    def test_verify_all_tensors(self, signed_file, sample_tensors):
        for name in sample_tensors:
            is_valid, _ = verify_tensor(signed_file, name)
            assert is_valid is True

    def test_verify_nonexistent_tensor(self, signed_file):
        is_valid, details = verify_tensor(signed_file, "nonexistent.weight")
        assert is_valid is False
        assert "error" in details


class TestDiff:
    def test_diff_identical(self, safetensors_file):
        result = diff(safetensors_file, safetensors_file)
        assert len(result["changed_params"]) == 0
        assert result["change_percentage"] == 0.0

    def test_diff_modified(self, safetensors_file, sample_tensors, tmp_path):
        modified = sample_tensors.copy()
        modified["layer.1.weight"] = np.random.randn(32, 64).astype(np.float32)
        modified_path = str(tmp_path / "modified.safetensors")
        np_save_file(modified, modified_path)

        result = diff(safetensors_file, modified_path)
        assert "layer.1.weight" in result["changed_params"]
        assert len(result["changed_params"]) >= 1
        assert result["change_percentage"] > 0

    def test_diff_reports_unchanged(self, safetensors_file, sample_tensors, tmp_path):
        modified = sample_tensors.copy()
        modified["layer.0.bias"] = np.zeros_like(modified["layer.0.bias"])
        modified_path = str(tmp_path / "modified.safetensors")
        np_save_file(modified, modified_path)

        result = diff(safetensors_file, modified_path)
        assert "layer.1.weight" in result["unchanged_params"]
