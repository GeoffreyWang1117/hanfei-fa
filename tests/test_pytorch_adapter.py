"""Tests for PyTorch adapter."""

import os
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from merkle_verify.pytorch_adapter import (
    merkle_save,
    merkle_load,
    verify_checkpoint,
    diff_checkpoints,
    _state_dict_to_blobs,
    _sidecar_path,
)


@pytest.fixture
def simple_model():
    """A small PyTorch model."""
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16),
    )
    return model


@pytest.fixture
def simple_state_dict(simple_model):
    return simple_model.state_dict()


@pytest.fixture
def saved_checkpoint(tmp_path, simple_model):
    """Save a model with merkle manifest."""
    filepath = str(tmp_path / "model.pt")
    merkle_save(simple_model, filepath)
    return filepath


class TestStateDictToBlobs:
    def test_converts_tensors(self, simple_state_dict):
        blobs = _state_dict_to_blobs(simple_state_dict)
        assert len(blobs) == len(simple_state_dict)
        for name, data in blobs.items():
            assert isinstance(data, bytes)
            assert len(data) > 0

    def test_deterministic(self, simple_state_dict):
        blobs1 = _state_dict_to_blobs(simple_state_dict)
        blobs2 = _state_dict_to_blobs(simple_state_dict)
        for name in blobs1:
            assert blobs1[name] == blobs2[name]


class TestMerkleSave:
    def test_creates_checkpoint_and_sidecar(self, tmp_path, simple_model):
        filepath = str(tmp_path / "model.pt")
        tree = merkle_save(simple_model, filepath)
        assert os.path.exists(filepath)
        assert os.path.exists(_sidecar_path(filepath))
        assert tree.model_root is not None

    def test_saves_from_state_dict(self, tmp_path, simple_state_dict):
        filepath = str(tmp_path / "model.pt")
        tree = merkle_save(simple_state_dict, filepath)
        assert os.path.exists(filepath)
        assert tree.model_root is not None

    def test_model_name(self, tmp_path, simple_model):
        filepath = str(tmp_path / "model.pt")
        tree = merkle_save(simple_model, filepath, model_name="test-model")
        assert tree.model_name == "test-model"

    def test_deterministic(self, tmp_path, simple_state_dict):
        fp1 = str(tmp_path / "m1.pt")
        fp2 = str(tmp_path / "m2.pt")
        t1 = merkle_save(simple_state_dict, fp1)
        t2 = merkle_save(simple_state_dict, fp2)
        assert t1.model_root == t2.model_root

    def test_has_all_params(self, tmp_path, simple_model, simple_state_dict):
        filepath = str(tmp_path / "model.pt")
        tree = merkle_save(simple_model, filepath)
        all_params = set()
        for lt in tree.layer_trees.values():
            all_params.update(lt.param_trees.keys())
        assert all_params == set(simple_state_dict.keys())


class TestMerkleLoad:
    def test_load_with_verify(self, saved_checkpoint):
        state_dict, details = merkle_load(saved_checkpoint)
        assert state_dict is not None
        assert details["verified"] is True

    def test_load_without_verify(self, saved_checkpoint):
        state_dict, details = merkle_load(saved_checkpoint, verify=False)
        assert state_dict is not None
        assert details is None

    def test_load_missing_manifest(self, tmp_path, simple_state_dict):
        filepath = str(tmp_path / "bare.pt")
        torch.save(simple_state_dict, filepath)
        state_dict, details = merkle_load(filepath)
        assert state_dict is not None
        assert details["verified"] is False
        assert "error" in details

    def test_load_detects_tampering(self, saved_checkpoint, tmp_path):
        # Load, modify, re-save without updating manifest
        sd, _ = merkle_load(saved_checkpoint, verify=False)
        sd["0.weight"] = torch.zeros_like(sd["0.weight"])
        torch.save(sd, saved_checkpoint)

        _, details = merkle_load(saved_checkpoint)
        assert details["verified"] is False
        assert "changed_params" in details


class TestVerifyCheckpoint:
    def test_verify_valid(self, saved_checkpoint):
        is_valid, details = verify_checkpoint(saved_checkpoint)
        assert is_valid is True

    def test_verify_root_only(self, saved_checkpoint):
        tree = merkle_save(
            torch.load(saved_checkpoint, weights_only=True),
            saved_checkpoint,
        )
        is_valid, details = verify_checkpoint(
            saved_checkpoint, expected_root=tree.model_root
        )
        assert is_valid is True
        assert details["mode"] == "root_only"

    def test_verify_wrong_root(self, saved_checkpoint):
        is_valid, _ = verify_checkpoint(saved_checkpoint, expected_root="0" * 64)
        assert is_valid is False


class TestDiffCheckpoints:
    def test_diff_identical(self, saved_checkpoint):
        result = diff_checkpoints(saved_checkpoint, saved_checkpoint)
        assert len(result["changed_params"]) == 0

    def test_diff_modified(self, saved_checkpoint, tmp_path, simple_state_dict):
        modified = {k: torch.randn_like(v) for k, v in simple_state_dict.items()}
        modified_path = str(tmp_path / "modified.pt")
        torch.save(modified, modified_path)

        result = diff_checkpoints(saved_checkpoint, modified_path)
        assert len(result["changed_params"]) > 0
        assert result["change_percentage"] > 0
