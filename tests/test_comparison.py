"""Tests for merkle_verify.comparison module."""

import os
import pytest

from merkle_verify.merkle_tree import build_model_merkle_tree, ModelMerkleTree
from merkle_verify.comparison import (
    compare_model_trees,
    estimate_sync_savings,
    ComparisonResult,
)


@pytest.fixture
def paired_trees():
    """Return (tree_a, tree_b) where tree_b has one changed param."""
    blobs = {f"layer.{i}.weight": os.urandom(4096) for i in range(4)}
    blobs["layer.0.bias"] = os.urandom(256)
    tree_a = build_model_merkle_tree(blobs, parallel=False)

    blobs_b = dict(blobs)
    blobs_b["layer.0.weight"] = os.urandom(4096)
    tree_b = build_model_merkle_tree(blobs_b, parallel=False)
    return tree_a, tree_b


class TestComparisonResult:
    def test_to_dict(self):
        r = ComparisonResult()
        d = r.to_dict()
        assert "changed_params" in d
        assert "hash_comparisons" in d

    def test_summary(self):
        r = ComparisonResult(changed_params=["a"], changed_layers=["l"], change_percentage=5.0, hash_comparisons=10)
        s = r.summary()
        assert "1 params" in s
        assert "5.00%" in s


class TestCompareModelTrees:
    def test_identical(self, sample_blobs):
        t = build_model_merkle_tree(sample_blobs, parallel=False)
        diff = compare_model_trees(t, t)
        assert diff["change_percentage"] == 0.0
        assert diff["changed_params"] == []

    def test_one_param_changed(self, paired_trees):
        tree_a, tree_b = paired_trees
        diff = compare_model_trees(tree_a, tree_b, detailed=True)
        assert len(diff["changed_params"]) >= 1
        assert diff["change_percentage"] > 0
        assert diff["hash_comparisons"] > 0

    def test_detailed_false(self, paired_trees):
        tree_a, tree_b = paired_trees
        diff = compare_model_trees(tree_a, tree_b, detailed=False)
        assert "changed_params" in diff

    def test_layer_only_in_one(self):
        blobs_a = {"layer.0.weight": os.urandom(1024)}
        blobs_b = {"layer.1.weight": os.urandom(1024)}
        t_a = build_model_merkle_tree(blobs_a, parallel=False)
        t_b = build_model_merkle_tree(blobs_b, parallel=False)
        diff = compare_model_trees(t_a, t_b)
        assert len(diff["changed_layers"]) >= 1

    def test_chunk_changes_populated(self, paired_trees):
        tree_a, tree_b = paired_trees
        diff = compare_model_trees(tree_a, tree_b, detailed=True)
        # At least one param should have chunk-level changes
        assert len(diff["chunk_changes"]) >= 1


class TestEstimateSyncSavings:
    def test_identical_100_pct_savings(self, sample_blobs):
        t = build_model_merkle_tree(sample_blobs, parallel=False)
        savings = estimate_sync_savings(t, t)
        assert savings["savings_percentage"] == 100.0
        assert savings["incremental_sync_bytes"] == 0

    def test_partial_change(self, paired_trees):
        tree_a, tree_b = paired_trees
        savings = estimate_sync_savings(tree_a, tree_b)
        assert 0 < savings["savings_percentage"] < 100
        assert savings["savings_bytes"] > 0

    def test_keys_present(self, paired_trees):
        savings = estimate_sync_savings(*paired_trees)
        for key in ["full_sync_bytes", "incremental_sync_bytes", "savings_bytes", "savings_percentage"]:
            assert key in savings
