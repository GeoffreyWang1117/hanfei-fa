"""Tests for merkle_verify.flat_tree module."""

import os
import pytest

from merkle_verify.flat_tree import (
    FlatModelTree,
    build_flat_merkle_tree,
    flat_compare,
    TwoLevelModelTree,
    build_two_level_merkle_tree,
    two_level_compare,
)


@pytest.fixture
def flat_blobs():
    return {f"param_{i}": os.urandom(2048) for i in range(4)}


class TestFlatModelTree:
    def test_build(self, flat_blobs):
        tree = build_flat_merkle_tree(flat_blobs, model_name="flat")
        assert tree.model_root is not None
        assert tree.model_name == "flat"
        assert len(tree.param_trees) == 4

    def test_deterministic(self, flat_blobs):
        t1 = build_flat_merkle_tree(flat_blobs)
        t2 = build_flat_merkle_tree(flat_blobs)
        assert t1.model_root == t2.model_root


class TestFlatCompare:
    def test_identical(self, flat_blobs):
        t = build_flat_merkle_tree(flat_blobs)
        result = flat_compare(t, t)
        assert result["change_percentage"] == 0.0

    def test_one_changed(self, flat_blobs):
        t1 = build_flat_merkle_tree(flat_blobs)
        blobs2 = dict(flat_blobs)
        blobs2["param_0"] = os.urandom(2048)
        t2 = build_flat_merkle_tree(blobs2)
        result = flat_compare(t1, t2, detailed=True)
        assert "param_0" in result["changed_params"]
        assert result["change_percentage"] > 0

    def test_not_detailed(self, flat_blobs):
        t1 = build_flat_merkle_tree(flat_blobs)
        blobs2 = dict(flat_blobs)
        blobs2["param_0"] = os.urandom(2048)
        t2 = build_flat_merkle_tree(blobs2)
        result = flat_compare(t1, t2, detailed=False)
        assert len(result["changed_params"]) >= 1

    def test_param_only_in_one(self, flat_blobs):
        t1 = build_flat_merkle_tree(flat_blobs)
        blobs2 = {k: v for k, v in flat_blobs.items() if k != "param_0"}
        blobs2["new_param"] = os.urandom(1024)
        t2 = build_flat_merkle_tree(blobs2)
        result = flat_compare(t1, t2)
        assert "param_0" in result["changed_params"] or "new_param" in result["changed_params"]


class TestTwoLevelAlias:
    def test_alias_is_flat(self):
        assert TwoLevelModelTree is FlatModelTree
        assert build_two_level_merkle_tree is build_flat_merkle_tree
        assert two_level_compare is flat_compare
