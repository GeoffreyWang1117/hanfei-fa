"""Tests for merkle_verify.strategies module."""

import pytest
import numpy as np

from merkle_verify.strategies import (
    FixedChunkStrategy,
    AdaptiveChunkStrategy,
    TheoreticalOptimalStrategy,
)


class FakeTensor:
    """Lightweight stand-in for torch.Tensor with nelement/element_size."""
    def __init__(self, numel: int, elem_size: int = 4):
        self._numel = numel
        self._elem_size = elem_size

    def nelement(self):
        return self._numel

    def element_size(self):
        return self._elem_size


class TestFixedChunkStrategy:
    def test_default(self):
        s = FixedChunkStrategy()
        assert s.get_chunk_size("any", None) == 16384

    def test_custom(self):
        s = FixedChunkStrategy(4096)
        assert s.get_chunk_size("any", None) == 4096

    def test_name(self):
        s = FixedChunkStrategy(8192)
        assert "8" in s.name and "KB" in s.name


class TestAdaptiveChunkStrategy:
    def test_small_tensor_gets_min(self):
        s = AdaptiveChunkStrategy(target_chunks=64, min_chunk=1024, max_chunk=65536)
        t = FakeTensor(100, 4)  # 400 bytes
        assert s.get_chunk_size("bias", t) == 1024

    def test_large_tensor_gets_max(self):
        s = AdaptiveChunkStrategy(target_chunks=64, min_chunk=1024, max_chunk=65536)
        t = FakeTensor(100_000_000, 4)  # 400 MB
        assert s.get_chunk_size("embedding", t) == 65536

    def test_medium_tensor_between_bounds(self):
        s = AdaptiveChunkStrategy(target_chunks=64, min_chunk=1024, max_chunk=65536)
        t = FakeTensor(65536, 4)  # 256 KB → ~4 KB per chunk
        cs = s.get_chunk_size("fc.weight", t)
        assert 1024 <= cs <= 65536
        assert cs % 1024 == 0  # aligned

    def test_zero_elements(self):
        s = AdaptiveChunkStrategy()
        t = FakeTensor(0, 4)
        assert s.get_chunk_size("empty", t) == s.min_chunk

    def test_name(self):
        s = AdaptiveChunkStrategy(target_chunks=32)
        assert "32" in s.name


class TestTheoreticalOptimalStrategy:
    def test_basic(self):
        s = TheoreticalOptimalStrategy(modification_prob=0.001)
        cs = s.get_chunk_size("w", FakeTensor(1000))
        assert 1024 <= cs <= 65536

    def test_theoretical_size(self):
        s = TheoreticalOptimalStrategy(modification_prob=0.01)
        assert s.theoretical_optimal_size == 100

    def test_high_p_gets_min(self):
        s = TheoreticalOptimalStrategy(modification_prob=1.0, min_chunk=1024)
        cs = s.get_chunk_size("w", FakeTensor(1000))
        assert cs == 1024

    def test_low_p_gets_max(self):
        s = TheoreticalOptimalStrategy(modification_prob=0.00001, max_chunk=65536)
        cs = s.get_chunk_size("w", FakeTensor(1000))
        assert cs == 65536

    def test_invalid_prob_raises(self):
        with pytest.raises(ValueError):
            TheoreticalOptimalStrategy(modification_prob=0)
        with pytest.raises(ValueError):
            TheoreticalOptimalStrategy(modification_prob=-0.1)

    def test_name(self):
        s = TheoreticalOptimalStrategy(modification_prob=0.005)
        assert "0.005" in s.name
