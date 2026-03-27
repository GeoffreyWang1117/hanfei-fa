"""
Chunk size strategies for adaptive Merkle tree construction.

Provides different strategies for determining chunk sizes:
- FixedChunkStrategy: constant chunk size (current behavior)
- AdaptiveChunkStrategy: size-based per-parameter chunk sizes
- TheoreticalOptimalStrategy: based on optimal c* ≈ 1/p theory
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import math

from merkle_verify.chunking import DEFAULT_CHUNK_SIZE


class ChunkStrategy(ABC):
    """Abstract base class for chunk size strategies."""

    @abstractmethod
    def get_chunk_size(self, param_name: str, tensor: Any) -> int:
        """Determine chunk size for a parameter.

        Args:
            param_name: Name of the parameter
            tensor: The tensor (torch.Tensor or numpy array)

        Returns:
            Chunk size in bytes
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for reporting."""
        ...


class FixedChunkStrategy(ChunkStrategy):
    """Fixed chunk size for all parameters (current default behavior)."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self._chunk_size = chunk_size

    def get_chunk_size(self, param_name: str, tensor: Any) -> int:
        return self._chunk_size

    @property
    def name(self) -> str:
        return f"fixed-{self._chunk_size // 1024}KB"


class AdaptiveChunkStrategy(ChunkStrategy):
    """Adaptive chunk size based on parameter tensor size.

    Uses target_chunks to determine chunk size per parameter:
    chunk_size = clamp(param_bytes / target_chunks, min_chunk, max_chunk)

    Effect: small params (3KB bias) -> small chunks (more granular),
            large params (150MB embedding) -> larger chunks (less overhead).
    """

    def __init__(
        self,
        target_chunks: int = 64,
        min_chunk: int = 1024,
        max_chunk: int = 65536,
    ):
        self.target_chunks = target_chunks
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk

    def get_chunk_size(self, param_name: str, tensor: Any) -> int:
        param_bytes = tensor.nelement() * tensor.element_size()
        if param_bytes <= 0:
            return self.min_chunk
        raw = param_bytes / self.target_chunks
        # Round to nearest power of 2 for alignment
        if raw <= self.min_chunk:
            return self.min_chunk
        if raw >= self.max_chunk:
            return self.max_chunk
        # Round to nearest 1024 boundary
        return max(self.min_chunk, ((int(raw) + 511) // 1024) * 1024)

    @property
    def name(self) -> str:
        return f"adaptive-{self.target_chunks}chunks"


class TheoreticalOptimalStrategy(ChunkStrategy):
    """Chunk size based on theoretical optimal c* ≈ 1/p.

    For modification probability p per byte, the optimal chunk size
    that minimizes sync transfer is c* = 1/p.

    In practice, p is estimated from expected modification density.
    """

    def __init__(
        self,
        modification_prob: float = 0.001,
        min_chunk: int = 1024,
        max_chunk: int = 65536,
    ):
        if modification_prob <= 0 or modification_prob > 1:
            raise ValueError("modification_prob must be in (0, 1]")
        self.modification_prob = modification_prob
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self._theoretical_optimal = int(1.0 / modification_prob)

    def get_chunk_size(self, param_name: str, tensor: Any) -> int:
        chunk = self._theoretical_optimal
        # Clamp to bounds
        chunk = max(self.min_chunk, min(self.max_chunk, chunk))
        # Round to 1024 boundary
        return ((chunk + 511) // 1024) * 1024

    @property
    def name(self) -> str:
        return f"theoretical-p{self.modification_prob}"

    @property
    def theoretical_optimal_size(self) -> int:
        """The raw theoretical optimal c* = 1/p before clamping."""
        return self._theoretical_optimal
