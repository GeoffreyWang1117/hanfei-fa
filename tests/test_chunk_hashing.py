"""The segmented, block-wise file hashing schedule must not change any leaf."""

import os
import random

import pytest

from merkle_verify import _chunk_hashing as ch
from merkle_verify.format import FORMAT_V1, FORMAT_V2


@pytest.fixture
def tiny_schedule(monkeypatch):
    # force many segments and read blocks even for small inputs
    monkeypatch.setattr(ch, "SEGMENT_BYTES", 1000)
    monkeypatch.setattr(ch, "READ_BLOCK_BYTES", 300)


@pytest.mark.parametrize("version", [FORMAT_V1, FORMAT_V2])
@pytest.mark.parametrize("seed", range(12))
def test_file_ranges_equal_in_memory(tmp_path, tiny_schedule, version, seed):
    rng = random.Random(seed)
    data = os.urandom(rng.randint(0, 6000))
    path = str(tmp_path / "blob")
    with open(path, "wb") as f:
        f.write(data)
    chunk = rng.choice([1, 7, 64, 97, 256, 1000, 1500])
    leaf = ch.leaf_hasher(version, "sha256")
    ranges, expected = [], {}
    for k in range(4):
        start = rng.randint(0, len(data))
        end = rng.randint(start, len(data))
        ranges.append((k, path, start, end - start))
        expected[k] = ch.hash_buffer(data[start:end], chunk, leaf)
    for workers in (1, 3):
        assert ch.hash_file_ranges(ranges, chunk, leaf, workers=workers) == expected


def test_truncated_file_is_an_error(tmp_path):
    path = str(tmp_path / "blob")
    with open(path, "wb") as f:
        f.write(b"x" * 100)
    with pytest.raises(ValueError, match="unexpected end of file"):
        ch.hash_file_range(path, 50, 100, 16, ch.leaf_hasher(FORMAT_V2, "sha256"))


def test_non_byte_buffers_are_hashed_as_bytes():
    np = pytest.importorskip("numpy")
    a = np.arange(100, dtype=np.float32)
    leaf = ch.leaf_hasher(FORMAT_V2, "sha256")
    assert ch.hash_buffer(memoryview(a), 64, leaf) == ch.hash_buffer(a.tobytes(), 64, leaf)
