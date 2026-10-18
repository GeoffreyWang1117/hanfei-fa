"""
Chunk hashing hot path.

Every root in this package is built from per-chunk leaf hashes, so this is
where the time goes. Three things keep it fast:

* no copies: chunks are memoryview slices of one buffer, and the v2 domain
  tag is fed to the hasher separately instead of being concatenated;
* large reads: files are read 4 MB at a time into a reusable buffer;
* balanced parallelism: byte ranges are split into 64 MB segments, so one
  huge embedding matrix does not occupy a single thread. hashlib releases the
  GIL while hashing each chunk.

The leaves produced are exactly those of the specification; only the
schedule differs.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Hashable, List, Sequence, Tuple

from merkle_verify.format import FORMAT_V1, TAG_LEAF
from merkle_verify.hasher import hasher_factory

READ_BLOCK_BYTES = 4 << 20
SEGMENT_BYTES = 64 << 20

LeafFn = Callable[[Any], str]

#: (key, path, offset, nbytes): a byte range of a file whose chunks to hash
FileRange = Tuple[Hashable, str, int, int]


def leaf_hasher(version: int, algorithm) -> LeafFn:
    """A function mapping one chunk (bytes-like) to its hex leaf hash."""
    ctor = hasher_factory(algorithm)
    if version == FORMAT_V1:
        def leaf(chunk) -> str:
            h = ctor()
            h.update(chunk)
            return h.hexdigest()
    else:
        def leaf(chunk) -> str:
            h = ctor()
            h.update(TAG_LEAF)
            h.update(chunk)
            return h.hexdigest()
    return leaf


def hash_buffer(data, chunk_size: int, leaf: LeafFn) -> List[str]:
    """Leaves of an in-memory buffer. An empty buffer is one empty chunk."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    mv = memoryview(data)
    if mv.itemsize != 1 or mv.format != "B":
        mv = mv.cast("B")
    n = len(mv)
    if n == 0:
        return [leaf(b"")]
    return [leaf(mv[i:i + chunk_size]) for i in range(0, n, chunk_size)]


def _aligned(size: int, chunk_size: int) -> int:
    return max(chunk_size, (size // chunk_size) * chunk_size)


def hash_file_range(path: str, offset: int, nbytes: int, chunk_size: int, leaf: LeafFn) -> List[str]:
    """Leaves of ``nbytes`` bytes of ``path`` starting at ``offset``, chunked from ``offset``.

    Returns an empty list for ``nbytes == 0``. Raises ValueError if the file ends early.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    out: List[str] = []
    if nbytes <= 0:
        return out
    block = bytearray(min(_aligned(READ_BLOCK_BYTES, chunk_size), _aligned(nbytes, chunk_size)))
    mv = memoryview(block)
    remaining = nbytes
    with open(path, "rb", buffering=0) as f:
        f.seek(offset)
        while remaining > 0:
            want = min(len(block), remaining)
            got = 0
            while got < want:
                n = f.readinto(mv[got:want])
                if not n:
                    raise ValueError(
                        f"{path}: unexpected end of file at offset {offset + nbytes - remaining + got}"
                    )
                got += n
            out.extend(leaf(mv[i:min(i + chunk_size, want)]) for i in range(0, want, chunk_size))
            remaining -= want
    return out


def hash_file_ranges(
    ranges: Sequence[FileRange],
    chunk_size: int,
    leaf: LeafFn,
    workers: int = 1,
) -> Dict[Hashable, List[str]]:
    """Leaves for several byte ranges, hashed in parallel 64 MB segments.

    Every key gets at least one leaf: an empty range is one empty chunk.
    """
    segment = _aligned(SEGMENT_BYTES, chunk_size)
    tasks: List[Tuple[Hashable, str, int, int]] = []
    for key, path, offset, nbytes in ranges:
        if nbytes == 0:
            tasks.append((key, path, offset, 0))
        for start in range(0, nbytes, segment):
            tasks.append((key, path, offset + start, min(segment, nbytes - start)))

    def run(task: Tuple[Hashable, str, int, int]) -> List[str]:
        _, path, offset, nbytes = task
        return hash_file_range(path, offset, nbytes, chunk_size, leaf)

    if workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run, tasks))
    else:
        results = [run(t) for t in tasks]

    out: Dict[Hashable, List[str]] = {key: [] for key, _, _, _ in ranges}
    for (key, _, _, _), leaves in zip(tasks, results):
        out[key].extend(leaves)  # tasks of one key are in offset order
    for key, leaves in out.items():
        if not leaves:
            leaves.append(leaf(b""))
    return out
