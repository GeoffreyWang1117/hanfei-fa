"""
Weight chunking utilities
"""

from typing import List, Iterator


DEFAULT_CHUNK_SIZE = 16384  # 16 KB


def chunk_bytes(data: bytes, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[bytes]:
    """
    Split bytes into fixed-size chunks.

    Args:
        data: Bytes to chunk
        chunk_size: Size of each chunk in bytes

    Returns:
        List of byte chunks
    """
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    return chunks


def chunk_bytes_iter(data: bytes, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[bytes]:
    """
    Iterator version of chunk_bytes for memory efficiency.

    Args:
        data: Bytes to chunk
        chunk_size: Size of each chunk in bytes

    Yields:
        Byte chunks
    """
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def chunk_tensor(tensor_bytes: bytes, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[bytes]:
    """
    Chunk serialized tensor data.

    Args:
        tensor_bytes: Serialized tensor as bytes
        chunk_size: Size of each chunk in bytes

    Returns:
        List of chunks
    """
    return chunk_bytes(tensor_bytes, chunk_size)


def estimate_chunk_count(data_size: int, chunk_size: int = DEFAULT_CHUNK_SIZE) -> int:
    """
    Estimate number of chunks for given data size.

    Args:
        data_size: Size of data in bytes
        chunk_size: Size of each chunk

    Returns:
        Estimated chunk count
    """
    return (data_size + chunk_size - 1) // chunk_size
