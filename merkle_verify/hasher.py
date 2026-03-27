"""
Hashing utilities for Merkle tree construction

Supports multiple hash algorithms: sha256 (default), sha512, sha3_256, blake2b
"""

import hashlib
from typing import Union, Callable, Optional
from enum import Enum


class HashAlgorithm(Enum):
    """Supported hash algorithms."""
    SHA256 = "sha256"
    SHA512 = "sha512"
    SHA3_256 = "sha3_256"
    BLAKE2B = "blake2b"
    BLAKE3 = "blake3"

    @property
    def hash_length(self) -> int:
        """Get expected hash length in hex characters."""
        lengths = {
            HashAlgorithm.SHA256: 64,
            HashAlgorithm.SHA512: 128,
            HashAlgorithm.SHA3_256: 64,
            HashAlgorithm.BLAKE2B: 128,
            HashAlgorithm.BLAKE3: 64,
        }
        return lengths[self]


# Default algorithm
_default_algorithm: HashAlgorithm = HashAlgorithm.SHA256


def set_default_algorithm(algorithm: Union[HashAlgorithm, str]) -> None:
    """
    Set the default hash algorithm.

    Args:
        algorithm: HashAlgorithm enum or string name

    Example:
        >>> set_default_algorithm(HashAlgorithm.SHA512)
        >>> set_default_algorithm("blake2b")
    """
    global _default_algorithm
    if isinstance(algorithm, str):
        algorithm = HashAlgorithm(algorithm.lower())
    _default_algorithm = algorithm


def get_default_algorithm() -> HashAlgorithm:
    """Get the current default hash algorithm."""
    return _default_algorithm


def _get_hasher(algorithm: HashAlgorithm) -> Callable:
    """Get the appropriate hasher function for an algorithm."""
    if algorithm == HashAlgorithm.BLAKE3:
        try:
            import blake3 as _blake3
        except ImportError:
            raise ImportError(
                "blake3 package required for BLAKE3 hashing. "
                "Install with: pip install blake3"
            )
        return _blake3.blake3

    hashers = {
        HashAlgorithm.SHA256: hashlib.sha256,
        HashAlgorithm.SHA512: hashlib.sha512,
        HashAlgorithm.SHA3_256: hashlib.sha3_256,
        HashAlgorithm.BLAKE2B: lambda: hashlib.blake2b(digest_size=64),
    }
    return hashers[algorithm]


def compute_hash(
    data: Union[bytes, str],
    algorithm: Optional[Union[HashAlgorithm, str]] = None
) -> str:
    """
    Compute hash of data using specified algorithm.

    Args:
        data: Bytes or string to hash
        algorithm: Hash algorithm (default: current default algorithm)

    Returns:
        Hex string of hash

    Example:
        >>> compute_hash(b"hello")  # Uses default (sha256)
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
        >>> compute_hash(b"hello", HashAlgorithm.SHA512)
        '9b71d224bd62f3785d96d46ad3ea3d73...'
    """
    if isinstance(data, str):
        data = data.encode('utf-8')

    if algorithm is None:
        algorithm = _default_algorithm
    elif isinstance(algorithm, str):
        algorithm = HashAlgorithm(algorithm.lower())

    hasher = _get_hasher(algorithm)()
    hasher.update(data)
    return hasher.hexdigest()


def hash_chunk(
    chunk: bytes,
    algorithm: Optional[Union[HashAlgorithm, str]] = None
) -> str:
    """
    Hash a chunk of weight data.

    Args:
        chunk: Bytes to hash
        algorithm: Hash algorithm (default: current default)

    Returns:
        Hex string of hash
    """
    return compute_hash(chunk, algorithm)


def hash_pair(
    left: str,
    right: str,
    algorithm: Optional[Union[HashAlgorithm, str]] = None
) -> str:
    """
    Hash a pair of hashes for Merkle tree internal nodes.

    Args:
        left: Left child hash
        right: Right child hash
        algorithm: Hash algorithm (default: current default)

    Returns:
        Combined hash
    """
    combined = left + right
    return compute_hash(combined, algorithm)


def verify_hash(
    data: bytes,
    expected_hash: str,
    algorithm: Optional[Union[HashAlgorithm, str]] = None
) -> bool:
    """
    Verify that data matches expected hash.

    Args:
        data: Data to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm (default: current default)

    Returns:
        True if hash matches
    """
    actual_hash = compute_hash(data, algorithm)
    return actual_hash == expected_hash


def detect_algorithm(hash_str: str) -> Optional[HashAlgorithm]:
    """
    Detect hash algorithm from hash string length.

    Note: BLAKE3 and SHA256/SHA3-256 all produce 64-char hex digests,
    so this function returns SHA256 for 64-char hashes (the default).
    Use explicit algorithm specification when using BLAKE3.

    Args:
        hash_str: Hash string to analyze

    Returns:
        Detected algorithm or None if unknown
    """
    length = len(hash_str)
    # BLAKE3 has same length as SHA256/SHA3-256 (64 hex chars).
    # We prefer SHA256 for ambiguous lengths since it's the default.
    for algo in (HashAlgorithm.SHA256, HashAlgorithm.SHA512,
                 HashAlgorithm.SHA3_256, HashAlgorithm.BLAKE2B):
        if algo.hash_length == length:
            return algo
    return None
