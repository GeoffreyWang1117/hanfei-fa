"""Tests for merkle_verify.hasher module."""

import pytest

from merkle_verify.hasher import (
    compute_hash,
    hash_chunk,
    hash_pair,
    verify_hash,
    detect_algorithm,
    HashAlgorithm,
    set_default_algorithm,
    get_default_algorithm,
)


class TestComputeHash:
    def test_bytes_input(self):
        h = compute_hash(b"hello")
        assert isinstance(h, str)
        assert len(h) == 64  # sha256

    def test_string_input(self):
        h = compute_hash("hello")
        assert h == compute_hash(b"hello")

    def test_empty_input(self):
        h = compute_hash(b"")
        assert isinstance(h, str) and len(h) == 64

    def test_deterministic(self):
        assert compute_hash(b"abc") == compute_hash(b"abc")

    def test_different_data_different_hash(self):
        assert compute_hash(b"a") != compute_hash(b"b")

    @pytest.mark.parametrize("algo,expected_len", [
        (HashAlgorithm.SHA256, 64),
        (HashAlgorithm.SHA512, 128),
        (HashAlgorithm.SHA3_256, 64),
        (HashAlgorithm.BLAKE2B, 128),
    ])
    def test_algorithms(self, algo, expected_len):
        h = compute_hash(b"test", algorithm=algo)
        assert len(h) == expected_len

    def test_algorithm_by_string(self):
        h1 = compute_hash(b"test", algorithm="sha512")
        h2 = compute_hash(b"test", algorithm=HashAlgorithm.SHA512)
        assert h1 == h2


class TestHashPair:
    def test_deterministic(self):
        a = compute_hash(b"left")
        b = compute_hash(b"right")
        assert hash_pair(a, b) == hash_pair(a, b)

    def test_order_matters(self):
        a = compute_hash(b"left")
        b = compute_hash(b"right")
        assert hash_pair(a, b) != hash_pair(b, a)


class TestVerifyHash:
    def test_correct(self):
        data = b"payload"
        h = compute_hash(data)
        assert verify_hash(data, h) is True

    def test_incorrect(self):
        assert verify_hash(b"payload", "0" * 64) is False


class TestHashChunk:
    def test_same_as_compute_hash(self):
        chunk = b"some chunk data"
        assert hash_chunk(chunk) == compute_hash(chunk)


class TestDetectAlgorithm:
    def test_sha256(self):
        h = compute_hash(b"x", HashAlgorithm.SHA256)
        assert detect_algorithm(h) == HashAlgorithm.SHA256

    def test_sha512(self):
        h = compute_hash(b"x", HashAlgorithm.SHA512)
        assert detect_algorithm(h) == HashAlgorithm.SHA512

    def test_unknown(self):
        assert detect_algorithm("abc") is None


class TestDefaultAlgorithm:
    def test_default_is_sha256(self):
        assert get_default_algorithm() == HashAlgorithm.SHA256

    def test_set_and_get(self):
        set_default_algorithm(HashAlgorithm.BLAKE2B)
        assert get_default_algorithm() == HashAlgorithm.BLAKE2B

    def test_set_by_string(self):
        set_default_algorithm("sha3_256")
        assert get_default_algorithm() == HashAlgorithm.SHA3_256

    def test_hash_uses_new_default(self):
        set_default_algorithm(HashAlgorithm.SHA512)
        h = compute_hash(b"test")
        assert len(h) == 128
