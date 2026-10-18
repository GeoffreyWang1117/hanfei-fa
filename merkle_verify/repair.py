"""
Chunk-level repair of damaged weight files.

Given a trusted manifest and a source that holds a good copy, ``repair``:

1. finds the chunks of the local copy that do not hash to the manifest's leaves,
2. fetches only those byte ranges from the source (adjacent chunks are merged
   into one request),
3. checks every fetched chunk against the manifest *before* writing it,
4. re-hashes the repaired copy and confirms it reproduces the manifest's root.

The source does not have to be trusted; only the manifest (or the
``expected_root`` it is checked against) does. A source can be a local file or
directory, an HTTP(S) URL that supports Range requests, or ``hf://org/name@rev``
for a HuggingFace Hub repository.

Manifests with per-chunk hashes allow chunk-level repair. With a compact
manifest a damaged tensor is re-fetched whole and checked against its chunk
root before it is written.
"""

import json
import os
import shutil
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from merkle_verify import format as fmt
from merkle_verify._safetensors_io import TensorInfo, read_header
from merkle_verify.format import FORMAT_V1, parse_root
from merkle_verify.hasher import compute_hash
from merkle_verify.merkle_tree import MerkleTree, ModelMerkleTree, build_file_merkle_tree

DEFAULT_MAX_REQUEST_BYTES = 8 * 1024 * 1024
DEFAULT_CONCURRENCY = 8
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


class RepairError(Exception):
    """The copy could not be repaired from this source."""


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class FileSource:
    """A good copy on a local or mounted filesystem."""

    def __init__(self, path: str):
        self.path = path
        self.name = path
        self.requests = 0
        self.bytes_read = 0
        self._lock = threading.Lock()

    def read_range(self, offset: int, length: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(offset)
            data = f.read(length)
        with self._lock:
            self.requests += 1
            self.bytes_read += len(data)
        if len(data) != length:
            raise RepairError(f"{self.path}: short read at offset {offset}")
        return data


class HTTPSource:
    """A good copy behind an HTTP(S) URL that honours Range requests.

    After the first response, requests go straight to the redirect target (for
    the HF Hub, a signed CDN URL); if that stops working they go back through
    the original URL once.
    """

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 60.0):
        self.url = url
        self.name = url
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.requests = 0
        self.bytes_read = 0
        self._resolved: Optional[str] = None
        self._lock = threading.Lock()

    def _get(self, url: str, offset: int, length: int):
        req = urllib.request.Request(url, headers={
            **self.headers, "Range": f"bytes={offset}-{offset + length - 1}",
        })
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            if resp.status != 206:
                raise RepairError(
                    f"{self.url}: server answered {resp.status} to a Range request; "
                    "refusing to download the whole file"
                )
            return resp.read(length + 1), resp.geturl()

    def read_range(self, offset: int, length: int) -> bytes:
        if length == 0:
            return b""
        span = f"bytes {offset}-{offset + length - 1}"
        resolved = self._resolved
        try:
            try:
                data, final = self._get(resolved or self.url, offset, length)
            except urllib.error.HTTPError:
                if resolved is None:
                    raise
                self._resolved = None  # e.g. an expired signed URL: resolve again
                data, final = self._get(self.url, offset, length)
        except urllib.error.HTTPError as e:
            raise RepairError(f"{self.url}: HTTP {e.code} for {span}") from None
        except urllib.error.URLError as e:
            raise RepairError(f"{self.url}: {e.reason}") from None
        with self._lock:
            self.requests += 1
            self.bytes_read += len(data)
            if final != self.url and final.startswith("https://"):
                self._resolved = final
        if len(data) != length:
            raise RepairError(f"{self.url}: expected {length} bytes at offset {offset}, got {len(data)}")
        return data


def hf_url(repo_id: str, filename: str, revision: str = "main") -> str:
    quoted = urllib.parse.quote(filename)
    return f"{HF_ENDPOINT}/{repo_id}/resolve/{urllib.parse.quote(revision, safe='')}/{quoted}"


def _hf_headers() -> Dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def open_source(spec: str, filename: Optional[str] = None):
    """Turn a source specification into a source object.

    ``spec`` is a file path, a directory, an ``http(s)://`` URL, or
    ``hf://org/name[@revision]``. For a directory, a URL ending in ``/`` or an
    ``hf://`` repository, ``filename`` names the file inside it.
    """
    if spec.startswith("hf://"):
        repo, _, revision = spec[len("hf://"):].partition("@")
        if not filename:
            raise RepairError("an hf:// source needs a file name")
        return HTTPSource(hf_url(repo.strip("/"), filename, revision or "main"), _hf_headers())
    if spec.startswith(("http://", "https://")):
        url = spec + urllib.parse.quote(filename) if (spec.endswith("/") and filename) else spec
        return HTTPSource(url)
    if os.path.isdir(spec):
        if not filename:
            raise RepairError(f"{spec} is a directory; a file name is needed")
        return FileSource(os.path.join(spec, filename))
    return FileSource(spec)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class RepairReport:
    path: str
    source: str
    dry_run: bool = False
    checked_chunks: int = 0
    bad_chunks: List[Tuple[str, int]] = field(default_factory=list)
    header_repaired: bool = False
    fetched_bytes: int = 0
    requests: int = 0
    total_bytes: int = 0
    verified: bool = False
    root: Optional[str] = None

    @property
    def damaged(self) -> bool:
        return bool(self.bad_chunks) or self.header_repaired

    def summary(self) -> str:
        what = "would fetch" if self.dry_run else "fetched"
        pct = 100.0 * self.fetched_bytes / self.total_bytes if self.total_bytes else 0.0
        return (f"{len(self.bad_chunks)}/{self.checked_chunks} chunks damaged"
                f"{', header damaged' if self.header_repaired else ''}; "
                f"{what} {self.fetched_bytes} bytes ({pct:.2f}% of {self.total_bytes}) "
                f"in {self.requests} request(s)")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

@dataclass
class _Unit:
    """One byte range of the file whose content the manifest fixes."""
    label: Tuple[str, int]
    offset: int
    length: int
    check: Callable[[bytes], bool]


def _leaf_fn(tree: MerkleTree) -> Callable[[bytes], str]:
    alg = tree.algorithm
    if tree.version == FORMAT_V1:
        return lambda chunk: compute_hash(chunk, alg)
    return lambda chunk: fmt.hash_leaf(chunk, alg)


def _chunk_units(name: str, offset: int, nbytes: int, tree: MerkleTree) -> List[_Unit]:
    """Per-chunk units (manifest with chunk hashes) or one whole-tensor unit (compact)."""
    leaf = _leaf_fn(tree)
    cs = tree.chunk_size
    if tree.chunk_hashes is not None:
        units = []
        for i, expected in enumerate(tree.chunk_hashes):
            length = max(0, min(cs, nbytes - i * cs))
            units.append(_Unit((name, i), offset + i * cs, length,
                               lambda data, e=expected: leaf(data) == e))
        return units

    def whole(data: bytes, t=tree) -> bool:
        if len(data) != nbytes:
            return False
        rebuilt = MerkleTree.from_bytes(data, cs, version=t.version, algorithm=t.algorithm)
        return rebuilt.root_hash == t.root_hash
    return [_Unit((name, -1), offset, nbytes, whole)]


def _layout_matches(tensors: Dict[str, TensorInfo], trees: Dict[str, MerkleTree]) -> bool:
    if set(tensors) != set(trees):
        return False
    for name, info in tensors.items():
        tree = trees[name]
        if tree.num_bytes is not None and tree.num_bytes != info.nbytes:
            return False
        if max(1, -(-info.nbytes // tree.chunk_size)) != tree.num_chunks:
            return False
        if tree.dtype is not None and (tree.dtype, tree.shape) != (info.dtype, info.shape):
            return False
    return True


def _source_header(source) -> Tuple[bytes, Dict[str, TensorInfo]]:
    """Fetch and parse the source's safetensors header."""
    prefix = source.read_range(0, 8)
    (n,) = struct.unpack("<Q", prefix)
    if n > 100 * 1024 * 1024:
        raise RepairError(f"{source.name}: implausible header length {n}")
    raw = prefix + source.read_range(8, n)
    header = json.loads(raw[8:].decode("utf-8"))
    header.pop("__metadata__", None)
    tensors = {}
    for name, e in header.items():
        start, end = e["data_offsets"]
        tensors[name] = TensorInfo(name, e["dtype"], tuple(e["shape"]), 8 + n + start, end - start)
    return raw, tensors


def _read_local(f, offset: int, length: int) -> bytes:
    f.seek(offset)
    return f.read(length)


def _repair_units(path: str, units: List[_Unit], source, report: RepairReport,
                  dry_run: bool, max_request: int, concurrency: int = DEFAULT_CONCURRENCY) -> None:
    """Find damaged units in ``path``, fetch them, verify, write."""
    bad: List[_Unit] = []
    with open(path, "rb") as f:
        for u in units:
            report.checked_chunks += 1
            if not u.check(_read_local(f, u.offset, u.length)):
                bad.append(u)
    report.bad_chunks.extend(u.label for u in bad)
    if not bad:
        return

    # merge byte-adjacent units into requests of at most max_request bytes
    groups: List[List[_Unit]] = []
    for u in sorted(bad, key=lambda u: u.offset):
        g = groups[-1] if groups else None
        if g and g[-1].offset + g[-1].length == u.offset and \
                u.offset + u.length - g[0].offset <= max_request:
            g.append(u)
        else:
            groups.append([u])

    for g in groups:
        report.fetched_bytes += g[-1].offset + g[-1].length - g[0].offset
        report.requests += 1
    if dry_run:
        return

    def fetch(g: List[_Unit]) -> Tuple[List[_Unit], bytes]:
        start = g[0].offset
        data = source.read_range(start, g[-1].offset + g[-1].length - start)
        for u in g:
            if not u.check(data[u.offset - start:u.offset - start + u.length]):
                raise RepairError(
                    f"{source.name}: data for {u.label[0]} chunk {u.label[1]} does not match "
                    "the manifest; the source is damaged or a different model"
                )
        return g, data

    # Requests run concurrently (they are latency-bound); a group is written only
    # after every chunk in it has been checked against the manifest.
    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(groups)))) as pool, \
            open(path, "r+b") as out:
        for g, data in pool.map(fetch, groups):
            start = g[0].offset
            for u in g:
                out.seek(u.offset)
                out.write(data[u.offset - start:u.offset - start + u.length])


def _repair_safetensors_file(path: str, trees: Dict[str, MerkleTree], source,
                             report: RepairReport, dry_run: bool, max_request: int) -> None:
    try:
        local, _ = read_header(path)
        layout_ok = _layout_matches(local, trees)
    except (ValueError, OSError, struct.error):
        local, layout_ok = {}, False

    if not layout_ok:
        raw_header, remote = _source_header(source)
        if not _layout_matches(remote, trees):
            raise RepairError(f"{source.name}: header does not describe the tensors in the manifest")
        report.header_repaired = True
        report.fetched_bytes += len(raw_header)
        report.requests += 2
        local = remote
        if not dry_run:
            if not os.path.exists(path):
                open(path, "wb").close()
            with open(path, "r+b") as out:
                out.seek(0)
                out.write(raw_header)

    # safetensors buffers have no holes, so the last tensor ends the file
    expected_size = max(i.offset + i.nbytes for i in local.values())
    report.total_bytes += expected_size
    units: List[_Unit] = []
    for name, info in local.items():
        units.extend(_chunk_units(name, info.offset, info.nbytes, trees[name]))
    _repair_units(path, units, source, report, dry_run, max_request)
    if not dry_run and os.path.getsize(path) != expected_size:
        with open(path, "r+b") as out:
            out.truncate(expected_size)


def _load_manifest(manifest) -> Union[ModelMerkleTree, MerkleTree]:
    if isinstance(manifest, (ModelMerkleTree, MerkleTree)):
        return manifest
    with open(manifest) as f:
        data = json.load(f)
    if "layer_trees" in data or "tensors" in data:
        return ModelMerkleTree.from_dict(data)
    return MerkleTree.from_dict(data)


def _check_trusted(tree, expected_root: Optional[str]) -> None:
    root = tree.model_root if isinstance(tree, ModelMerkleTree) else tree.root_hash
    if isinstance(tree, ModelMerkleTree) and tree.format_version == FORMAT_V1 and not tree.verify():
        raise RepairError("the manifest is internally inconsistent")
    if expected_root is not None and root != parse_root(expected_root)[1]:
        raise RepairError("the manifest's root is not the expected root; refusing to use it")


def repair(
    path: str,
    source: str,
    manifest=None,
    *,
    expected_root: Optional[str] = None,
    output: Optional[str] = None,
    dry_run: bool = False,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> RepairReport:
    """
    Repair a damaged safetensors file, sharded model directory, or plain file.

    Args:
        path: The damaged copy (``.safetensors`` file, model directory, or any file
            signed with ``merkle-verify sign``)
        source: Where a good copy lives: path, directory, ``http(s)://`` URL, or
            ``hf://org/name[@revision]`` (file names are appended for directories,
            URLs ending in ``/`` and hf:// repositories)
        manifest: Trusted manifest path or tree (default: the sidecar next to
            ``path``, or ``<dir>/model.merkle.json``)
        expected_root: If given, the manifest must have this root
        output: Write the repaired copy here instead of in place (files only)
        dry_run: Only report what would be fetched
        max_request_bytes: Upper bound for one merged range request

    Returns:
        RepairReport. ``verified`` is True when the repaired copy reproduces the
        manifest root (always False in a dry run).

    Raises:
        RepairError: the manifest is not trusted, or the source cannot supply
            data matching it.
    """
    p = Path(path)
    is_dir = p.is_dir()
    if manifest is None:
        manifest = str(p / "model.merkle.json") if is_dir else str(p.with_suffix(".merkle.json"))
    tree = _load_manifest(manifest)
    _check_trusted(tree, expected_root)

    if output is not None:
        if is_dir:
            raise RepairError("output= is only supported for single files")
        if not dry_run:
            shutil.copyfile(path, output)
            path = output

    if is_dir:
        return _repair_directory(path, tree, source, dry_run, max_request_bytes)

    if isinstance(tree, MerkleTree):
        src = open_source(source, os.path.basename(path))
        report = RepairReport(path=path, source=src.name, dry_run=dry_run)
        nbytes = tree.num_bytes
        if nbytes is None:  # v1 manifests do not record the size; trust a local source's
            if not isinstance(src, FileSource):
                raise RepairError("a v1 single-file manifest does not record the file size; "
                                  "use a local source or re-sign with format v2")
            nbytes = os.path.getsize(src.path)
        if max(1, -(-nbytes // tree.chunk_size)) != tree.num_chunks:
            raise RepairError("the source's size does not match the manifest's chunk count")
        report.total_bytes = nbytes
        _repair_units(path, _chunk_units("file", 0, nbytes, tree), src, report, dry_run,
                      max_request_bytes)
        if not dry_run:
            if os.path.getsize(path) != nbytes:
                with open(path, "r+b") as out:
                    out.truncate(nbytes)
            rebuilt = build_file_merkle_tree(path, tree.chunk_size, format_version=tree.version,
                                             algorithm=tree.algorithm)
            report.root = rebuilt.root_hash
            report.verified = rebuilt.root_hash == tree.root_hash
        return report

    trees = dict(tree.iter_params())
    src = open_source(source, os.path.basename(path))
    report = RepairReport(path=path, source=src.name, dry_run=dry_run)
    _repair_safetensors_file(path, trees, src, report, dry_run, max_request_bytes)
    if not dry_run:
        _final_check(report, tree, lambda kw: _build_single(path, kw))
    return report


def _build_single(path: str, kw) -> ModelMerkleTree:
    from merkle_verify.safetensors_adapter import build_tree
    return build_tree(path, **kw)


def _final_check(report: RepairReport, tree: ModelMerkleTree, build) -> None:
    chunk_size = next(t.chunk_size for _, t in tree.iter_params())
    rebuilt = build(dict(chunk_size=chunk_size, format_version=tree.format_version,
                         algorithm=tree.algorithm))
    report.root = rebuilt.model_root
    report.verified = rebuilt.model_root == tree.model_root


def _repair_directory(directory: str, tree: ModelMerkleTree, source: str,
                      dry_run: bool, max_request: int) -> RepairReport:
    from merkle_verify.safetensors_adapter import (
        INDEX_FILENAME, SINGLE_FILENAME, build_sharded_tree,
    )
    d = Path(directory)
    trees = dict(tree.iter_params())
    index_path = d / INDEX_FILENAME
    if index_path.exists():
        with open(index_path) as f:
            weight_map: Dict[str, str] = json.load(f).get("weight_map", {})
        if set(weight_map) != set(trees):
            raise RepairError(f"{index_path}: weight_map does not list the tensors in the manifest")
    else:
        weight_map = {name: SINGLE_FILENAME for name in trees}

    report = RepairReport(path=directory, source=source, dry_run=dry_run)
    for shard in sorted(set(weight_map.values())):
        shard_trees = {n: trees[n] for n, s in weight_map.items() if s == shard}
        _repair_safetensors_file(str(d / shard), shard_trees, open_source(source, shard),
                                 report, dry_run, max_request)
    if not dry_run:
        _final_check(report, tree, lambda kw: build_sharded_tree(directory, **kw))
    return report
