"""
Merkle trees for model weights.

Two hashing formats are supported (see docs/MANIFEST_V2.md):

* **v2** (default): domain-separated RFC 6962 trees; tensor roots bind name,
  dtype, shape, byte length and chunk size; the model root is a Merkle tree
  over tensor roots, so single tensors and chunks have inclusion proofs.
* **v1**: the 0.2.x construction, kept bit-for-bit so existing manifests
  keep verifying.

Every tree remembers the hash algorithm it was built with and uses it for all
later hashing, so changing the process-wide default afterwards is harmless.
"""

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os

from merkle_verify import format as fmt
from merkle_verify.format import (
    DEFAULT_FORMAT_VERSION,
    FORMAT_NAME,
    FORMAT_V1,
    FORMAT_V2,
    check_format_version,
    name_sort_key,
)
from merkle_verify.hasher import HashAlgorithm, compute_hash, hash_pair, resolve_algorithm
from merkle_verify.chunking import DEFAULT_CHUNK_SIZE
from merkle_verify._tensor_bytes import to_bytes_and_meta
from merkle_verify._chunk_hashing import hash_buffer, hash_file_ranges, leaf_hasher

# Default number of workers for parallel processing
DEFAULT_NUM_WORKERS = min(4, os.cpu_count() or 1)

AlgorithmLike = Optional[Union[HashAlgorithm, str]]


def _created_by() -> str:
    from merkle_verify import __version__
    return f"merkle-weight-verify {__version__}"


@dataclass
class MerkleNode:
    """Node in a Merkle tree"""
    hash_value: str
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    is_leaf: bool = False
    chunk_index: Optional[int] = None  # For leaf nodes
    chunk_start: int = 0   # First chunk index covered by this subtree
    chunk_count: int = 1   # Number of unique chunks covered by this subtree


@dataclass
class MerkleProof:
    """Inclusion proof for one chunk of a MerkleTree.

    v1 proofs are folded with the stored left/right positions. v2 proofs are
    checked with the RFC 9162 algorithm from ``chunk_index`` and
    ``num_chunks``; the positions are informational only.
    """
    chunk_index: int
    chunk_hash: str
    sibling_hashes: List[Tuple[str, str]]  # (hash, position: 'left'/'right')
    root_hash: str
    version: int = FORMAT_V1
    algorithm: Optional[str] = None
    num_chunks: Optional[int] = None

    def verify(self) -> bool:
        """Verify this proof leads to root hash"""
        if self.version == FORMAT_V1:
            current = self.chunk_hash
            for sibling_hash, position in self.sibling_hashes:
                if position == 'left':
                    current = hash_pair(sibling_hash, current, self.algorithm)
                else:
                    current = hash_pair(current, sibling_hash, self.algorithm)
            return current == self.root_hash
        if self.num_chunks is None:
            return False
        root = fmt.root_from_inclusion_path(
            self.chunk_hash, self.chunk_index, self.num_chunks,
            [h for h, _ in self.sibling_hashes], self.algorithm,
        )
        return root is not None and root == self.root_hash

    def verify_chunk(self, chunk: bytes) -> bool:
        """Check that ``chunk`` is the leaf this proof is about and that the proof holds."""
        if self.version == FORMAT_V1:
            leaf = compute_hash(chunk, self.algorithm)
        else:
            leaf = fmt.hash_leaf(chunk, self.algorithm)
        return leaf == self.chunk_hash and self.verify()


class MerkleTree:
    """
    Merkle tree over the chunks of one blob (a tensor or a file).

    Attributes:
        chunk_hashes: List of leaf hashes (``None`` for a compact, root-only tree)
        root_hash: Hash at root
        chunk_size: Size of each chunk in bytes
        num_chunks: Number of leaves
        version: Hash format (1 or 2)
        algorithm: HashAlgorithm used for every hash in this tree
        num_bytes, dtype, shape: Tensor metadata (bound into the v2 tensor root)
    """

    def __init__(
        self,
        chunk_hashes: List[str],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        *,
        version: int = DEFAULT_FORMAT_VERSION,
        algorithm: AlgorithmLike = None,
        num_bytes: Optional[int] = None,
        dtype: Optional[str] = None,
        shape: Optional[Sequence[int]] = None,
    ):
        """
        Build Merkle tree from chunk hashes.

        Args:
            chunk_hashes: Leaf hashes. For v2 these must be ``format.hash_leaf(chunk)``;
                prefer ``MerkleTree.from_bytes`` / ``from_file``, which hash correctly.
            chunk_size: Size of chunks in bytes
            version: Hash format, 1 or 2 (default 2)
            algorithm: Hash algorithm (default: the current process default)
            num_bytes, dtype, shape: Tensor metadata, required to compute a v2 tensor root
        """
        self.version = check_format_version(version)
        self.algorithm = resolve_algorithm(algorithm)
        self.chunk_hashes = chunk_hashes
        self.chunk_size = chunk_size
        self.num_chunks = len(chunk_hashes)
        self.num_bytes = num_bytes
        self.dtype = dtype
        self.shape = tuple(int(d) for d in shape) if shape is not None else None
        self._tensor_roots: Dict[str, str] = {}

        if self.version == FORMAT_V1:
            self._root = self._build_tree(chunk_hashes)
            self._levels: Optional[List[List[str]]] = None
            self.root_hash = self._root.hash_value
        else:
            if not chunk_hashes:
                raise ValueError("Cannot build tree from empty hash list")
            self._root = None
            self._levels = self._build_levels(chunk_hashes)
            self.root_hash = self._levels[-1][0]

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        *,
        version: int = DEFAULT_FORMAT_VERSION,
        algorithm: AlgorithmLike = None,
        dtype: Optional[str] = None,
        shape: Optional[Sequence[int]] = None,
    ) -> 'MerkleTree':
        """Chunk and hash ``data`` with the leaf rule of the chosen format."""
        alg = resolve_algorithm(algorithm)
        return cls(
            _hash_chunks(data, chunk_size, version, alg), chunk_size,
            version=version, algorithm=alg, num_bytes=len(data), dtype=dtype, shape=shape,
        )

    @classmethod
    def from_root(
        cls,
        chunk_root: str,
        num_chunks: int,
        chunk_size: int,
        *,
        algorithm: AlgorithmLike = None,
        num_bytes: Optional[int] = None,
        dtype: Optional[str] = None,
        shape: Optional[Sequence[int]] = None,
    ) -> 'MerkleTree':
        """A compact v2 tree that knows only its root (from a manifest without chunk hashes).

        It verifies and diffs at tensor granularity; chunk-level diffs report
        every chunk of a changed tensor, and proofs are unavailable.
        """
        tree = cls.__new__(cls)
        tree.version = FORMAT_V2
        tree.algorithm = resolve_algorithm(algorithm)
        tree.chunk_hashes = None
        tree.chunk_size = chunk_size
        tree.num_chunks = num_chunks
        tree.num_bytes = num_bytes
        tree.dtype = dtype
        tree.shape = tuple(int(d) for d in shape) if shape is not None else None
        tree._tensor_roots = {}
        tree._root = None
        tree._levels = None
        tree.root_hash = chunk_root
        return tree

    @property
    def has_chunk_hashes(self) -> bool:
        return self.chunk_hashes is not None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_tree(self, hashes: List[str]) -> MerkleNode:
        """Build the v1 tree (odd levels duplicate their last node)."""
        if len(hashes) == 0:
            raise ValueError("Cannot build tree from empty hash list")

        # Create leaf nodes
        if len(hashes) == 1:
            return MerkleNode(
                hash_value=hashes[0], is_leaf=True, chunk_index=0,
                chunk_start=0, chunk_count=1,
            )

        # Build leaves
        nodes = [
            MerkleNode(
                hash_value=h, is_leaf=True, chunk_index=i,
                chunk_start=i, chunk_count=1,
            )
            for i, h in enumerate(hashes)
        ]

        # Build tree bottom-up
        while len(nodes) > 1:
            next_level = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                if i + 1 < len(nodes):
                    right = nodes[i + 1]
                    parent_count = left.chunk_count + right.chunk_count
                else:
                    # Duplicate last node if odd number
                    right = nodes[i]
                    parent_count = left.chunk_count  # No new chunks

                parent_hash = hash_pair(left.hash_value, right.hash_value, self.algorithm)
                parent = MerkleNode(
                    hash_value=parent_hash,
                    left=left,
                    right=right,
                    is_leaf=False,
                    chunk_start=left.chunk_start,
                    chunk_count=parent_count,
                )
                next_level.append(parent)

            nodes = next_level

        return nodes[0]

    def _build_levels(self, leaves: List[str]) -> List[List[str]]:
        """Build the v2 tree as a list of levels (RFC 6962 shape)."""
        levels = [list(leaves)]
        while len(levels[-1]) > 1:
            cur = levels[-1]
            nxt = [
                fmt.hash_node(cur[i], cur[i + 1], self.algorithm)
                for i in range(0, len(cur) - 1, 2)
            ]
            if len(cur) % 2:
                nxt.append(cur[-1])  # carried up unchanged
            levels.append(nxt)
        return levels

    @property
    def root(self) -> MerkleNode:
        """Root node. For v2 trees the node structure is built on demand."""
        if self._root is None:
            if self._levels is None:
                raise ValueError("a compact (root-only) tree has no node structure")
            self._root = self._nodes_from_levels()
        return self._root

    def _nodes_from_levels(self) -> MerkleNode:
        nodes = [
            MerkleNode(hash_value=h, is_leaf=True, chunk_index=i, chunk_start=i, chunk_count=1)
            for i, h in enumerate(self._levels[0])
        ]
        for level in self._levels[1:]:
            nxt = []
            for j, h in enumerate(level):
                left = nodes[2 * j]
                if 2 * j + 1 < len(nodes):
                    right = nodes[2 * j + 1]
                    nxt.append(MerkleNode(
                        hash_value=h, left=left, right=right, chunk_start=left.chunk_start,
                        chunk_count=left.chunk_count + right.chunk_count,
                    ))
                else:
                    nxt.append(left)
            nodes = nxt
        return nodes[0]

    # ------------------------------------------------------------------
    # Tensor commitment (v2)
    # ------------------------------------------------------------------

    def tensor_root(self, name: str) -> str:
        """The v2 tensor root binding ``name`` and this tree's metadata.

        For v1 trees this is simply the chunk root, as in 0.2.x.
        """
        if self.version == FORMAT_V1:
            return self.root_hash
        cached = self._tensor_roots.get(name)
        if cached is None:
            if self.num_bytes is None:
                raise ValueError(
                    f"tensor {name!r}: a v2 tensor root needs num_bytes; build trees with "
                    "MerkleTree.from_bytes() or build_model_merkle_tree()"
                )
            cached = fmt.tensor_root(
                name, self.root_hash, self.num_bytes, self.chunk_size,
                self.dtype, self.shape, self.algorithm,
            )
            self._tensor_roots[name] = cached
        return cached

    # ------------------------------------------------------------------
    # Proofs
    # ------------------------------------------------------------------

    def get_proof(self, chunk_index: int) -> MerkleProof:
        """
        Generate Merkle proof for a chunk.

        Args:
            chunk_index: Index of chunk

        Returns:
            MerkleProof object
        """
        if not 0 <= chunk_index < self.num_chunks:
            raise ValueError(f"Chunk index {chunk_index} out of range")
        if self.version == FORMAT_V2:
            return self._get_proof_v2(chunk_index)

        chunk_hash = self.chunk_hashes[chunk_index]
        sibling_hashes = []

        # Traverse from leaf to root, collecting sibling hashes.
        # When a level has an odd number of nodes, _build_tree duplicates
        # the last node (right = left). The proof must include this
        # self-sibling so that the verifier can reconstruct the parent hash.
        current_idx = chunk_index
        level_size = self.num_chunks

        while level_size > 1:
            level_hashes = self._get_level_hashes(level_size)

            if current_idx % 2 == 0:
                if current_idx + 1 < level_size:
                    # Normal case: right sibling exists
                    sibling_hash = level_hashes[current_idx + 1]
                else:
                    # Odd level: last node is duplicated as its own sibling
                    sibling_hash = level_hashes[current_idx]
                sibling_hashes.append((sibling_hash, 'right'))
            else:
                sibling_hash = level_hashes[current_idx - 1]
                sibling_hashes.append((sibling_hash, 'left'))

            # Move to parent level
            current_idx = current_idx // 2
            level_size = (level_size + 1) // 2

        return MerkleProof(
            chunk_index=chunk_index,
            chunk_hash=chunk_hash,
            sibling_hashes=sibling_hashes,
            root_hash=self.root_hash,
            version=FORMAT_V1,
            algorithm=self.algorithm.value,
            num_chunks=self.num_chunks,
        )

    def _get_proof_v2(self, chunk_index: int) -> MerkleProof:
        if self._levels is None:
            raise ValueError("a compact (root-only) tree cannot produce chunk proofs")
        siblings: List[Tuple[str, str]] = []
        i = chunk_index
        for level in self._levels[:-1]:
            if i % 2 == 1:
                siblings.append((level[i - 1], 'left'))
            elif i + 1 < len(level):
                siblings.append((level[i + 1], 'right'))
            i //= 2
        return MerkleProof(
            chunk_index=chunk_index,
            chunk_hash=self._levels[0][chunk_index],
            sibling_hashes=siblings,
            root_hash=self.root_hash,
            version=FORMAT_V2,
            algorithm=self.algorithm.value,
            num_chunks=self.num_chunks,
        )

    def _get_level_hashes(self, level_size: int) -> List[str]:
        """Get all hashes at a level of given size (v1)"""
        if level_size == self.num_chunks:
            return self.chunk_hashes

        # Build up from leaves
        current_level = self.chunk_hashes[:]
        while len(current_level) > level_size:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else current_level[i]
                next_level.append(hash_pair(left, right, self.algorithm))
            current_level = next_level

        return current_level

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def _check_comparable(self, other: 'MerkleTree') -> None:
        if self.version != other.version:
            raise ValueError(
                f"cannot diff a format v{self.version} tree against a v{other.version} tree"
            )
        if self.version == FORMAT_V2:
            if self.algorithm != other.algorithm:
                raise ValueError(
                    f"cannot diff trees hashed with {self.algorithm.value} and {other.algorithm.value}"
                )
            if self.chunk_size != other.chunk_size:
                raise ValueError(
                    f"cannot diff trees with chunk sizes {self.chunk_size} and {other.chunk_size}"
                )

    def diff_tree(self, other: 'MerkleTree') -> Tuple[List[int], int]:
        """
        Find changed chunk indices by walking both trees in parallel.

        Only descends into subtrees where hashes differ, achieving
        O(k * log C) complexity where k is the number of changed chunks
        and C is the total number of chunks.

        Args:
            other: Another MerkleTree to compare against

        Returns:
            Tuple of (changed_chunk_indices, hash_comparisons_count)

        Raises:
            ValueError: if the trees use different formats, or (v2) different
                algorithms or chunk sizes, so chunk indices are not comparable.
        """
        self._check_comparable(other)
        if self.version == FORMAT_V2:
            return self._diff_v2(other)

        comparisons = [0]  # Use list for mutability in nested function
        changed: List[int] = []

        # Walk the trees using node-embedded chunk ranges
        # (the root comparison happens inside _diff_walk)
        self._diff_walk(self.root, other.root, changed, comparisons)

        if not changed and self.num_chunks == other.num_chunks:
            return [], comparisons[0]

        # Handle chunks that exist in only one tree (different sizes)
        min_chunks = min(self.num_chunks, other.num_chunks)
        max_chunks = max(self.num_chunks, other.num_chunks)
        if min_chunks != max_chunks:
            changed_set = set(changed)
            for i in range(min_chunks, max_chunks):
                if i not in changed_set:
                    changed.append(i)

        changed.sort()
        return changed, comparisons[0]

    def _diff_v2(self, other: 'MerkleTree') -> Tuple[List[int], int]:
        if self.root_hash == other.root_hash and self.num_chunks == other.num_chunks:
            return [], 1
        if self._levels is None or other._levels is None:
            # compact trees: only the root is known
            return list(range(max(self.num_chunks, other.num_chunks))), 1

        if self.num_chunks != other.num_chunks:
            # Different shapes: compare the common prefix leaf by leaf.
            a, b = self._levels[0], other._levels[0]
            common = min(len(a), len(b))
            changed = [i for i in range(common) if a[i] != b[i]]
            changed.extend(range(common, max(len(a), len(b))))
            return changed, common

        # Same number of leaves => identical shapes; walk from the root.
        levels_a, levels_b = self._levels, other._levels
        changed: List[int] = []
        comparisons = 0
        stack = [(len(levels_a) - 1, 0)]
        while stack:
            k, j = stack.pop()
            comparisons += 1
            if levels_a[k][j] == levels_b[k][j]:
                continue
            # descend through carried-up nodes, which repeat their only child
            while k > 0 and 2 * j + 1 >= len(levels_a[k - 1]):
                k, j = k - 1, 2 * j
            if k == 0:
                changed.append(j)
                continue
            stack.append((k - 1, 2 * j + 1))
            stack.append((k - 1, 2 * j))
        changed.sort()
        return changed, comparisons

    def _diff_walk(
        self,
        node_a: Optional['MerkleNode'],
        node_b: Optional['MerkleNode'],
        changed: List[int],
        comparisons: List[int],
    ) -> None:
        """
        Recursively walk both v1 trees, only descending where hashes differ.

        Uses chunk_start and chunk_count metadata stored on each node
        to correctly identify which chunk indices are affected.
        """
        if node_a is None and node_b is None:
            return

        if node_a is None or node_b is None:
            # One subtree is missing entirely - all chunks in range changed
            existing = node_a if node_a is not None else node_b
            for i in range(existing.chunk_start,
                           existing.chunk_start + existing.chunk_count):
                changed.append(i)
            return

        comparisons[0] += 1
        if node_a.hash_value == node_b.hash_value:
            return  # Subtree identical, prune

        # Both are leaves - this chunk changed
        if node_a.is_leaf and node_b.is_leaf:
            changed.append(node_a.chunk_start)
            return

        # One is a leaf but the other isn't (different tree depths) -
        # mark all chunks in the larger range as changed
        if node_a.is_leaf or node_b.is_leaf:
            wider = node_a if node_a.chunk_count >= node_b.chunk_count else node_b
            for i in range(wider.chunk_start,
                           wider.chunk_start + wider.chunk_count):
                changed.append(i)
            return

        # Recurse into left children
        self._diff_walk(node_a.left, node_b.left, changed, comparisons)

        # Recurse into right children, but detect duplication from odd counts.
        # When a level has an odd number of nodes, _build_tree sets right = left
        # (same object). Those duplicated right children cover no new chunks.
        right_a = node_a.right
        right_b = node_b.right
        a_right_is_dup = (right_a is node_a.left)
        b_right_is_dup = (right_b is node_b.left)

        if a_right_is_dup and b_right_is_dup:
            # Both sides duplicated their right child - no new chunks
            return
        elif a_right_is_dup:
            # tree_a has no extra right chunks, but tree_b does
            for i in range(right_b.chunk_start,
                           right_b.chunk_start + right_b.chunk_count):
                changed.append(i)
        elif b_right_is_dup:
            # tree_b has no extra right chunks, but tree_a does
            for i in range(right_a.chunk_start,
                           right_a.chunk_start + right_a.chunk_count):
                changed.append(i)
        else:
            self._diff_walk(right_a, right_b, changed, comparisons)

    # ------------------------------------------------------------------
    # Serialisation (single-blob manifests)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        if self.version == FORMAT_V1:
            return {
                'root_hash': self.root_hash,
                'num_chunks': self.num_chunks,
                'chunk_size': self.chunk_size,
                'chunk_hashes': self.chunk_hashes,
            }
        out: Dict[str, Any] = {
            'format': FORMAT_NAME,
            'format_version': FORMAT_V2,
            'algorithm': self.algorithm.value,
            'root_hash': self.root_hash,
            'num_bytes': self.num_bytes,
            'num_chunks': self.num_chunks,
            'chunk_size': self.chunk_size,
            'dtype': self.dtype,
            'shape': list(self.shape) if self.shape is not None else None,
        }
        if self.chunk_hashes is not None:
            out['chunk_hashes'] = self.chunk_hashes
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any], algorithm: AlgorithmLike = None) -> 'MerkleTree':
        """Deserialize from dictionary.

        v1 dicts carry no algorithm; ``algorithm`` (default: process default) is used.
        v2 dicts are checked for internal consistency.
        """
        version = data.get('format_version', FORMAT_V1)
        if check_format_version(version) == FORMAT_V1:
            return cls(chunk_hashes=data['chunk_hashes'], chunk_size=data['chunk_size'],
                       version=FORMAT_V1, algorithm=algorithm)
        alg = resolve_algorithm(data['algorithm'])
        if 'chunk_hashes' in data:
            tree = cls(data['chunk_hashes'], data['chunk_size'], version=FORMAT_V2, algorithm=alg,
                       num_bytes=data.get('num_bytes'), dtype=data.get('dtype'),
                       shape=data.get('shape'))
            if tree.root_hash != data['root_hash']:
                raise ValueError("manifest is inconsistent: root_hash does not match chunk_hashes")
        else:
            tree = cls.from_root(data['root_hash'], data['num_chunks'], data['chunk_size'],
                                 algorithm=alg, num_bytes=data.get('num_bytes'),
                                 dtype=data.get('dtype'), shape=data.get('shape'))
        return tree

    @classmethod
    def from_file(
        cls,
        file_handle,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        *,
        version: int = DEFAULT_FORMAT_VERSION,
        algorithm: AlgorithmLike = None,
    ) -> 'MerkleTree':
        """
        Build a Merkle tree by streaming from a file handle.

        Memory usage is O(chunk_size + num_chunks * hash_length),
        NOT O(file_size). Suitable for multi-GB files.

        Args:
            file_handle: Readable binary file object (supports .read(n))
            chunk_size: Size of each chunk in bytes
            version: Hash format (default 2). An empty file is one empty
                chunk in v2 and an error in v1.
            algorithm: Hash algorithm (default: process default)

        Returns:
            MerkleTree

        Example:
            >>> with open("model.bin", "rb") as f:
            ...     tree = MerkleTree.from_file(f)
        """
        version = check_format_version(version)
        alg = resolve_algorithm(algorithm)
        leaf = leaf_hasher(version, alg)
        chunk_hashes: List[str] = []
        total = 0
        # read whole chunks at a time; a short read is only accepted at EOF
        buf = bytearray(chunk_size)
        mv = memoryview(buf)
        while True:
            got = 0
            while got < chunk_size:
                n = file_handle.readinto(mv[got:]) if hasattr(file_handle, "readinto") else None
                if n is None:  # file-like without readinto
                    piece = file_handle.read(chunk_size - got)
                    n = len(piece)
                    mv[got:got + n] = piece
                if not n:
                    break
                got += n
            if not got:
                break
            total += got
            chunk_hashes.append(leaf(mv[:got]))
            if got < chunk_size:
                break
        if not chunk_hashes:
            if version == FORMAT_V1:
                raise ValueError("Cannot build tree from empty file")
            chunk_hashes.append(leaf(b""))
        return cls(chunk_hashes, chunk_size, version=version, algorithm=alg, num_bytes=total)


def _common_format(trees: Sequence[MerkleTree]) -> Tuple[int, HashAlgorithm]:
    """The (version, algorithm) shared by all trees; mixing is an error."""
    combos = {(t.version, t.algorithm) for t in trees}
    if not combos:
        return DEFAULT_FORMAT_VERSION, resolve_algorithm(None)
    if len(combos) != 1:
        desc = sorted(f"v{v}/{a.value}" for v, a in combos)
        raise ValueError(f"trees mix hash formats: {desc}")
    return next(iter(combos))


@dataclass
class LayerMerkleTree:
    """
    Merkle tree for a model layer.

    Attributes:
        layer_name: Name of the layer
        param_trees: Dict mapping parameter name to MerkleTree
        layer_root: Combined root hash for entire layer (in v2 an index
            digest that is not part of the model root)
    """
    layer_name: str
    param_trees: Dict[str, MerkleTree] = field(default_factory=dict)
    layer_root: Optional[str] = None

    def compute_layer_root(self) -> str:
        """
        Compute combined root hash for all parameters in layer.

        Returns:
            Layer root hash
        """
        if not self.param_trees:
            raise ValueError("No parameter trees in layer")

        version, algorithm = _common_format(list(self.param_trees.values()))
        if version == FORMAT_V2:
            names = sorted(self.param_trees, key=name_sort_key)
            roots = [self.param_trees[n].tensor_root(n) for n in names]
            self.layer_root = fmt.layer_digest(self.layer_name, roots, algorithm)
            return self.layer_root

        # Sort parameter names for deterministic ordering
        param_names = sorted(self.param_trees.keys())

        # Combine all parameter roots
        combined = ""
        for param_name in param_names:
            combined += self.param_trees[param_name].root_hash

        self.layer_root = compute_hash(combined, algorithm)
        return self.layer_root

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (v1 layout)"""
        return {
            'layer_name': self.layer_name,
            'layer_root': self.layer_root,
            'param_trees': {
                name: tree.to_dict()
                for name, tree in self.param_trees.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], algorithm: AlgorithmLike = None) -> 'LayerMerkleTree':
        """Deserialize from dictionary (v1 layout)"""
        layer_tree = cls(layer_name=data['layer_name'])
        layer_tree.layer_root = data['layer_root']
        layer_tree.param_trees = {
            name: MerkleTree.from_dict(tree_data, algorithm=algorithm)
            for name, tree_data in data['param_trees'].items()
        }
        return layer_tree


@dataclass
class TensorProof:
    """Proof that one tensor (name, metadata, chunk root) is committed by a v2 model root."""
    name: str
    dtype: Optional[str]
    shape: Optional[Tuple[int, ...]]
    num_bytes: int
    chunk_size: int
    chunk_root: str
    index: int
    num_tensors: int
    path: List[str]
    model_root: str
    algorithm: str

    def tensor_root(self) -> str:
        return fmt.tensor_root(self.name, self.chunk_root, self.num_bytes, self.chunk_size,
                               self.dtype, self.shape, self.algorithm)

    def verify(self) -> bool:
        tree_root = fmt.root_from_inclusion_path(
            self.tensor_root(), self.index, self.num_tensors, self.path, self.algorithm)
        if tree_root is None:
            return False
        return fmt.wrap_model_root(tree_root, self.num_tensors, self.algorithm) == self.model_root


@dataclass
class ChunkProof:
    """Proof that one chunk's bytes belong to a model with a given v2 model root."""
    chunk: MerkleProof
    tensor: TensorProof

    def verify(self, chunk_bytes: Optional[bytes] = None) -> bool:
        if self.chunk.root_hash != self.tensor.chunk_root:
            return False
        ok = self.chunk.verify() if chunk_bytes is None else self.chunk.verify_chunk(chunk_bytes)
        return ok and self.tensor.verify()


@dataclass
class ModelMerkleTree:
    """
    Complete Merkle tree for a model.

    Attributes:
        model_name: Name of model (a label; not part of any root)
        layer_trees: Dict mapping layer name to LayerMerkleTree
        model_root: Root hash for entire model
    """
    model_name: str = ""
    layer_trees: Dict[str, LayerMerkleTree] = field(default_factory=dict)
    model_root: Optional[str] = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def iter_params(self) -> Iterator[Tuple[str, MerkleTree]]:
        for layer_tree in self.layer_trees.values():
            for name, tree in layer_tree.param_trees.items():
                yield name, tree

    def get_param_tree(self, name: str) -> MerkleTree:
        for layer_tree in self.layer_trees.values():
            if name in layer_tree.param_trees:
                return layer_tree.param_trees[name]
        raise KeyError(name)

    @property
    def format_version(self) -> int:
        return _common_format([t for _, t in self.iter_params()])[0]

    @property
    def algorithm(self) -> HashAlgorithm:
        return _common_format([t for _, t in self.iter_params()])[1]

    def param_root(self, name: str) -> str:
        """The commitment for one parameter: v2 tensor root, or v1 chunk root."""
        return self.get_param_tree(name).tensor_root(name)

    def _sorted_params(self) -> List[Tuple[str, MerkleTree]]:
        params = list(self.iter_params())
        names = [n for n, _ in params]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tensor names across layers")
        return sorted(params, key=lambda p: name_sort_key(p[0]))

    # ------------------------------------------------------------------
    # Roots
    # ------------------------------------------------------------------

    def compute_model_root(self) -> str:
        """
        Compute root hash for entire model.

        Returns:
            Model root hash
        """
        if not self.layer_trees:
            raise ValueError("No layer trees in model")

        version, algorithm = _common_format([t for _, t in self.iter_params()])
        if version == FORMAT_V2:
            for layer_tree in self.layer_trees.values():
                layer_tree.compute_layer_root()
            roots = [tree.tensor_root(name) for name, tree in self._sorted_params()]
            self.model_root = fmt.model_root(roots, algorithm)
            return self.model_root

        # Sort layer names for deterministic ordering
        layer_names = sorted(self.layer_trees.keys())

        # Combine all layer roots
        combined = ""
        for layer_name in layer_names:
            layer_tree = self.layer_trees[layer_name]
            if layer_tree.layer_root is None:
                layer_tree.compute_layer_root()
            combined += layer_tree.layer_root

        self.model_root = compute_hash(combined, algorithm)
        return self.model_root

    def verify(self) -> bool:
        """
        Verify the internal consistency of this Merkle tree.

        Recomputes layer and model roots from the parameter trees and compares
        them with the stored values. It does not re-read any weights.

        Returns:
            True if all hashes match, False otherwise
        """
        if not self.layer_trees:
            return False

        version, algorithm = _common_format([t for _, t in self.iter_params()])
        if version == FORMAT_V2:
            for layer_name, layer_tree in self.layer_trees.items():
                names = sorted(layer_tree.param_trees, key=name_sort_key)
                expected = fmt.layer_digest(
                    layer_name, [layer_tree.param_trees[n].tensor_root(n) for n in names], algorithm)
                if expected != layer_tree.layer_root:
                    return False
            roots = [tree.tensor_root(name) for name, tree in self._sorted_params()]
            return fmt.model_root(roots, algorithm) == self.model_root

        # Verify each layer
        for layer_name, layer_tree in self.layer_trees.items():
            if not layer_tree.param_trees:
                continue

            # Recalculate layer root
            param_names = sorted(layer_tree.param_trees.keys())
            combined = ""
            for param_name in param_names:
                param_tree = layer_tree.param_trees[param_name]
                combined += param_tree.root_hash

            expected_layer_root = compute_hash(combined, algorithm)
            if expected_layer_root != layer_tree.layer_root:
                return False

        # Verify model root
        layer_names = sorted(self.layer_trees.keys())
        combined = ""
        for layer_name in layer_names:
            combined += self.layer_trees[layer_name].layer_root

        expected_model_root = compute_hash(combined, algorithm)
        return expected_model_root == self.model_root

    # ------------------------------------------------------------------
    # Proofs (v2)
    # ------------------------------------------------------------------

    def get_tensor_proof(self, name: str) -> TensorProof:
        """Prove that tensor ``name`` is committed by ``model_root`` (v2 only)."""
        if self.format_version != FORMAT_V2:
            raise ValueError("tensor inclusion proofs need a format v2 tree")
        params = self._sorted_params()
        names = [n for n, _ in params]
        try:
            index = names.index(name)
        except ValueError:
            raise KeyError(name) from None
        algorithm = self.algorithm
        roots = [t.tensor_root(n) for n, t in params]
        tree = params[index][1]
        return TensorProof(
            name=name, dtype=tree.dtype, shape=tree.shape, num_bytes=tree.num_bytes,
            chunk_size=tree.chunk_size, chunk_root=tree.root_hash, index=index,
            num_tensors=len(params), path=fmt.inclusion_path(roots, index, algorithm),
            model_root=self.model_root, algorithm=algorithm.value,
        )

    def get_chunk_proof(self, name: str, chunk_index: int) -> ChunkProof:
        """Prove that chunk ``chunk_index`` of tensor ``name`` belongs to this model (v2 only)."""
        return ChunkProof(
            chunk=self.get_param_tree(name).get_proof(chunk_index),
            tensor=self.get_tensor_proof(name),
        )

    # ------------------------------------------------------------------
    # Diff helpers
    # ------------------------------------------------------------------

    def get_changed_layers(self, other: 'ModelMerkleTree') -> List[str]:
        """
        Find layers that differ from another model.

        Args:
            other: Other ModelMerkleTree to compare

        Returns:
            List of changed layer names
        """
        changed = []

        all_layers = set(self.layer_trees.keys()) | set(other.layer_trees.keys())

        for layer_name in all_layers:
            if layer_name not in self.layer_trees or layer_name not in other.layer_trees:
                changed.append(layer_name)
            elif self.layer_trees[layer_name].layer_root != other.layer_trees[layer_name].layer_root:
                changed.append(layer_name)

        return changed

    def get_changed_chunks(
        self,
        other: 'ModelMerkleTree',
        layer_name: str
    ) -> Dict[str, List[int]]:
        """
        Find changed chunks in a specific layer.

        Args:
            other: Other ModelMerkleTree
            layer_name: Layer to compare

        Returns:
            Dict mapping param name to list of changed chunk indices
        """
        if layer_name not in self.layer_trees or layer_name not in other.layer_trees:
            raise ValueError(f"Layer {layer_name} not in both models")

        self_layer = self.layer_trees[layer_name]
        other_layer = other.layer_trees[layer_name]

        changed_chunks = {}

        all_params = set(self_layer.param_trees.keys()) | set(other_layer.param_trees.keys())

        for param_name in all_params:
            if param_name not in self_layer.param_trees or param_name not in other_layer.param_trees:
                # Entire parameter changed
                continue

            self_tree = self_layer.param_trees[param_name]
            other_tree = other_layer.param_trees[param_name]

            # Use tree-walk diff for O(k log C) instead of O(C) linear scan
            changed_indices, _ = self_tree.diff_tree(other_tree)

            if changed_indices:
                changed_chunks[param_name] = changed_indices

        return changed_chunks

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self, include_chunks: bool = True) -> Dict[str, Any]:
        """Serialize to a manifest dictionary.

        Args:
            include_chunks: v2 only. False writes a compact manifest without
                per-chunk hashes (verifies and diffs per tensor, not per chunk).
        """
        if not self.layer_trees or self.format_version == FORMAT_V1:
            return {
                'model_name': self.model_name,
                'model_root': self.model_root,
                'layer_trees': {
                    name: tree.to_dict()
                    for name, tree in self.layer_trees.items()
                }
            }

        tensors: Dict[str, Any] = {}
        for name, tree in self._sorted_params():
            entry: Dict[str, Any] = {
                'dtype': tree.dtype,
                'shape': list(tree.shape) if tree.shape is not None else None,
                'num_bytes': tree.num_bytes,
                'chunk_size': tree.chunk_size,
                'num_chunks': tree.num_chunks,
                'chunk_root': tree.root_hash,
                'tensor_root': tree.tensor_root(name),
            }
            if include_chunks and tree.chunk_hashes is not None:
                entry['chunk_hashes'] = tree.chunk_hashes
            tensors[name] = entry
        return {
            'format': FORMAT_NAME,
            'format_version': FORMAT_V2,
            'algorithm': self.algorithm.value,
            'model_name': self.model_name,
            'model_root': self.model_root,
            'num_tensors': len(tensors),
            'created_by': _created_by(),
            'tensors': tensors,
        }

    def to_json(self, filepath: str, include_chunks: bool = True):
        """Save to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(include_chunks=include_chunks), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], algorithm: AlgorithmLike = None) -> 'ModelMerkleTree':
        """Deserialize a manifest dictionary.

        v2 manifests are fully re-derived and rejected (ValueError) if any
        stored chunk root, tensor root or model root disagrees with what its
        inputs imply. v1 manifests carry no algorithm; ``algorithm`` (default:
        process default) is used for them.
        """
        version = data.get('format_version', FORMAT_V1)
        if check_format_version(version) == FORMAT_V1:
            model_tree = cls(model_name=data.get('model_name', ''))
            model_tree.model_root = data['model_root']
            model_tree.layer_trees = {
                name: LayerMerkleTree.from_dict(tree_data, algorithm=algorithm)
                for name, tree_data in data['layer_trees'].items()
            }
            return model_tree

        if data.get('format', FORMAT_NAME) != FORMAT_NAME:
            raise ValueError(f"not a {FORMAT_NAME} manifest: format={data.get('format')!r}")
        alg = resolve_algorithm(data['algorithm'])
        param_trees: Dict[str, MerkleTree] = {}
        for name, e in data['tensors'].items():
            num_bytes, chunk_size = int(e['num_bytes']), int(e['chunk_size'])
            if num_bytes < 0 or chunk_size <= 0:
                raise ValueError(f"manifest is inconsistent: {name!r} num_bytes/chunk_size")
            expected_chunks = max(1, -(-num_bytes // chunk_size))
            if int(e['num_chunks']) != expected_chunks:
                raise ValueError(f"manifest is inconsistent: {name!r} num_chunks")
            if 'chunk_hashes' in e:
                tree = MerkleTree(e['chunk_hashes'], chunk_size, version=FORMAT_V2, algorithm=alg,
                                  num_bytes=num_bytes, dtype=e.get('dtype'), shape=e.get('shape'))
                if tree.num_chunks != expected_chunks or tree.root_hash != e['chunk_root']:
                    raise ValueError(f"manifest is inconsistent: {name!r} chunk_hashes")
            else:
                tree = MerkleTree.from_root(e['chunk_root'], expected_chunks, chunk_size,
                                            algorithm=alg, num_bytes=num_bytes,
                                            dtype=e.get('dtype'), shape=e.get('shape'))
            if tree.tensor_root(name) != e['tensor_root']:
                raise ValueError(f"manifest is inconsistent: {name!r} tensor_root")
            param_trees[name] = tree

        model_tree = assemble_model_tree(param_trees, model_name=data.get('model_name', ''))
        if model_tree.model_root != data['model_root']:
            raise ValueError("manifest is inconsistent: model_root")
        if data.get('num_tensors', len(param_trees)) != len(param_trees):
            raise ValueError("manifest is inconsistent: num_tensors")
        return model_tree

    @classmethod
    def from_json(cls, filepath: str, algorithm: AlgorithmLike = None) -> 'ModelMerkleTree':
        """Load from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, algorithm=algorithm)

    @property
    def total_params(self) -> int:
        """
        Get total parameter count.

        Exact when every tensor's dtype and shape are known (v2 trees built
        from tensors); otherwise estimated from chunk counts assuming float32.
        """
        trees = [t for _, t in self.iter_params()]
        if trees and all(t.shape is not None for t in trees):
            return sum(math.prod(t.shape) if t.shape else 1 for t in trees)

        total_chunks = 0
        chunk_size = 16384  # Default

        for tree in trees:
            total_chunks += tree.num_chunks
            chunk_size = tree.chunk_size

        # Rough estimate: assume float32 (4 bytes per param)
        total_bytes = total_chunks * chunk_size
        return total_bytes // 4


def _extract_layer_name(param_name: str) -> str:
    """
    Extract layer name from parameter name.

    Supports multiple model architectures:
    - GPT-2: transformer.h.0.* -> transformer.h.0
    - LLaMA: model.layers.0.* -> model.layers.0
    - BERT: encoder.layer.0.* -> encoder.layer.0
    - Generic: first two parts or 'other'

    Args:
        param_name: Full parameter name

    Returns:
        Layer name string
    """
    parts = param_name.split('.')

    # GPT-2 style: transformer.h.N
    if len(parts) >= 3 and parts[0] == 'transformer' and parts[1] == 'h':
        return f"{parts[0]}.{parts[1]}.{parts[2]}"

    # LLaMA style: model.layers.N
    if len(parts) >= 3 and parts[0] == 'model' and parts[1] == 'layers':
        return f"{parts[0]}.{parts[1]}.{parts[2]}"

    # BERT style: encoder.layer.N
    if len(parts) >= 3 and parts[0] == 'encoder' and parts[1] == 'layer':
        return f"{parts[0]}.{parts[1]}.{parts[2]}"

    # Generic: try to identify numbered layers
    for i, part in enumerate(parts):
        if part.isdigit() and i > 0:
            return '.'.join(parts[:i+1])

    # Default: use first part or 'other'
    return parts[0] if parts else "other"


def _hash_chunks(
    data: bytes,
    chunk_size: int,
    version: int = FORMAT_V1,
    algorithm: AlgorithmLike = None,
) -> List[str]:
    """Hash ``data`` in fixed-size chunks with the leaf rule of ``version``.

    Zero-length data (e.g. a tensor with a 0-sized dimension) is a single
    empty chunk, so every parameter has a well-defined root.
    """
    return hash_buffer(data, chunk_size, leaf_hasher(version, algorithm))


def _build_param_tree(
    param_name: str,
    data: Any,
    chunk_size: int,
    version: int = DEFAULT_FORMAT_VERSION,
    algorithm: AlgorithmLike = None,
) -> Tuple[str, MerkleTree]:
    """
    Build Merkle tree for a single named blob (used for parallel processing).

    Args:
        param_name: Name / key for this blob
        data: Raw bytes, a torch.Tensor or a numpy array
        chunk_size: Chunk size in bytes
        version: Hash format
        algorithm: Hash algorithm

    Returns:
        Tuple of (param_name, MerkleTree)
    """
    raw, meta = to_bytes_and_meta(data)
    dtype, shape = meta if meta is not None else (None, None)
    alg = resolve_algorithm(algorithm)
    tree = MerkleTree(
        _hash_chunks(raw, chunk_size, version, alg), chunk_size,
        version=version, algorithm=alg, num_bytes=len(raw), dtype=dtype, shape=shape,
    )
    return param_name, tree


def assemble_model_tree(
    param_trees: Dict[str, MerkleTree],
    model_name: str = "",
) -> ModelMerkleTree:
    """
    Group already-built parameter trees into layers and compute all roots.

    This is the step shared by every builder (in-memory blobs, streamed
    safetensors files, sharded checkpoints, manifests).

    Args:
        param_trees: Dict mapping parameter name to its chunk-level MerkleTree
            (all with the same format version and algorithm)
        model_name: Name of model

    Returns:
        ModelMerkleTree with layer and model roots computed
    """
    if not param_trees:
        raise ValueError("No layer trees in model: no parameters given")
    _common_format(list(param_trees.values()))

    model_tree = ModelMerkleTree(model_name=model_name)
    for param_name in sorted(param_trees, key=name_sort_key):
        layer_name = _extract_layer_name(param_name)
        layer_tree = model_tree.layer_trees.get(layer_name)
        if layer_tree is None:
            layer_tree = LayerMerkleTree(layer_name=layer_name)
            model_tree.layer_trees[layer_name] = layer_tree
        layer_tree.param_trees[param_name] = param_trees[param_name]

    for layer_tree in model_tree.layer_trees.values():
        layer_tree.compute_layer_root()
    model_tree.compute_model_root()
    return model_tree


def build_model_merkle_tree(
    blobs: Dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: str = "",
    num_workers: Optional[int] = None,
    parallel: bool = True,
    *,
    format_version: int = DEFAULT_FORMAT_VERSION,
    algorithm: AlgorithmLike = None,
) -> ModelMerkleTree:
    """
    Build complete Merkle tree from a dict of named blobs.

    Values may be raw ``bytes``, ``torch.Tensor`` objects (any dtype,
    including bfloat16) or numpy arrays, so a PyTorch ``state_dict()`` can be
    passed directly. Tensors are hashed as their contiguous little-endian
    bytes, which is exactly what safetensors stores on disk; in format v2
    their dtype and shape are bound into the root as well.

    Args:
        blobs: Dict mapping parameter/file name to bytes or a tensor
        chunk_size: Size of chunks in bytes
        model_name: Name of model
        num_workers: Number of parallel workers (default: auto)
        parallel: Whether to use parallel processing (default: True)
        format_version: Hash format, 2 (default) or 1 (0.2.x compatible)
        algorithm: Hash algorithm (default: process default, SHA-256)

    Returns:
        ModelMerkleTree
    """
    check_format_version(format_version)
    alg = resolve_algorithm(algorithm)
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    if parallel and num_workers > 1 and len(blobs) > 4:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_build_param_tree, name, data, chunk_size, format_version, alg)
                for name, data in blobs.items()
            ]
            param_trees = dict(f.result() for f in futures)
    else:
        param_trees = dict(
            _build_param_tree(name, data, chunk_size, format_version, alg)
            for name, data in blobs.items()
        )

    return assemble_model_tree(param_trees, model_name=model_name)


def build_file_merkle_tree(
    filepath: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    *,
    format_version: int = DEFAULT_FORMAT_VERSION,
    algorithm: AlgorithmLike = None,
    num_workers: Optional[int] = None,
) -> MerkleTree:
    """
    Build a Merkle tree by streaming a file from disk.

    Memory usage is O(chunk_size), not O(file_size).
    Suitable for multi-GB model files.

    Args:
        filepath: Path to any binary file
        chunk_size: Size of each chunk in bytes
        format_version: Hash format (default 2)
        algorithm: Hash algorithm (default: process default)
        num_workers: Hashing threads (default: auto)

    Returns:
        MerkleTree

    Example:
        >>> tree = build_file_merkle_tree("model.safetensors")
        >>> print(tree.root_hash)
    """
    check_format_version(format_version)
    alg = resolve_algorithm(algorithm)
    size = os.path.getsize(filepath)
    if size == 0 and format_version == FORMAT_V1:
        raise ValueError("Cannot build tree from empty file")
    leaves = hash_file_ranges(
        [("file", filepath, 0, size)], chunk_size, leaf_hasher(format_version, alg),
        workers=num_workers or DEFAULT_NUM_WORKERS,
    )["file"]
    return MerkleTree(leaves, chunk_size, version=format_version, algorithm=alg, num_bytes=size)
