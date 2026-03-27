"""
Merkle Tree implementation for model weights
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os

from merkle_verify.hasher import compute_hash, hash_pair
from merkle_verify.chunking import chunk_bytes, DEFAULT_CHUNK_SIZE

# Default number of workers for parallel processing
DEFAULT_NUM_WORKERS = min(4, os.cpu_count() or 1)


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
    """Merkle proof for a chunk"""
    chunk_index: int
    chunk_hash: str
    sibling_hashes: List[Tuple[str, str]]  # (hash, position: 'left'/'right')
    root_hash: str

    def verify(self) -> bool:
        """Verify this proof leads to root hash"""
        current = self.chunk_hash
        for sibling_hash, position in self.sibling_hashes:
            if position == 'left':
                current = hash_pair(sibling_hash, current)
            else:
                current = hash_pair(current, sibling_hash)
        return current == self.root_hash


class MerkleTree:
    """
    Merkle Tree for a sequence of chunks.

    Attributes:
        chunk_hashes: List of leaf hashes
        root: Root node of tree
        root_hash: Hash at root
        chunk_size: Size of each chunk in bytes
    """

    def __init__(self, chunk_hashes: List[str], chunk_size: int = DEFAULT_CHUNK_SIZE):
        """
        Build Merkle tree from chunk hashes.

        Args:
            chunk_hashes: List of hashes for each chunk
            chunk_size: Size of chunks in bytes
        """
        self.chunk_hashes = chunk_hashes
        self.chunk_size = chunk_size
        self.num_chunks = len(chunk_hashes)

        # Build tree
        self.root = self._build_tree(chunk_hashes)
        self.root_hash = self.root.hash_value

    def _build_tree(self, hashes: List[str]) -> MerkleNode:
        """
        Recursively build Merkle tree.

        Args:
            hashes: List of hashes at current level

        Returns:
            Root node of (sub)tree
        """
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

                parent_hash = hash_pair(left.hash_value, right.hash_value)
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

    def get_proof(self, chunk_index: int) -> MerkleProof:
        """
        Generate Merkle proof for a chunk.

        Args:
            chunk_index: Index of chunk

        Returns:
            MerkleProof object
        """
        if chunk_index >= self.num_chunks:
            raise ValueError(f"Chunk index {chunk_index} out of range")

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
            root_hash=self.root_hash
        )

    def _get_level_hashes(self, level_size: int) -> List[str]:
        """Get all hashes at a level of given size"""
        if level_size == self.num_chunks:
            return self.chunk_hashes

        # Build up from leaves
        current_level = self.chunk_hashes[:]
        while len(current_level) > level_size:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else current_level[i]
                next_level.append(hash_pair(left, right))
            current_level = next_level

        return current_level

    def diff_tree(self, other: 'MerkleTree') -> Tuple[List[int], int]:
        """
        Find changed chunk indices by walking both trees in parallel.

        Only descends into subtrees where hashes differ, achieving
        O(k * log C) complexity where k is the number of changed chunks
        and C is the total number of chunks.

        Uses chunk_start/chunk_count metadata on each node to correctly
        track which chunk indices each subtree covers, handling the
        odd-count duplication in bottom-up tree construction.

        Args:
            other: Another MerkleTree to compare against

        Returns:
            Tuple of (changed_chunk_indices, hash_comparisons_count)
        """
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

    def _diff_walk(
        self,
        node_a: Optional['MerkleNode'],
        node_b: Optional['MerkleNode'],
        changed: List[int],
        comparisons: List[int],
    ) -> None:
        """
        Recursively walk both trees, only descending where hashes differ.

        Uses chunk_start and chunk_count metadata stored on each node
        to correctly identify which chunk indices are affected.

        Args:
            node_a: Node from self's tree
            node_b: Node from other's tree
            changed: Accumulator list for changed chunk indices
            comparisons: Single-element list tracking hash comparison count
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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'root_hash': self.root_hash,
            'num_chunks': self.num_chunks,
            'chunk_size': self.chunk_size,
            'chunk_hashes': self.chunk_hashes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MerkleTree':
        """Deserialize from dictionary"""
        return cls(
            chunk_hashes=data['chunk_hashes'],
            chunk_size=data['chunk_size']
        )

    @classmethod
    def from_file(
        cls,
        file_handle,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> 'MerkleTree':
        """
        Build a Merkle tree by streaming from a file handle.

        Memory usage is O(chunk_size + num_chunks * hash_length),
        NOT O(file_size). Suitable for multi-GB files.

        Args:
            file_handle: Readable binary file object (supports .read(n))
            chunk_size: Size of each chunk in bytes

        Returns:
            MerkleTree

        Example:
            >>> with open("model.bin", "rb") as f:
            ...     tree = MerkleTree.from_file(f)
        """
        chunk_hashes = []
        while True:
            chunk = file_handle.read(chunk_size)
            if not chunk:
                break
            chunk_hashes.append(compute_hash(chunk))
        if not chunk_hashes:
            raise ValueError("Cannot build tree from empty file")
        return cls(chunk_hashes, chunk_size)


@dataclass
class LayerMerkleTree:
    """
    Merkle tree for a model layer.

    Attributes:
        layer_name: Name of the layer
        param_trees: Dict mapping parameter name to MerkleTree
        layer_root: Combined root hash for entire layer
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

        # Sort parameter names for deterministic ordering
        param_names = sorted(self.param_trees.keys())

        # Combine all parameter roots
        combined = ""
        for param_name in param_names:
            combined += self.param_trees[param_name].root_hash

        self.layer_root = compute_hash(combined)
        return self.layer_root

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'layer_name': self.layer_name,
            'layer_root': self.layer_root,
            'param_trees': {
                name: tree.to_dict()
                for name, tree in self.param_trees.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerMerkleTree':
        """Deserialize from dictionary"""
        layer_tree = cls(layer_name=data['layer_name'])
        layer_tree.layer_root = data['layer_root']
        layer_tree.param_trees = {
            name: MerkleTree.from_dict(tree_data)
            for name, tree_data in data['param_trees'].items()
        }
        return layer_tree


@dataclass
class ModelMerkleTree:
    """
    Complete Merkle tree for a model.

    Attributes:
        model_name: Name of model
        layer_trees: Dict mapping layer name to LayerMerkleTree
        model_root: Root hash for entire model
    """
    model_name: str = ""
    layer_trees: Dict[str, LayerMerkleTree] = field(default_factory=dict)
    model_root: Optional[str] = None

    def compute_model_root(self) -> str:
        """
        Compute root hash for entire model.

        Returns:
            Model root hash
        """
        if not self.layer_trees:
            raise ValueError("No layer trees in model")

        # Sort layer names for deterministic ordering
        layer_names = sorted(self.layer_trees.keys())

        # Combine all layer roots
        combined = ""
        for layer_name in layer_names:
            layer_tree = self.layer_trees[layer_name]
            if layer_tree.layer_root is None:
                layer_tree.compute_layer_root()
            combined += layer_tree.layer_root

        self.model_root = compute_hash(combined)
        return self.model_root

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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'model_name': self.model_name,
            'model_root': self.model_root,
            'layer_trees': {
                name: tree.to_dict()
                for name, tree in self.layer_trees.items()
            }
        }

    def to_json(self, filepath: str):
        """Save to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMerkleTree':
        """Deserialize from dictionary"""
        model_tree = cls(model_name=data.get('model_name', ''))
        model_tree.model_root = data['model_root']
        model_tree.layer_trees = {
            name: LayerMerkleTree.from_dict(tree_data)
            for name, tree_data in data['layer_trees'].items()
        }
        return model_tree

    @classmethod
    def from_json(cls, filepath: str) -> 'ModelMerkleTree':
        """Load from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def verify(self) -> bool:
        """
        Verify the integrity of this Merkle tree.

        Recalculates all hashes and compares them with stored values.

        Returns:
            True if all hashes match, False otherwise
        """
        if not self.layer_trees:
            return False

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

            expected_layer_root = compute_hash(combined)
            if expected_layer_root != layer_tree.layer_root:
                return False

        # Verify model root
        layer_names = sorted(self.layer_trees.keys())
        combined = ""
        for layer_name in layer_names:
            combined += self.layer_trees[layer_name].layer_root

        expected_model_root = compute_hash(combined)
        return expected_model_root == self.model_root

    @property
    def total_params(self) -> int:
        """
        Get estimated total parameter count.

        Returns:
            Estimated number of parameters based on chunk count and size.
        """
        total_chunks = 0
        chunk_size = 16384  # Default

        for layer_tree in self.layer_trees.values():
            for param_tree in layer_tree.param_trees.values():
                total_chunks += param_tree.num_chunks
                chunk_size = param_tree.chunk_size

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


def _build_param_tree(
    param_name: str,
    data: bytes,
    chunk_size: int
) -> Tuple[str, MerkleTree]:
    """
    Build Merkle tree for a single named blob (used for parallel processing).

    Args:
        param_name: Name / key for this blob
        data: Raw bytes to hash
        chunk_size: Chunk size in bytes

    Returns:
        Tuple of (param_name, MerkleTree)
    """
    chunks = chunk_bytes(data, chunk_size)
    chunk_hashes = [compute_hash(chunk) for chunk in chunks]
    param_tree = MerkleTree(chunk_hashes, chunk_size)
    return param_name, param_tree


def build_model_merkle_tree(
    blobs: Dict[str, bytes],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: str = "",
    num_workers: Optional[int] = None,
    parallel: bool = True,
) -> ModelMerkleTree:
    """
    Build complete Merkle tree from a dict of named byte blobs.

    Works with any dict mapping names to bytes -- for PyTorch models,
    serialize tensors first (e.g. ``{k: v.numpy().tobytes() for k, v in state_dict.items()}``).

    Args:
        blobs: Dict mapping parameter/file name to raw bytes
        chunk_size: Size of chunks in bytes
        model_name: Name of model
        num_workers: Number of parallel workers (default: auto)
        parallel: Whether to use parallel processing (default: True)

    Returns:
        ModelMerkleTree
    """
    model_tree = ModelMerkleTree(model_name=model_name)

    # Group by layer using generic layer extraction
    layer_params: Dict[str, Dict[str, bytes]] = {}
    for param_name, data in blobs.items():
        layer_name = _extract_layer_name(param_name)
        if layer_name not in layer_params:
            layer_params[layer_name] = {}
        layer_params[layer_name][param_name] = data

    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    if parallel and num_workers > 1 and len(blobs) > 4:
        for layer_name, params in layer_params.items():
            layer_tree = LayerMerkleTree(layer_name=layer_name)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_build_param_tree, name, data, chunk_size): name
                    for name, data in params.items()
                }
                for future in as_completed(futures):
                    name, param_tree = future.result()
                    layer_tree.param_trees[name] = param_tree
            layer_tree.compute_layer_root()
            model_tree.layer_trees[layer_name] = layer_tree
    else:
        for layer_name, params in layer_params.items():
            layer_tree = LayerMerkleTree(layer_name=layer_name)
            for param_name, data in params.items():
                chunks = chunk_bytes(data, chunk_size)
                chunk_hashes = [compute_hash(chunk) for chunk in chunks]
                param_tree = MerkleTree(chunk_hashes, chunk_size)
                layer_tree.param_trees[param_name] = param_tree
            layer_tree.compute_layer_root()
            model_tree.layer_trees[layer_name] = layer_tree

    model_tree.compute_model_root()
    return model_tree


def build_file_merkle_tree(
    filepath: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> MerkleTree:
    """
    Build a Merkle tree by streaming a file from disk.

    Memory usage is O(chunk_size), not O(file_size).
    Suitable for multi-GB model files.

    Args:
        filepath: Path to any binary file
        chunk_size: Size of each chunk in bytes

    Returns:
        MerkleTree

    Example:
        >>> tree = build_file_merkle_tree("model.safetensors")
        >>> print(tree.root_hash)
    """
    with open(filepath, 'rb') as f:
        return MerkleTree.from_file(f, chunk_size)
