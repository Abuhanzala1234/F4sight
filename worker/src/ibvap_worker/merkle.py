"""Merkle trees for evidence anchoring (BUILD_SPEC §7.12).

One ledger write per alert would be slow, expensive and pointless. Instead the
anchor service batches alert hashes into a Merkle tree every
``anchor_interval_s`` and writes a single 32-byte root. Each alert keeps its
inclusion proof, so any single alert can be proven to belong to that root
without revealing or transmitting the others — which matters, because the other
alerts in the batch may be from a different sector.

Two details that are easy to get wrong and expensive to get wrong:

**Domain separation.** Leaves are hashed ``0x00 || leaf`` and internal nodes
``0x01 || left || right``. Without it, an internal node's 64-byte preimage can
be presented as if it were a leaf, and you can forge an inclusion proof for data
that was never in the tree. This is textbook (CVE-2012-2459 in Bitcoin is the
famous instance) and it costs one byte to prevent.

**Odd node counts.** When a level has an odd number of nodes we duplicate the
last one. Every off-by-one in Merkle code lives at this boundary, which is why
the property tests cover every tree size from 1 to 257 rather than a few
convenient powers of two.

Pure module: hashlib only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "LEAF_PREFIX",
    "NODE_PREFIX",
    "MerkleTree",
    "build_tree",
    "proof",
    "verify_proof",
]

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"

Side = Literal["L", "R"]
ProofPath = list[tuple[Side, str]]


def _hash_leaf(leaf_hex: str) -> str:
    """Leaf hash with domain separation: sha256(0x00 || leaf_bytes)."""
    try:
        raw = bytes.fromhex(leaf_hex)
    except ValueError as exc:
        raise ValueError(f"leaf is not hex: {leaf_hex!r}") from exc
    if len(raw) != 32:
        raise ValueError(f"leaf must be a 32-byte SHA-256 hex digest, got {len(raw)} bytes")
    return hashlib.sha256(LEAF_PREFIX + raw).hexdigest()


def _hash_node(left_hex: str, right_hex: str) -> str:
    """Internal node: sha256(0x01 || left || right)."""
    return hashlib.sha256(
        NODE_PREFIX + bytes.fromhex(left_hex) + bytes.fromhex(right_hex)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class MerkleTree:
    """An immutable Merkle tree.

    ``levels[0]`` is the hashed leaves, ``levels[-1]`` is a single-element list
    holding the root. ``leaves`` keeps the original (unprefixed) inputs so a
    caller can map an alert back to its index.
    """

    leaves: tuple[str, ...]
    levels: tuple[tuple[str, ...], ...]

    @property
    def root(self) -> str:
        return self.levels[-1][0]

    @property
    def leaf_count(self) -> int:
        return len(self.leaves)

    @property
    def depth(self) -> int:
        return len(self.levels)

    def index_of(self, leaf: str) -> int:
        try:
            return self.leaves.index(leaf)
        except ValueError as exc:
            raise ValueError(f"leaf {leaf[:16]}… is not in this tree") from exc


def build_tree(leaves: Sequence[str]) -> MerkleTree:
    """Build a tree over SHA-256 hex digests.

    Raises on an empty batch: an empty tree has no meaningful root, and
    silently anchoring a constant would be worse than refusing.
    """
    if not leaves:
        raise ValueError("cannot build a Merkle tree over zero leaves")

    level: list[str] = [_hash_leaf(x) for x in leaves]
    levels: list[tuple[str, ...]] = [tuple(level)]

    while len(level) > 1:
        if len(level) % 2 == 1:
            level = [*level, level[-1]]  # duplicate the last node
        level = [_hash_node(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        levels.append(tuple(level))

    return MerkleTree(leaves=tuple(leaves), levels=tuple(levels))


def proof(tree: MerkleTree, index: int) -> ProofPath:
    """Inclusion proof for ``leaves[index]``, as a list of (side, sibling_hash).

    ``side`` is the position of the SIBLING, so verification concatenates in the
    right order: ``"L"`` means sibling-then-us, ``"R"`` means us-then-sibling.
    """
    if not 0 <= index < tree.leaf_count:
        raise IndexError(f"leaf index {index} out of range (0..{tree.leaf_count - 1})")

    path: ProofPath = []
    idx = index
    for level in tree.levels[:-1]:
        nodes = list(level)
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])  # mirror build_tree's duplication
        sibling = idx ^ 1
        side: Side = "L" if sibling < idx else "R"
        path.append((side, nodes[sibling]))
        idx //= 2
    return path


def verify_proof(leaf: str, path: Sequence[tuple[str, str]], root: str) -> bool:
    """Recompute the root from a leaf and its proof.

    Returns a bool because this is the one place a plain answer is right: the
    caller (``/alerts/{id}/verify``) reports every intermediate value itself.
    """
    try:
        current = _hash_leaf(leaf)
    except ValueError:
        return False

    for side, sibling in path:
        if side not in ("L", "R"):
            raise ValueError(f"bad proof side {side!r}; expected 'L' or 'R'")
        try:
            current = _hash_node(sibling, current) if side == "L" else _hash_node(current, sibling)
        except ValueError:
            return False
    return current == root
