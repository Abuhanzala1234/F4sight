"""Merkle trees (§7.12). Every size 1..257, because off-by-ones live at the
odd/even boundary."""

from __future__ import annotations

import hashlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from drishti_worker.merkle import (
    LEAF_PREFIX,
    NODE_PREFIX,
    build_tree,
    proof,
    verify_proof,
)


def leaf(i: int) -> str:
    return hashlib.sha256(f"evidence-{i}".encode()).hexdigest()


@pytest.mark.parametrize(
    "n", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256, 257]
)
def test_every_leaf_verifies(n):
    leaves = [leaf(i) for i in range(n)]
    tree = build_tree(leaves)
    for i in range(n):
        assert verify_proof(leaves[i], proof(tree, i), tree.root), (n, i)


def test_all_sizes_1_to_257():
    """The exhaustive version of the above — this is the acceptance criterion
    for Phase 7 (§16)."""
    for n in range(1, 258):
        leaves = [leaf(i) for i in range(n)]
        tree = build_tree(leaves)
        for i in (0, n // 2, n - 1):
            assert verify_proof(leaves[i], proof(tree, i), tree.root), (n, i)


def test_forged_leaf_does_not_verify():
    leaves = [leaf(i) for i in range(8)]
    tree = build_tree(leaves)
    forged = hashlib.sha256(b"i was never in this batch").hexdigest()
    assert not verify_proof(forged, proof(tree, 0), tree.root)


def test_tampered_proof_does_not_verify():
    leaves = [leaf(i) for i in range(8)]
    tree = build_tree(leaves)
    path = proof(tree, 3)
    path[0] = (path[0][0], hashlib.sha256(b"wrong sibling").hexdigest())
    assert not verify_proof(leaves[3], path, tree.root)


def test_domain_separation_prevents_node_as_leaf():
    """Without the 0x00/0x01 prefixes an internal node's preimage could be
    presented as a leaf. This is the classic Merkle forgery."""
    assert LEAF_PREFIX != NODE_PREFIX
    leaves = [leaf(i) for i in range(4)]
    tree = build_tree(leaves)
    internal = tree.levels[1][0]
    # The internal node must not verify as if it were a leaf of the tree.
    assert not verify_proof(internal, proof(tree, 0), tree.root)


def test_root_changes_if_any_leaf_changes():
    base = build_tree([leaf(i) for i in range(16)]).root
    altered = [leaf(i) for i in range(16)]
    altered[7] = hashlib.sha256(b"altered").hexdigest()
    assert build_tree(altered).root != base


def test_order_matters():
    a = build_tree([leaf(0), leaf(1)]).root
    b = build_tree([leaf(1), leaf(0)]).root
    assert a != b


def test_empty_batch_raises():
    """An empty tree has no meaningful root; anchoring a constant would be
    worse than refusing."""
    with pytest.raises(ValueError):
        build_tree([])


def test_non_hex_leaf_raises():
    with pytest.raises(ValueError):
        build_tree(["not a hash"])


def test_short_leaf_raises():
    with pytest.raises(ValueError):
        build_tree(["abcd"])


def test_index_out_of_range():
    tree = build_tree([leaf(0), leaf(1)])
    with pytest.raises(IndexError):
        proof(tree, 5)


@given(n=st.integers(min_value=1, max_value=64), i=st.integers(min_value=0))
@settings(max_examples=300)
def test_property_any_index_verifies(n, i):
    index = i % n
    leaves = [leaf(k) for k in range(n)]
    tree = build_tree(leaves)
    assert verify_proof(leaves[index], proof(tree, index), tree.root)
    assert tree.index_of(leaves[index]) == index


@given(n=st.integers(min_value=1, max_value=64))
@settings(max_examples=100)
def test_proof_length_is_tree_depth(n):
    leaves = [leaf(k) for k in range(n)]
    tree = build_tree(leaves)
    assert len(proof(tree, 0)) == tree.depth - 1
