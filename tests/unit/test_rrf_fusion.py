import pytest

from search.rrf_fusion import rrf_fuse


def test_document_in_both_lists_wins():
    fused = rrf_fuse({"dense": ["a", "b", "c"], "sparse": ["b", "d"]})
    assert fused[0]["key"] == "b"
    assert fused[0]["ranks"] == {"dense": 2, "sparse": 1}


def test_scores_match_formula():
    k = 60
    fused = rrf_fuse({"dense": ["a"], "sparse": ["a"]}, k=k)
    assert fused[0]["score"] == pytest.approx(1 / (k + 1) + 1 / (k + 1))


def test_weights_bias_the_fusion():
    lists = {"dense": ["d1", "shared"], "sparse": ["s1", "shared"]}
    dense_heavy = rrf_fuse(lists, weights={"dense": 2.0, "sparse": 0.0})
    assert [r["key"] for r in dense_heavy[:2]] == ["d1", "shared"]
    assert all(r["key"] != "s1" or r["score"] == 0 for r in dense_heavy)


def test_duplicates_within_a_list_count_once():
    fused = rrf_fuse({"dense": ["a", "a", "a"]})
    assert len(fused) == 1
    assert fused[0]["score"] == pytest.approx(1 / 61)


def test_equal_fused_scores_ordered_by_better_sparse_rank():
    # "z" has dense rank 2 but sparse rank 1, so it wins against key order.
    fused = rrf_fuse({"dense": ["a", "z"], "sparse": ["z", "a"]})
    assert fused[0]["score"] == pytest.approx(fused[1]["score"])
    assert [r["key"] for r in fused] == ["z", "a"]


def test_swapped_ranks_tie_goes_to_better_sparse_rank():
    fused = rrf_fuse({"dense": ["near", "sku"], "sparse": ["sku", "near"]})
    assert fused[0]["key"] == "sku"
    assert fused[0]["ranks"] == {"dense": 2, "sparse": 1}
    for r in fused:
        assert r["score"] == pytest.approx(1 / 61 + 1 / 62)

    # Key order alone would pick "a_near"; the sparse rank must win instead.
    fused = rrf_fuse({"dense": ["a_near", "z_sku"], "sparse": ["z_sku", "a_near"]})
    assert [r["key"] for r in fused] == ["z_sku", "a_near"]


def test_tie_with_rank_on_one_side_only():
    # Sparse rank 1 beats a missing sparse rank, against key order.
    fused = rrf_fuse({"dense": ["a"], "sparse": ["z"]})
    assert [r["key"] for r in fused] == ["z", "a"]

    # No sparse ranks at all: dense rank 1 beats a missing dense rank.
    fused = rrf_fuse({"dense": ["z"], "other": ["a"]})
    assert [r["key"] for r in fused] == ["z", "a"]


def test_fully_equal_tie_falls_back_to_key():
    fused = rrf_fuse({"x": ["z"], "y": ["a"]})
    assert [r["key"] for r in fused] == ["a", "z"]


def test_non_tied_ordering_unchanged():
    fused = rrf_fuse({"dense": ["a", "b", "c"], "sparse": ["b", "d"]})
    assert [r["key"] for r in fused] == ["b", "a", "d", "c"]
    scores = [r["score"] for r in fused]
    assert all(x > y for x, y in zip(scores, scores[1:]))


def test_empty_input_and_invalid_k():
    assert rrf_fuse({}) == []
    assert rrf_fuse({"dense": []}) == []
    with pytest.raises(ValueError):
        rrf_fuse({"dense": ["a"]}, k=0)
