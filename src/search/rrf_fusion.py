"""Reciprocal Rank Fusion (RRF).

RRF merges ranked lists using only rank positions - no score calibration
needed between heterogeneous scorers (cosine distance vs BM25):

    rrf_score(d) = sum_over_lists  w_list * 1 / (k + rank_list(d))

k=60 is the value from the original RRF paper (Cormack et al.) and remains
the standard default. Larger k flattens the contribution of top ranks.

Equal fused scores (for example, when dense and sparse rank two documents in
swapped order, so both score 1/61 + 1/62) are ordered by better sparse rank,
then better dense rank, then key. An exact keyword match (an SKU or error
code) therefore beats a near-miss whenever BM25 ranks the exact match higher.
Scores are compared with a 1e-12 relative tolerance so float noise doesn't
defeat the tie-break.
"""
import functools
import math
from collections import defaultdict

DEFAULT_RRF_K = 60

# Lists consulted, in order, to break ties between equal fused scores.
TIE_BREAK_LISTS = ("sparse", "dense")
SCORE_REL_TOL = 1e-12


def rrf_fuse(
    ranked_lists: dict[str, list[str]],
    k: int = DEFAULT_RRF_K,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Fuse named ranked lists of document keys.

    Args:
        ranked_lists: {"dense": [key1, key2, ...], "sparse": [...]}, each list
            ordered best-first.
        k: RRF constant (> 0).
        weights: optional per-list weight, default 1.0 each.

    Returns:
        [{"key", "score", "ranks": {list_name: rank}}] ordered by fused score.
        Ties go to the lower sparse rank, then the lower dense rank (a missing
        rank counts as infinity), then key order.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    weights = weights or {}

    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for name, keys in ranked_lists.items():
        w = float(weights.get(name, 1.0))
        for position, key in enumerate(keys, start=1):
            if key in ranks and name in ranks[key]:
                continue  # ignore duplicate occurrences within one list
            scores[key] += w / (k + position)
            ranks[key][name] = position

    def tie_break(key: str) -> tuple:
        return tuple(ranks[key].get(name, math.inf) for name in TIE_BREAK_LISTS) + (key,)

    def compare(a: tuple[str, float], b: tuple[str, float]) -> int:
        (key_a, score_a), (key_b, score_b) = a, b
        if not math.isclose(score_a, score_b, rel_tol=SCORE_REL_TOL):
            return -1 if score_a > score_b else 1
        tb_a, tb_b = tie_break(key_a), tie_break(key_b)
        return (tb_a > tb_b) - (tb_a < tb_b)

    fused = sorted(scores.items(), key=functools.cmp_to_key(compare))
    return [{"key": key, "score": score, "ranks": dict(ranks[key])} for key, score in fused]
