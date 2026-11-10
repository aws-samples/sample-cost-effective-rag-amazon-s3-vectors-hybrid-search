"""Dependency-free text utilities shared by indexing and search.

The SAME tokenization pipeline must be used when building the BM25 index and
when scoring queries - otherwise terms never match. Keep changes backward
compatible or plan a re-index.

Tokens are lowercase runs of ``[a-z0-9]``. Stopwords and single letters are
dropped, but single digits are kept, so identifiers that differ only in a
one-digit segment stay distinct: ``ZX-ALPHA-9`` -> ``zx alpha 9`` and
``ZX-ALPHA-8`` -> ``zx alpha 8``. Multi-character tokens are unaffected
(``ERR-4471`` -> ``err 4471``).

normalize_terms() also adds each whole hyphenated identifier that contains a
digit as one extra, unstemmed term (``ZX-ALPHA-9`` -> ``zx alpha 9
zx-alpha-9``), so an exact identifier outranks a document that only shares
its parts. Hyphenated words without a digit (``e-mail``) get no extra term.
"""
import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Maximal hyphen-joined runs, matched on the lowercased text; normalize_terms()
# keeps only the runs that contain a hyphen and a digit. The optional group
# (``*``) keeps matching linear in the input length. Requiring a hyphen in the
# pattern (``+``) backtracks quadratically on long hyphen-free runs, and the
# search path passes the raw, uncapped query here.
_IDENTIFIER_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# Compact English stopword list (high-frequency function words only).
STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i if in is it its
    me my no nor not of on or our she so that the their them then there these they
    this to us was we were what when where which who will with you your""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; stopwords and single letters dropped, single digits kept."""
    # Single digits are kept so that identifiers such as ZX-ALPHA-8 and
    # ZX-ALPHA-9 don't collapse to the same terms. Existing deployments must
    # re-index (re-upload their documents) after this tokenizer change: chunks
    # indexed before it have no postings for single-digit terms.
    return [
        t
        for t in _TOKEN_RE.findall(text.lower())
        if (len(t) > 1 or t.isdigit()) and t not in STOPWORDS
    ]


def light_stem(term: str) -> str:
    """Very conservative suffix stripping (plural/gerund/past).

    Deliberately avoids a bare ``-es`` rule: it breaks singular/plural
    consistency for words ending in -e (databases -> databas vs database).
    """
    if len(term) > 5 and term.endswith("ing"):
        return term[:-3]
    if len(term) > 4 and term.endswith("ed"):
        return term[:-2]
    if len(term) > 4 and term.endswith("sses"):
        return term[:-2]
    if term.endswith("ss"):
        return term
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term


def normalize_terms(text: str) -> list[str]:
    """Tokenize + stem, plus whole identifiers: the canonical term pipeline for BM25 index and query."""
    # Whole identifiers keep ZX-ALPHA-9 ahead of a near-miss that shares its
    # parts (ZX-ALPHA-8 and ZX-BETA-9 in one document). Like the single-digit
    # tokens, they need a re-index: chunks indexed before this change have no
    # postings for identifier terms.
    identifiers = [
        m
        for m in _IDENTIFIER_RE.findall(text.lower())
        if "-" in m and any(c.isdigit() for c in m)
    ]
    return [light_stem(t) for t in tokenize(text)] + identifiers


def estimate_tokens(text: str) -> int:
    """Approximate LLM token count (~4 characters/token for English prose).

    Exact tokenizers for Claude are not distributable inside Lambda; the
    4-chars heuristic is the standard budget approximation and errs slightly
    high for code-heavy text, which is the safe direction for budgeting.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))
