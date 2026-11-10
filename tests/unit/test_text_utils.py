from common.text_utils import estimate_tokens, light_stem, normalize_terms, tokenize


def test_tokenize_lowercases_and_removes_stopwords():
    tokens = tokenize("The Quick BROWN fox and the lazy dog!")
    assert "the" not in tokens and "and" not in tokens
    assert "quick" in tokens and "brown" in tokens and "fox" in tokens


def test_tokenize_drops_single_letters_and_punctuation():
    assert tokenize("a b c d2 e-mail") == ["d2", "mail"]


def test_tokenize_keeps_single_digits_but_not_single_letters():
    assert tokenize("Plan B, step 2 of 3") == ["plan", "step", "2", "3"]


def test_identifiers_differing_in_one_digit_stay_distinct():
    # normalize_terms() is shared by the BM25 indexer and the sparse search,
    # so this holds at index time and at query time.
    assert normalize_terms("ZX-ALPHA-9") == ["zx", "alpha", "9", "zx-alpha-9"]
    assert set(normalize_terms("ZX-ALPHA-9")) != set(normalize_terms("ZX-ALPHA-8"))
    # Only hyphenated identifiers that contain a digit become whole terms.
    assert normalize_terms("e-mail") == ["mail"]
    # Linear time on a long hyphen-free run: the search path passes the raw,
    # uncapped query here (a backtracking pattern takes about a minute).
    assert normalize_terms("a" * 100_000) == ["a" * 100_000]


def test_multi_digit_identifier_tokens_unchanged():
    assert tokenize("ERR-4471") == ["err", "4471"]


def test_light_stem_is_conservative():
    assert light_stem("caches") == "cache"  # plural -s stripped, stem intact
    assert light_stem("databases") == "database"
    assert light_stem("classes") == "class"  # -sses rule
    assert light_stem("cats") == "cat"
    assert light_stem("boss") == "boss"  # -ss protected
    assert light_stem("running") == "runn"
    assert light_stem("used") == "used"  # too short for -ed rule
    assert light_stem("go") == "go"


def test_normalize_terms_pipeline_is_consistent():
    # Index-time and query-time must produce matching terms.
    assert set(normalize_terms("Databases")) & set(normalize_terms("database"))


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("x" * 4001) == 1001
