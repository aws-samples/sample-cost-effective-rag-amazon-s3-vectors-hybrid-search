import pytest

from ingestion.chunker import chunk_text


def test_empty_and_whitespace_input():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_single_chunk():
    text = "A short paragraph that easily fits."
    assert chunk_text(text) == [text]


def test_chunks_respect_size_limit():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(30))
    chunks = chunk_text(text, chunk_chars=500, overlap_chars=50)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    assert all(c.strip() for c in chunks)


def test_consecutive_chunks_overlap():
    text = " ".join(f"w{i:04d}" for i in range(2000))
    chunks = chunk_text(text, chunk_chars=200, overlap_chars=50)
    assert len(chunks) > 2
    for left, right in zip(chunks, chunks[1:]):
        tail_words = set(left.split()[-3:])
        head_words = set(right.split()[:15])
        assert tail_words & head_words, "expected overlap words shared across boundary"


def test_prefers_paragraph_boundaries():
    para_a = "alpha " * 60
    para_b = "bravo " * 60
    chunks = chunk_text(f"{para_a.strip()}\n\n{para_b.strip()}", chunk_chars=400, overlap_chars=0)
    # With overlap disabled, no chunk should mix both paragraphs' vocabulary.
    assert all(not ("alpha" in c and "bravo" in c) for c in chunks)


def test_unbroken_token_falls_back_to_windows():
    text = "x" * 5000
    chunks = chunk_text(text, chunk_chars=1000, overlap_chars=100)
    assert all(len(c) <= 1000 for c in chunks)
    assert sum(len(c) for c in chunks) >= 5000  # windows + overlap cover everything


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_chars=0)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_chars=100, overlap_chars=100)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_chars=100, overlap_chars=-1)
