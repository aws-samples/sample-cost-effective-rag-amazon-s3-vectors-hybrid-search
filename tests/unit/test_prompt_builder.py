from query.prompt_builder import SYSTEM_PROMPT, build_context_block, build_messages


def _ctx(i: int, text: str) -> dict:
    return {"key": f"t#d#{i:05d}", "text": text, "metadata": {"source": f"docs/file{i}.md"}}


def test_no_assistant_prefill():
    """Claude 4.6 removed prefill support (hard 400); only a user turn is allowed."""
    system, messages, _ = build_messages("What is RAG?", [_ctx(1, "RAG combines retrieval with generation.")], 6000)
    assert system == SYSTEM_PROMPT
    assert [m["role"] for m in messages] == ["user"]


def test_contexts_numbered_with_sources():
    _, messages, included = build_messages(
        "q?", [_ctx(1, "first passage"), _ctx(2, "second passage")], 6000
    )
    text = messages[0]["content"][0]["text"]
    assert "[1] (source: docs/file1.md)" in text
    assert "[2] (source: docs/file2.md)" in text
    assert "Question: q?" in text
    assert len(included) == 2


def test_budget_limits_included_contexts():
    contexts = [_ctx(i, "word " * 400) for i in range(10)]  # ~500 tokens each
    block, included = build_context_block(contexts, max_context_tokens=1400)
    # ~1100-token budget after the question reserve -> roughly two passages.
    assert 1 <= len(included) <= 3
    assert len(included) < 10
    assert block


def test_empty_texts_are_skipped():
    contexts = [_ctx(1, ""), _ctx(2, "real content")]
    block, included = build_context_block(contexts, max_context_tokens=6000)
    assert len(included) == 1
    assert "real content" in block
    assert "[1] (source: docs/file2.md)" in block  # numbering follows inclusion order


def test_citation_numbering_matches_included_order():
    contexts = [_ctx(i, f"passage number {i}") for i in range(1, 4)]
    _, messages, included = build_messages("q", contexts, 6000)
    text = messages[0]["content"][0]["text"]
    for position, ctx in enumerate(included, start=1):
        assert f"[{position}] (source: {ctx['metadata']['source']})" in text
