"""Prompt assembly with context-window budgeting.

Compatibility note (verified): Claude Sonnet 4.6 and later reject a prefilled
final assistant turn with a 400 error, so this builder emits only a system
prompt and one user message (works for Sonnet 4.6 and Haiku 4.5).
"""
from common.text_utils import estimate_tokens

SYSTEM_PROMPT = (
    "You are a precise retrieval-augmented assistant. Answer the user's question "
    "using ONLY the numbered context passages provided. Cite the passages you used "
    "with bracketed numbers, e.g. [1] or [2][3]. If the context does not contain "
    "the answer, say you don't know - never invent information."
)

# Reserve room inside the user message for the question and instructions.
_QUESTION_RESERVE_TOKENS = 300


def build_context_block(contexts: list[dict], max_context_tokens: int) -> tuple[str, list[dict]]:
    """Pack retrieved chunks into a numbered block within the token budget.

    Returns the formatted block and the list of contexts actually included
    (so citations map 1:1 to what the model saw).
    """
    included: list[dict] = []
    parts: list[str] = []
    budget = max(max_context_tokens - _QUESTION_RESERVE_TOKENS, 200)
    used = 0

    for ctx in contexts:
        text = (ctx.get("text") or "").strip()
        if not text:
            continue
        source = (ctx.get("metadata") or {}).get("source", "unknown")
        entry = f"[{len(included) + 1}] (source: {source})\n{text}"
        cost = estimate_tokens(entry)

        if used + cost > budget:
            remaining = budget - used
            # Only truncate if a meaningful amount of the passage still fits.
            if remaining >= 100:
                entry = entry[: remaining * 4].rstrip() + " ..."
                parts.append(entry)
                included.append(ctx)
            break

        parts.append(entry)
        included.append(ctx)
        used += cost

    return "\n\n".join(parts), included


def build_messages(question: str, contexts: list[dict], max_context_tokens: int) -> tuple[str, list[dict], list[dict]]:
    """Return (system_prompt, converse_messages, included_contexts)."""
    context_block, included = build_context_block(contexts, max_context_tokens)
    user_text = (
        f"Context passages:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above and cite passages "
        "with bracketed numbers."
    )
    messages = [{"role": "user", "content": [{"text": user_text}]}]
    return SYSTEM_PROMPT, messages, included
