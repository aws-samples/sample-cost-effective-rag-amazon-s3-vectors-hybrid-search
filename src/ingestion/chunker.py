"""Recursive character splitting with overlap.

Targets ~512 tokens per chunk using the ~4 chars/token heuristic:
    512 tokens  ~= 2048 characters (chunk)
     64 tokens  ~=  256 characters (overlap)

Splitting prefers natural boundaries in order: paragraph, line, sentence,
word - falling back to hard character windows only for pathological input
(e.g. one enormous unbroken token).
"""
DEFAULT_CHUNK_CHARS = 2048
DEFAULT_OVERLAP_CHARS = 256

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _recursive_split(text: str, size: int, separators: list[str]) -> list[str]:
    """Split ``text`` into pieces each <= ``size``, preferring earlier separators."""
    if len(text) <= size:
        return [text] if text else []
    if not separators:
        # Hard fallback: fixed character windows.
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    if len(parts) == 1:  # separator absent; try the next one
        return _recursive_split(text, size, rest)

    units: list[str] = []
    for i, part in enumerate(parts):
        unit = part + (sep if i < len(parts) - 1 else "")
        if not unit:
            continue
        if len(unit) > size:
            units.extend(_recursive_split(unit, size, rest))
        else:
            units.append(unit)
    return units


def chunk_text(
    text: str,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Split ``text`` into overlapping chunks of at most ``chunk_chars`` chars."""
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if not 0 <= overlap_chars < chunk_chars:
        raise ValueError("overlap_chars must satisfy 0 <= overlap < chunk_chars")

    text = (text or "").strip()
    if not text:
        return []

    units = _recursive_split(text, chunk_chars, SEPARATORS)

    chunks: list[str] = []
    current = ""
    for unit in units:
        if len(current) + len(unit) <= chunk_chars:
            current += unit
            continue
        if current.strip():
            chunks.append(current.strip())
        # Seed the next chunk with the tail of the previous one (overlap).
        tail = current[-overlap_chars:] if overlap_chars else ""
        current = tail + unit
        # Guard: unit near chunk size can push the seeded buffer over the cap.
        while len(current) > chunk_chars:
            chunks.append(current[:chunk_chars].strip())
            step = chunk_chars - overlap_chars
            current = current[step:]
    if current.strip():
        chunks.append(current.strip())
    return chunks
