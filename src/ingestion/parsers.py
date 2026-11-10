"""Text extraction for supported input formats: PDF, TXT, MD, HTML."""
import io
import re
from html.parser import HTMLParser

SUPPORTED_FORMATS = frozenset({"pdf", "txt", "md", "markdown", "html", "htm"})

_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


class _HtmlTextExtractor(HTMLParser):
    """Collect visible text, skipping script/style, inserting paragraph breaks."""

    _SKIP = {"script", "style", "noscript", "template"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def _clean(text: str) -> str:
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def format_from_key(key: str) -> str | None:
    """Map an S3 object key to a supported format identifier (or None)."""
    ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
    return ext if ext in SUPPORTED_FORMATS else None


def extract_text(data: bytes, fmt: str) -> str:
    """Extract plain text from raw document bytes.

    Raises ValueError for unsupported formats.
    """
    fmt = fmt.lower()
    if fmt in ("txt", "md", "markdown"):
        return _clean(data.decode("utf-8", errors="replace"))

    if fmt in ("html", "htm"):
        parser = _HtmlTextExtractor()
        parser.feed(data.decode("utf-8", errors="replace"))
        return _clean(parser.text())

    if fmt == "pdf":
        # Imported lazily so environments without pypdf can still use the
        # text/HTML paths (and unit tests avoid the dependency).
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return _clean("\n\n".join(pages))

    raise ValueError(f"Unsupported format: {fmt}. Supported: {sorted(SUPPORTED_FORMATS)}")
