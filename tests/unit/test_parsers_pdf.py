"""PDF path of ingestion.parsers.extract_text, exercised through pypdf."""
import io

import pypdf  # pinned in requirements-dev.txt; fail loudly rather than skip

from ingestion.parsers import extract_text

PAGE_TOKENS = ("PAGE-ONE-TOKEN", "PAGE-TWO-TOKEN", "PAGE-THREE-TOKEN")


def _build_pdf(page_texts) -> bytes:
    """Minimal valid PDF: one Helvetica text content stream per page."""
    page_count = len(page_texts)
    font_num = 3
    first_page_num = 4
    page_nums = [first_page_num + 2 * i for i in range(page_count)]
    kids = " ".join(f"{n} 0 R" for n in page_nums)

    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode(),
        font_num: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for page_num, text in zip(page_nums, page_texts):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects[page_num] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> "
            f"/Contents {page_num + 1} 0 R >>"
        ).encode()
        objects[page_num + 1] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = {}
    for num in sorted(objects):
        offsets[num] = out.tell()
        out.write(f"{num} 0 obj\n".encode() + objects[num] + b"\nendobj\n")
    xref_offset = out.tell()
    size = max(objects) + 1
    out.write(f"xref\n0 {size}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for num in range(1, size):
        out.write(f"{offsets[num]:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return out.getvalue()


def test_pdf_fixture_is_a_three_page_pdf():
    reader = pypdf.PdfReader(io.BytesIO(_build_pdf(PAGE_TOKENS)))
    assert len(reader.pages) == 3


def test_extract_text_pdf_returns_every_page():
    text = extract_text(_build_pdf(PAGE_TOKENS), "pdf")
    for token in PAGE_TOKENS:
        assert token in text
    # Pages are joined in order.
    positions = [text.index(token) for token in PAGE_TOKENS]
    assert positions == sorted(positions)


def test_extract_text_pdf_format_is_case_insensitive():
    assert "PAGE-ONE-TOKEN" in extract_text(_build_pdf(PAGE_TOKENS[:1]), "PDF")
