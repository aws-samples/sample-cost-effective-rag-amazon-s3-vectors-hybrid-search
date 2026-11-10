from ingestion.handler import doc_id_from_key, tenant_from_key, vector_key
from ingestion.parsers import extract_text, format_from_key


def test_tenant_from_key_convention():
    assert tenant_from_key("tenants/anyexamplecompany/reports/q1.pdf") == "anyexamplecompany"
    assert tenant_from_key("tenants/ANYEXAMPLECOMPANY/reports/q1.pdf") == "anyexamplecompany"  # lowercased
    assert tenant_from_key("reports/q1.pdf") == "default"
    assert tenant_from_key("tenants/bad tenant!/x.pdf") is None  # rejected, not default
    assert tenant_from_key("tenants//x.pdf") is None  # empty folder name
    assert tenant_from_key("tenants/orphan.pdf") == "default"  # no trailing path


def test_doc_id_is_stable_and_readable():
    a = doc_id_from_key("tenants/anyexamplecompany/Annual Report 2026.pdf")
    b = doc_id_from_key("tenants/anyexamplecompany/Annual Report 2026.pdf")
    assert a == b
    assert a.startswith("annual-report-2026-")
    assert doc_id_from_key("other/Annual Report 2026.pdf") != a  # path-sensitive


def test_vector_key_format():
    assert vector_key("anyexamplecompany", "doc-1", 7) == "anyexamplecompany#doc-1#00007"


def test_format_from_key():
    assert format_from_key("a/b/report.PDF") == "pdf"
    assert format_from_key("notes.md") == "md"
    assert format_from_key("page.htm") == "htm"
    assert format_from_key("binary.exe") is None
    assert format_from_key("no-extension") is None


def test_extract_text_plain_and_markdown():
    assert extract_text(b"hello  world", "txt") == "hello world"
    assert "# Title" in extract_text(b"# Title\n\nBody text.", "md")


def test_extract_text_html_strips_tags_and_scripts():
    html = b"""<html><head><style>.x{color:red}</style>
    <script>alert('nope')</script></head>
    <body><h1>Heading</h1><p>First para.</p><p>Second para.</p></body></html>"""
    text = extract_text(html, "html")
    assert "Heading" in text and "First para." in text
    assert "alert" not in text and "color:red" not in text
