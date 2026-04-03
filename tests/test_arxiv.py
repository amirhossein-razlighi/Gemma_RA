from pathlib import Path

from gemma_ra.sources.arxiv import ArxivPaperSource
from gemma_ra.core.config import ArxivConfig
from gemma_ra.core.schemas import PaperMetadata


def test_parse_feed_extracts_basic_metadata() -> None:
    xml = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/1234.5678v1</id>
        <updated>2026-01-01T00:00:00Z</updated>
        <published>2025-12-31T00:00:00Z</published>
        <title>Sample Paper</title>
        <summary>Short abstract.</summary>
        <author><name>Jane Doe</name></author>
        <link title="pdf" href="http://arxiv.org/pdf/1234.5678v1"/>
      </entry>
    </feed>
    """
    source = ArxivPaperSource(ArxivConfig())

    docs = source._parse_feed(xml)

    assert len(docs) == 1
    assert docs[0].metadata.paper_id == "1234.5678v1"
    assert docs[0].metadata.authors == ["Jane Doe"]
    assert docs[0].content == "Short abstract."


def test_search_with_fallbacks_uses_broader_queries_when_needed() -> None:
    source = ArxivPaperSource(ArxivConfig())
    calls: list[str] = []

    def fake_search_query(query: str):
        calls.append(query)
        if query == 'au:"amiri"':
            return source._parse_feed(
                """
                <feed xmlns="http://www.w3.org/2005/Atom">
                  <entry>
                    <id>http://arxiv.org/abs/9999.0001v1</id>
                    <title>Recovered Paper</title>
                    <summary>Abstract.</summary>
                    <author><name>Ali Mahdavi Amiri</name></author>
                  </entry>
                </feed>
                """
            )
        return []

    source.search_query = fake_search_query  # type: ignore[method-assign]

    docs, notes = source.search_with_fallbacks("Ali mahdavi amiri", "generative ai")

    assert docs[0].metadata.title == "Recovered Paper"
    assert any('surname-only' in note for note in notes)
    assert calls[:4] == [
        'au:"Ali mahdavi amiri" AND all:"generative ai"',
        'au:"Ali mahdavi amiri"',
        'au:"amiri" AND all:"generative ai"',
        'au:"amiri"',
    ]


def test_fetch_pdf_document_downloads_and_reads_full_text(tmp_path: Path) -> None:
    source = ArxivPaperSource(ArxivConfig())
    metadata = PaperMetadata(
        paper_id="1234.5678v1",
        title="Sample Paper",
        authors=["Jane Doe"],
        pdf_url="https://arxiv.org/pdf/1234.5678v1",
        source="arxiv",
    )

    class FakeResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: float, follow_redirects: bool):
        assert "1234.5678v1" in url
        return FakeResponse(b"%PDF-1.4 fake")

    def fake_read(path: Path):
        return type(
            "Doc",
            (),
            {
                "metadata": metadata,
                "content": "Full paper text",
                "sections": [],
            },
        )()

    source.local_source.read = fake_read  # type: ignore[method-assign]

    import gemma_ra.sources.arxiv as arxiv_module

    original_get = arxiv_module.httpx.get
    arxiv_module.httpx.get = fake_get  # type: ignore[assignment]
    try:
        document = source.fetch_pdf_document(metadata, tmp_path)
    finally:
        arxiv_module.httpx.get = original_get  # type: ignore[assignment]

    assert document.content == "Full paper text"
    assert document.metadata.source == "arxiv_pdf"
    assert document.metadata.local_path == tmp_path / "1234.5678v1.pdf"
