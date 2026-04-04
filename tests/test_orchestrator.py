from pathlib import Path

from gemma_ra.agent.orchestrator import ResearchAgent, RunRequest
from gemma_ra.core.exceptions import SourceError
from gemma_ra.core.config import AppConfig
from gemma_ra.core.schemas import PaperDocument, PaperMetadata, TaskType


class FakeLocalSource:
    def discover(self, papers_dir: Path) -> list[Path]:
        return [papers_dir / "paper.pdf"]

    def read_many(self, paths: list[Path]) -> list:
        return [
            PaperDocument(
                metadata=PaperMetadata(
                    paper_id="paper-1",
                    title="Paper",
                    authors=["Jane Doe"],
                    source="local",
                ),
                content="Paper content.",
                sections=[],
            )
        ]


class FakeArxivSource:
    def search_and_load(self, professors: list[str], topic: str | None) -> list:
        return []


class FakeAnalysisEngine:
    def run(self, task, task_spec, context):
        return type(
            "FakeResult",
            (),
            {
                "title": "review",
                "task": task,
                "content": {
                    "field_overview": "overview",
                    "key_problems": ["problem"],
                    "paper_summaries": [
                        {
                            "paper": {
                                "paper_id": "paper-1",
                                "title": "Paper",
                                "authors": ["Jane Doe"],
                                "abstract": None,
                                "published": None,
                                "updated": None,
                                "pdf_url": None,
                                "local_path": None,
                                "source": "local",
                            },
                            "problem": "problem",
                            "method": "method",
                            "contributions": ["contribution"],
                            "limitations": ["limitation"],
                        }
                    ],
                    "themes": ["theme"],
                    "methodological_trends": ["trend"],
                    "gaps": ["gap"],
                    "disagreements": ["none"],
                    "synthesis": "summary",
                },
            },
        )()


class PrefetchingArxivSource:
    def __init__(self) -> None:
        self.fetch_calls: list[str] = []

    def search_and_load(self, professors: list[str], topic: str | None) -> tuple[list[PaperDocument], list[str]]:
        return (
            [
                PaperDocument(
                    metadata=PaperMetadata(
                        paper_id="arxiv-1",
                        title="Remote Paper",
                        authors=["Jane Doe"],
                        pdf_url="https://arxiv.org/pdf/arxiv-1",
                        source="arxiv",
                    ),
                    content="Abstract only.",
                    sections=[],
                )
            ],
            ['arXiv search "author-only" returned 1 result(s).'],
        )

    def fetch_pdf_document(self, metadata: PaperMetadata, download_dir: Path) -> PaperDocument:
        self.fetch_calls.append(metadata.paper_id)
        return PaperDocument(
            metadata=PaperMetadata(
                paper_id=metadata.paper_id,
                title=metadata.title,
                authors=metadata.authors,
                pdf_url=metadata.pdf_url,
                local_path=download_dir / f"{metadata.paper_id}.pdf",
                source="arxiv_pdf",
            ),
            content="Full remote paper text.",
            sections=[],
        )


def test_orchestrator_writes_artifact(tmp_path: Path) -> None:
    config = AppConfig(output_dir=tmp_path)
    agent = ResearchAgent(config)
    agent.local_source = FakeLocalSource()
    agent.arxiv_source = FakeArxivSource()
    agent.analysis_engine = FakeAnalysisEngine()

    artifact = agent.run(
        RunRequest(
            task=TaskType.REVIEW_TOPIC,
            topic="topic",
            professors=[],
            papers_dir=tmp_path,
            paper_paths=[],
            output_dir=tmp_path,
        )
    )

    assert artifact.markdown_path.exists()
    assert artifact.json_path.exists()


def test_orchestrator_prefetches_arxiv_full_text_when_local_papers_missing(tmp_path: Path) -> None:
    config = AppConfig(output_dir=tmp_path)
    agent = ResearchAgent(config)

    class MissingLocalSource:
        def discover(self, papers_dir: Path) -> list[Path]:
            raise SourceError(f"Papers directory does not exist: {papers_dir}")

        def read_many(self, paths: list[Path]) -> list[PaperDocument]:
            return []

    arxiv_source = PrefetchingArxivSource()
    agent.local_source = MissingLocalSource()
    agent.arxiv_source = arxiv_source

    context = agent._build_context(
        RunRequest(
            task=TaskType.REVIEW_TOPIC,
            topic="graphics",
            professors=["Ali Mahdavi Amiri"],
            papers_dir=tmp_path / "missing",
            paper_paths=[],
            output_dir=tmp_path,
        )
    )

    assert arxiv_source.fetch_calls == ["arxiv-1"]
    assert context.papers[0].metadata.source == "arxiv_pdf"
    assert any("Fetched full PDF text" in note for note in context.discovery_notes)


def test_orchestrator_prefetches_arxiv_full_text_for_find_papers(tmp_path: Path) -> None:
    config = AppConfig(output_dir=tmp_path)
    agent = ResearchAgent(config)

    class MissingLocalSource:
        def discover(self, papers_dir: Path) -> list[Path]:
            raise SourceError(f"Papers directory does not exist: {papers_dir}")

        def read_many(self, paths: list[Path]) -> list[PaperDocument]:
            return []

    arxiv_source = PrefetchingArxivSource()
    agent.local_source = MissingLocalSource()
    agent.arxiv_source = arxiv_source

    context = agent._build_context(
        RunRequest(
            task=TaskType.FIND_PAPERS,
            topic="graphics",
            professors=["Ali Mahdavi Amiri"],
            papers_dir=tmp_path / "missing",
            paper_paths=[],
            output_dir=tmp_path,
        )
    )

    assert arxiv_source.fetch_calls == ["arxiv-1"]
    assert context.papers[0].metadata.source == "arxiv_pdf"


def test_orchestrator_prefetches_arxiv_full_text_even_when_local_papers_exist(tmp_path: Path) -> None:
    config = AppConfig(output_dir=tmp_path)
    agent = ResearchAgent(config)
    agent.local_source = FakeLocalSource()
    arxiv_source = PrefetchingArxivSource()
    agent.arxiv_source = arxiv_source

    context = agent._build_context(
        RunRequest(
            task=TaskType.REVIEW_TOPIC,
            topic="graphics",
            professors=["Ali Mahdavi Amiri"],
            papers_dir=tmp_path,
            paper_paths=[],
            output_dir=tmp_path,
        )
    )

    assert arxiv_source.fetch_calls == ["arxiv-1"]
    assert any(paper.metadata.source == "local" for paper in context.papers)
    assert any(paper.metadata.source == "arxiv_pdf" for paper in context.papers)


def test_orchestrator_allows_empty_map_research_context_when_no_papers_found(tmp_path: Path) -> None:
    config = AppConfig(output_dir=tmp_path)
    agent = ResearchAgent(config)

    class MissingLocalSource:
        def discover(self, papers_dir: Path) -> list[Path]:
            raise SourceError(f"Papers directory does not exist: {papers_dir}")

        def read_many(self, paths: list[Path]) -> list[PaperDocument]:
            return []

    class EmptyArxivSource:
        def search_and_load(self, professors: list[str], topic: str | None) -> tuple[list[PaperDocument], list[str]]:
            return [], ['No arXiv papers matched professor "Daniel CohenOr" with topic "computer vision".']

    agent.local_source = MissingLocalSource()
    agent.arxiv_source = EmptyArxivSource()

    context = agent._build_context(
        RunRequest(
            task=TaskType.MAP_RESEARCH_OPPORTUNITIES,
            topic="computer vision",
            professors=["Daniel CohenOr"],
            papers_dir=tmp_path / "missing",
            paper_paths=[],
            output_dir=tmp_path,
        )
    )

    assert context.papers == []
    assert any("No arXiv papers matched" in note for note in context.discovery_notes)
