from pathlib import Path

from gemma_ra.agent.orchestrator import ResearchAgent, RunRequest
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
                    "themes": ["theme"],
                    "gaps": ["gap"],
                    "disagreements": ["none"],
                    "synthesis": "summary",
                },
            },
        )()


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
