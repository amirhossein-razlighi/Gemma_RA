from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gemma_ra.analysis.engine import AnalysisEngine
from gemma_ra.analysis.renderers import render_analysis_result
from gemma_ra.agent.tools import ToolRegistry
from gemma_ra.core.artifacts import write_artifact
from gemma_ra.core.config import AppConfig
from gemma_ra.core.exceptions import SourceError
from gemma_ra.core.model_client import OllamaClient
from gemma_ra.core.schemas import ArtifactRecord, ResearchContext, TaskType
from gemma_ra.core.tasks import get_task_spec
from gemma_ra.core.workspace import WorkspaceExecutor
from gemma_ra.sources.arxiv import ArxivPaperSource
from gemma_ra.sources.local_papers import LocalPaperSource


@dataclass(slots=True)
class RunRequest:
    task: TaskType
    topic: str | None
    professors: list[str]
    papers_dir: Path | None
    paper_paths: list[Path]
    output_dir: Path | None
    instructions: str | None = None
    instructions_path: Path | None = None


class ResearchAgent:
    def __init__(self, config: AppConfig, reporter=None, stream_chat: bool = False, interactive_guidance=None) -> None:
        self.config = config
        self.tool_registry = ToolRegistry.default()
        self.local_source = LocalPaperSource()
        self.arxiv_source = ArxivPaperSource(config.arxiv)
        self.workspace = WorkspaceExecutor(config.executor)
        self.analysis_engine = AnalysisEngine(
            OllamaClient(config.ollama),
            arxiv_search=self.arxiv_source.search_and_load,
            arxiv_fetch=self.arxiv_source.fetch_pdf_document,
            workspace=self.workspace,
            max_iterations=config.executor.max_iterations,
            reporter=reporter,
            stream_chat=stream_chat,
            interactive_guidance=interactive_guidance,
        )

    def run(self, request: RunRequest) -> ArtifactRecord:
        task_spec = get_task_spec(request.task)
        self.tool_registry.validate_allowed(task_spec.allowed_tools)
        context = self._build_context(request)
        result = self.analysis_engine.run(task=request.task, task_spec=task_spec, context=context)
        markdown = render_analysis_result(result)
        output_dir = request.output_dir or self.config.output_dir
        return write_artifact(
            output_dir=output_dir,
            task=request.task,
            task_name=request.task.value,
            title=result.title,
            markdown=markdown,
            payload={
                "title": result.title,
                "task": result.task.value,
                "content": result.content,
            },
        )

    def _build_context(self, request: RunRequest) -> ResearchContext:
        papers = []
        local_paths = list(request.paper_paths)
        notes: list[str] = []
        if request.papers_dir:
            try:
                local_paths.extend(self.local_source.discover(request.papers_dir))
            except SourceError as exc:
                if request.professors:
                    notes.append(str(exc))
                else:
                    raise

        if local_paths:
            papers.extend(self.local_source.read_many(local_paths))

        local_papers_found = bool(papers)
        if request.professors:
            arxiv_papers, arxiv_notes = self.arxiv_source.search_and_load(
                professors=request.professors,
                topic=request.topic,
            )
            notes.extend(arxiv_notes)
            if self._should_prefetch_arxiv_full_text(request=request, local_papers_found=local_papers_found):
                fetched_papers: list = []
                download_dir = self.workspace.resolve_path(".") / "arxiv_papers"
                for paper in arxiv_papers[: min(3, self.config.arxiv.max_results)]:
                    try:
                        fetched = self.arxiv_source.fetch_pdf_document(paper.metadata, download_dir)
                    except SourceError as exc:
                        notes.append(f'Failed to fetch full text for "{paper.metadata.title}": {exc}')
                        fetched_papers.append(paper)
                    else:
                        notes.append(f'Fetched full PDF text for "{fetched.metadata.title}" during context build.')
                        fetched_papers.append(fetched)
                arxiv_papers = fetched_papers
            papers.extend(arxiv_papers)

        if request.task not in {TaskType.FIND_PAPERS, TaskType.RUN_INSTRUCTIONS} and not papers:
            raise SourceError("No papers were found. Provide local PDFs or professor names.")

        return ResearchContext(
            task=request.task,
            topic=request.topic,
            professors=request.professors,
            local_paper_paths=local_paths,
            papers=papers,
            discovery_notes=notes,
            instructions=request.instructions,
            instructions_path=request.instructions_path,
        )

    @staticmethod
    def _should_prefetch_arxiv_full_text(request: RunRequest, local_papers_found: bool) -> bool:
        return (
            not local_papers_found
            and bool(request.professors)
            and request.task
            in {
                TaskType.ANALYZE_PAPER,
                TaskType.REVIEW_TOPIC,
                TaskType.GENERATE_IDEAS,
                TaskType.SUGGEST_EXPERIMENTS,
                TaskType.MAP_RESEARCH_OPPORTUNITIES,
            }
        )

    def shutdown(self) -> None:
        self.workspace.terminate_all()
