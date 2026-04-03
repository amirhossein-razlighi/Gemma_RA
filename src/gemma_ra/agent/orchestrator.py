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
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.tool_registry = ToolRegistry.default()
        self.local_source = LocalPaperSource()
        self.arxiv_source = ArxivPaperSource(config.arxiv)
        self.analysis_engine = AnalysisEngine(
            OllamaClient(config.ollama),
            arxiv_search=self.arxiv_source.search_and_load,
            workspace=WorkspaceExecutor(config.executor),
            max_iterations=config.executor.max_iterations,
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
        if request.papers_dir:
            local_paths.extend(self.local_source.discover(request.papers_dir))

        if local_paths:
            papers.extend(self.local_source.read_many(local_paths))

        if request.professors:
            arxiv_papers, notes = self.arxiv_source.search_and_load(
                professors=request.professors,
                topic=request.topic,
            )
            papers.extend(arxiv_papers)
        else:
            notes = []

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
