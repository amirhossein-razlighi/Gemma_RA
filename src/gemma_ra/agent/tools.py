from __future__ import annotations

from dataclasses import dataclass

from gemma_ra.core.schemas import ToolKind, ToolSpec


@dataclass(slots=True)
class ToolRegistry:
    tools: dict[str, ToolSpec]

    @classmethod
    def default(cls) -> "ToolRegistry":
        tools = {
            "local_pdf_reader": ToolSpec(
                name="local_pdf_reader",
                kind=ToolKind.SOURCE,
                description="Read PDFs from the configured local papers directory or explicit paths.",
                entrypoint="gemma_ra.sources.local_papers:LocalPaperSource",
            ),
            "arxiv_search": ToolSpec(
                name="arxiv_search",
                kind=ToolKind.SOURCE,
                description="Search recent arXiv papers by professor name and optional topic keywords.",
                entrypoint="gemma_ra.sources.arxiv:ArxivPaperSource",
            ),
            "analysis_engine": ToolSpec(
                name="analysis_engine",
                kind=ToolKind.ANALYSIS,
                description="Run structured research analysis prompts through Gemma via Ollama.",
                entrypoint="gemma_ra.analysis.engine:AnalysisEngine",
            ),
            "future_experiment_runner": ToolSpec(
                name="future_experiment_runner",
                kind=ToolKind.FUTURE,
                description="Run constrained workspace experiments, edit code, and inspect logs within the allowed folder.",
                entrypoint="gemma_ra.core.workspace:WorkspaceExecutor",
            ),
        }
        return cls(tools=tools)

    def validate_allowed(self, allowed_tools: list[str]) -> None:
        unknown = sorted(set(allowed_tools) - set(self.tools))
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"Unknown tools declared in task spec: {names}")
