from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from gemma_ra.agent.orchestrator import ResearchAgent, RunRequest
from gemma_ra.core.config import load_config
from gemma_ra.core.exceptions import GemmaRAError
from gemma_ra.core.schemas import TaskType

app = typer.Typer(help="Gemma-powered research assistant for local and arXiv papers.")
console = Console()


def _run_task(
    task: TaskType,
    topic: str | None,
    professors: list[str],
    papers_dir: Path | None,
    paper_paths: list[Path],
    config_path: Path | None,
    output_dir: Path | None,
) -> None:
    config = load_config(config_path)
    agent = ResearchAgent(config)
    try:
        artifact = agent.run(
            RunRequest(
                task=task,
                topic=topic,
                professors=professors,
                papers_dir=papers_dir,
                paper_paths=paper_paths,
                output_dir=output_dir,
            )
        )
    except GemmaRAError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Markdown:[/green] {artifact.markdown_path}")
    console.print(f"[green]JSON:[/green] {artifact.json_path}")


CommonTopic = Annotated[str | None, typer.Option(help="Topic, theme, or question to focus on.")]
CommonProfessors = Annotated[
    list[str],
    typer.Option("--professor", help="Professor name to use for arXiv author search.", show_default=False),
]
CommonPapersDir = Annotated[
    Path | None,
    typer.Option(help="Directory containing local PDF papers."),
]
CommonPaper = Annotated[
    list[Path],
    typer.Option("--paper", help="Explicit path to a local PDF.", show_default=False),
]
CommonConfig = Annotated[
    Path | None,
    typer.Option(help="Optional path to a gemma_ra.yaml config file."),
]
CommonOutput = Annotated[
    Path | None,
    typer.Option(help="Override output directory for generated markdown/json artifacts."),
]


@app.command("analyze-paper")
def analyze_paper(
    paper: CommonPaper,
    config: CommonConfig = None,
    output_dir: CommonOutput = None,
) -> None:
    """Analyze one local paper into a structured summary."""
    _run_task(
        task=TaskType.ANALYZE_PAPER,
        topic=None,
        professors=[],
        papers_dir=None,
        paper_paths=paper,
        config_path=config,
        output_dir=output_dir,
    )


@app.command("review-topic")
def review_topic(
    topic: CommonTopic = None,
    professor: CommonProfessors = [],
    papers_dir: CommonPapersDir = None,
    paper: CommonPaper = [],
    config: CommonConfig = None,
    output_dir: CommonOutput = None,
) -> None:
    """Synthesize a literature review from local papers or arXiv search."""
    _run_task(TaskType.REVIEW_TOPIC, topic, professor, papers_dir, paper, config, output_dir)


@app.command("find-papers")
def find_papers(
    topic: CommonTopic = None,
    professor: CommonProfessors = [],
    config: CommonConfig = None,
    output_dir: CommonOutput = None,
) -> None:
    """Find recent relevant papers from arXiv by professor name."""
    _run_task(TaskType.FIND_PAPERS, topic, professor, None, [], config, output_dir)


@app.command("generate-ideas")
def generate_ideas(
    topic: CommonTopic = None,
    professor: CommonProfessors = [],
    papers_dir: CommonPapersDir = None,
    paper: CommonPaper = [],
    config: CommonConfig = None,
    output_dir: CommonOutput = None,
) -> None:
    """Generate grounded research ideas from local papers or arXiv search."""
    _run_task(TaskType.GENERATE_IDEAS, topic, professor, papers_dir, paper, config, output_dir)


@app.command("suggest-experiments")
def suggest_experiments(
    topic: CommonTopic = None,
    professor: CommonProfessors = [],
    papers_dir: CommonPapersDir = None,
    paper: CommonPaper = [],
    config: CommonConfig = None,
    output_dir: CommonOutput = None,
) -> None:
    """Suggest lightweight experiments to validate a research idea."""
    _run_task(TaskType.SUGGEST_EXPERIMENTS, topic, professor, papers_dir, paper, config, output_dir)


@app.command("map-research-opportunities")
def map_research_opportunities(
    topic: CommonTopic = None,
    professor: CommonProfessors = [],
    papers_dir: CommonPapersDir = None,
    paper: CommonPaper = [],
    config: CommonConfig = None,
    output_dir: CommonOutput = None,
) -> None:
    """Map a field from professors and papers, then propose novel directions and experiments."""
    _run_task(
        TaskType.MAP_RESEARCH_OPPORTUNITIES,
        topic,
        professor,
        papers_dir,
        paper,
        config,
        output_dir,
    )
