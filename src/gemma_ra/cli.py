from __future__ import annotations

import queue
import sys
import threading
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


class InteractiveGuidanceBridge:
    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._show_prompt = False

    def start(self) -> bool:
        if not sys.stdin.isatty():
            return False
        self._show_prompt = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return True

    def _reader_loop(self) -> None:
        while True:
            try:
                line = input("guidance> ")
            except Exception:  # noqa: BLE001
                return
            self._queue.put(line.rstrip("\n"))

    def drain(self) -> list[str]:
        messages: list[str] = []
        while True:
            try:
                messages.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return messages


def _normalize_professors(professors: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in professors:
        parts = [part.strip() for part in value.split(",")]
        normalized.extend(part for part in parts if part)
    return normalized


def _run_task(
    task: TaskType,
    topic: str | None,
    professors: list[str],
    papers_dir: Path | None,
    paper_paths: list[Path],
    config_path: Path | None,
    output_dir: Path | None,
    instructions: str | None = None,
    instructions_path: Path | None = None,
    verbose: bool = False,
) -> None:
    config = load_config(config_path)
    professors = _normalize_professors(professors)
    active_stream_kind: str | None = None
    guidance_bridge = InteractiveGuidanceBridge() if verbose else None

    def reporter(kind: str, message: str, end: str = "\n") -> None:
        nonlocal active_stream_kind
        if kind == "model" and task != TaskType.RUN_INSTRUCTIONS:
            return
        prefixes = {
            "agent": "[cyan]agent[/cyan]",
            "tool": "[yellow]tool[/yellow]",
            "tool_result": "[magenta]result[/magenta]",
            "thinking": "[blue]thinking[/blue]",
            "model": "[green]model[/green]",
        }
        prefix = prefixes.get(kind, "[white]info[/white]")
        is_stream = kind in {"thinking", "model"} and end == ""

        if is_stream:
            if active_stream_kind != kind:
                if active_stream_kind is not None:
                    console.print("")
                console.print(f"{prefix} ", end="", markup=True, soft_wrap=True)
                active_stream_kind = kind
            console.print(message, end="", markup=False, soft_wrap=True)
            return

        if active_stream_kind is not None:
            console.print("")
            active_stream_kind = None

        console.print(f"{prefix} {message}", end=end, markup=True, soft_wrap=True)

    guidance_enabled = guidance_bridge.start() if guidance_bridge is not None else False
    if guidance_enabled:
        console.print("[dim]Interactive guidance is available below while the agent runs.[/dim]")
        console.print("[dim]" + ("-" * min(console.size.width, 80)) + "[/dim]")

    agent = ResearchAgent(
        config,
        reporter=reporter if verbose else None,
        stream_chat=verbose,
        interactive_guidance=guidance_bridge.drain if guidance_bridge is not None else None,
    )
    try:
        artifact = agent.run(
            RunRequest(
                task=task,
                topic=topic,
                professors=professors,
                papers_dir=papers_dir,
                paper_paths=paper_paths,
                output_dir=output_dir,
                instructions=instructions,
                instructions_path=instructions_path,
            )
        )
    except GemmaRAError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt as exc:
        console.print("\n[red]Interrupted.[/red] Terminating active agent processes...")
        raise typer.Exit(code=130) from exc
    finally:
        agent.shutdown()

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
CommonInstructions = Annotated[
    Path,
    typer.Option(help="Path to a free-form instruction file.", dir_okay=False),
]
CommonVerbose = Annotated[
    bool,
    typer.Option("--verbose/--quiet", help="Stream agent progress, tool calls, and model output to stdout."),
]


@app.command("analyze-paper")
def analyze_paper(
    paper: CommonPaper,
    config: CommonConfig = None,
    output_dir: CommonOutput = None,
    verbose: CommonVerbose = True,
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
        verbose=verbose,
    )


@app.command("review-topic")
def review_topic(
    topic: CommonTopic = None,
    professor: CommonProfessors = [],
    papers_dir: CommonPapersDir = None,
    paper: CommonPaper = [],
    config: CommonConfig = None,
    output_dir: CommonOutput = None,
    verbose: CommonVerbose = True,
) -> None:
    """Synthesize a literature review from local papers or arXiv search."""
    _run_task(TaskType.REVIEW_TOPIC, topic, professor, papers_dir, paper, config, output_dir, verbose=verbose)


@app.command("find-papers")
def find_papers(
    topic: CommonTopic = None,
    professor: CommonProfessors = [],
    config: CommonConfig = None,
    output_dir: CommonOutput = None,
    verbose: CommonVerbose = True,
) -> None:
    """Find recent relevant papers from arXiv by professor name."""
    _run_task(TaskType.FIND_PAPERS, topic, professor, None, [], config, output_dir, verbose=verbose)


@app.command("generate-ideas")
def generate_ideas(
    topic: CommonTopic = None,
    professor: CommonProfessors = [],
    papers_dir: CommonPapersDir = None,
    paper: CommonPaper = [],
    config: CommonConfig = None,
    output_dir: CommonOutput = None,
    verbose: CommonVerbose = True,
) -> None:
    """Generate grounded research ideas from local papers or arXiv search."""
    _run_task(TaskType.GENERATE_IDEAS, topic, professor, papers_dir, paper, config, output_dir, verbose=verbose)


@app.command("suggest-experiments")
def suggest_experiments(
    topic: CommonTopic = None,
    professor: CommonProfessors = [],
    papers_dir: CommonPapersDir = None,
    paper: CommonPaper = [],
    config: CommonConfig = None,
    output_dir: CommonOutput = None,
    verbose: CommonVerbose = True,
) -> None:
    """Suggest lightweight experiments to validate a research idea."""
    _run_task(TaskType.SUGGEST_EXPERIMENTS, topic, professor, papers_dir, paper, config, output_dir, verbose=verbose)


@app.command("map-research-opportunities")
def map_research_opportunities(
    topic: CommonTopic = None,
    professor: CommonProfessors = [],
    papers_dir: CommonPapersDir = None,
    paper: CommonPaper = [],
    config: CommonConfig = None,
    output_dir: CommonOutput = None,
    verbose: CommonVerbose = True,
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
        verbose=verbose,
    )


@app.command("run-instructions")
def run_instructions(
    instructions_file: CommonInstructions = Path("INSTRUCTIONS.md"),
    topic: CommonTopic = None,
    professor: CommonProfessors = [],
    papers_dir: CommonPapersDir = None,
    paper: CommonPaper = [],
    config: CommonConfig = None,
    output_dir: CommonOutput = None,
    verbose: CommonVerbose = True,
) -> None:
    """Read INSTRUCTIONS.md, let the model choose tools, and write a structured result."""
    if not instructions_file.exists():
        console.print(f"[red]Error:[/red] Instructions file not found: {instructions_file}")
        raise typer.Exit(code=1)
    instructions = instructions_file.read_text().strip()
    _run_task(
        TaskType.RUN_INSTRUCTIONS,
        topic,
        professor,
        papers_dir,
        paper,
        config,
        output_dir,
        instructions=instructions,
        instructions_path=instructions_file,
        verbose=verbose,
    )
