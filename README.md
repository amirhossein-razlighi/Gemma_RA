# Gemma_RA

Gemma_RA is a CLI-first research assistant agent for on-device workflows using Gemma through Ollama. It can read local PDFs, discover recent papers on arXiv by professor name, and turn those sources into structured paper analyses, literature reviews, research ideas, and lightweight experiment suggestions.

## What v1 does

- Analyze a local paper into problem, inputs, method, outputs, ideas, contributions, and limitations
- Review a topic from local papers or professor-based arXiv discovery
- Find recent papers from arXiv and explain why they matter
- Generate possible research directions grounded in the supplied papers
- Suggest small experiments to test whether an idea is worth pursuing
- Map a field end-to-end from seed professors and papers to novel research opportunities
- Read a root `INSTRUCTIONS.md`, choose tools, and execute a broader autonomous research workflow

The codebase is intentionally extensible. Tasks are defined in Python, tools are registry-driven, and the orchestration loop is designed so future experiment-running tools can plug in without rewriting the core architecture.

The runtime now uses two model interaction modes:

- structured JSON generation for final task outputs
- Ollama chat tool calling for source discovery before the final synthesis pass

## Project layout

- `src/gemma_ra/agent`: orchestration, skill loading, and tool registry
- `src/gemma_ra/sources`: local PDF ingestion and arXiv search
- `src/gemma_ra/analysis`: prompt assembly, structured model calls, and markdown rendering
- `src/gemma_ra/core`: config, schemas, artifacts, exceptions, the Ollama client, and constrained workspace execution
- `src/gemma_ra/core/tasks.py`: code-defined task behavior and prompt templates
- `tests/`: unit and integration-style tests with mocked model boundaries

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.com/)
- A local Gemma model available in Ollama

Example:

```bash
ollama pull gemma4
```

## Setup

```bash
uv venv
uv sync --extra dev
```

The default config lives in `gemma_ra.yaml`:

```yaml
ollama:
  host: "http://localhost:11434"
  model: "gemma4"
executor:
  workspace_root: "."
  max_iterations: 20
  command_timeout_seconds: 120
papers_dir: "./papers"
output_dir: "./outputs"
```

You can edit that file or pass `--config /path/to/config.yaml` to any command.

## CLI usage

Analyze a local paper:

```bash
uv run gemma-ra analyze-paper --paper ./papers/example.pdf
```

Review a topic from local PDFs:

```bash
uv run gemma-ra review-topic --topic "graph representation learning" --papers-dir ./papers
```

Find papers by professor name on arXiv:

```bash
uv run gemma-ra find-papers --topic "multimodal reasoning" --professor "Percy Liang"
```

Generate research ideas:

```bash
uv run gemma-ra generate-ideas --topic "small-model agents" --professor "Yejin Choi"
```

Suggest lightweight experiments:

```bash
uv run gemma-ra suggest-experiments --topic "retrieval-free literature agents" --papers-dir ./papers
```

Map a field in one run:

```bash
uv run gemma-ra map-research-opportunities --topic "agentic retrieval" --professor "Percy Liang"
```

Run a free-form instruction file:

```bash
uv run gemma-ra run-instructions --instructions-file ./INSTRUCTIONS.md
```

Each run writes:

- Markdown output for reading
- JSON output for downstream tooling

By default artifacts are stored under `./outputs`.

For `find-papers`, the system now tries broader arXiv fallbacks when an exact author-plus-topic query is empty, and includes a search trace in the output when no strong matches are found.

For `run-instructions`, the model can now use constrained workspace tools to:

- list files under a specified workspace root
- read workspace files and logs
- write workspace files
- run Python scripts through `uv run python`

Those tools are intentionally limited to the configured workspace root, and the tool loop is capped by `executor.max_iterations` rather than running forever.

## Notes on scope

- v1 uses Ollama as the only model backend.
- Online paper discovery is arXiv-only for now.
- Local PDF ingestion assumes text-extractable PDFs rather than OCR.
- Future work can add experiment execution, code-running tools, hyperparameter loops, and result analysis through the existing tool and artifact interfaces.
- arXiv API requests use `https://export.arxiv.org/api/query`.

## Testing

```bash
uv run pytest
```
