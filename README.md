# Gemma_RA

Personal research assistant for Gemma 4 on Ollama.

Gemma_RA can:
- read local PDFs
- search arXiv by professor name
- write paper summaries and mini-surveys
- generate grounded research ideas and experiments
- run constrained code-and-log loops from a free-form `INSTRUCTIONS.md`

## 🚀 TL;DR

If you want your own local research assistant with Gemma:

```bash
ollama pull gemma4
uv sync --extra dev
uv run gemma-ra run-instructions --instructions-file ./INSTRUCTIONS.md
```

Or if you want to see it work immediately:

```bash
uv run gemma-ra run-instructions \
  --instructions-file examples/regression_task/INSTRUCTIONS.md \
  --config examples/regression_task/gemma_ra.example.yaml
```

## Why This Repo Is Useful

- Local-first: Gemma runs through Ollama on your machine
- Paper-first: works with PDFs, arXiv, and professor-name discovery
- Agentic when needed: can inspect files, run code, read logs, edit configs, and iterate
- Constrained by design: tool use stays inside a configured workspace
- Structured outputs: each run writes both Markdown and JSON

## What You Can Do

### Paper workflows

- `analyze-paper`: break one paper into problem, method, contributions, and limitations
- `review-topic`: produce a compact survey across papers
- `find-papers`: search and triage relevant papers
- `generate-ideas`: propose research directions grounded in papers
- `suggest-experiments`: propose lightweight validation experiments
- `map-research-opportunities`: go from seed professors to field summary, open problems, ideas, and experiments

### Instruction workflow

`run-instructions` reads a free-form `INSTRUCTIONS.md` and lets the model decide which tools to call.

That means you can write things like:

- “Read papers from Andreas Tagliasacchi related to Gaussian Splatting and summarize the field.”
- “Tune this training script until validation accuracy reaches 0.88.”
- “Stop changing hyperparameters and edit `train.py` instead.”

## 🔧 Quick Start

### 1. Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.com/)
- a local Gemma 4 model

```bash
ollama pull gemma4
```

### 2. Install

```bash
uv venv
uv sync --extra dev
```

### 3. Check the CLI

```bash
uv run gemma-ra --help
```

## ⚡ Fastest Ways To Try It

### A. Analyze a paper

```bash
uv run gemma-ra analyze-paper --paper ./papers/example.pdf
```

### B. Run the autonomous demo

```bash
uv run gemma-ra run-instructions \
  --instructions-file examples/regression_task/INSTRUCTIONS.md \
  --config examples/regression_task/gemma_ra.example.yaml
```

This demo starts from a failing setup and lets the agent:
- inspect the workspace
- run training
- read logs
- edit config/code
- rerun until the target is met

📊 The bundled regression demo reached:
- initial loss: `16.7727`
- final loss: `0.0000803726`
- success target: `final_loss <= 0.005`

![Regression tuning history](static/regression_tuning_history.png)

There is also a harder `torch` example in [examples/mnist_mlp_task/README.md](examples/mnist_mlp_task/README.md).

## 🧭 Common Commands

Analyze a local paper:

```bash
uv run gemma-ra analyze-paper --paper ./papers/example.pdf
```

Review a topic from local papers:

```bash
uv run gemma-ra review-topic --topic "diffusion models" --papers-dir ./papers
```

Search papers by professor name:

```bash
uv run gemma-ra find-papers --topic "computer vision" --professor "Ali Mahdavi Amiri"
```

Use multiple professors:

```bash
uv run gemma-ra generate-ideas \
  --topic "computer vision" \
  --professor "Daniel CohenOr, Richard Zhang"
```

Map a field end-to-end:

```bash
uv run gemma-ra map-research-opportunities \
  --topic "3d computer vision" \
  --professor "Luc Van Gool, Andreas Tagliasacchi"
```

Run your own instruction file:

```bash
uv run gemma-ra run-instructions --instructions-file ./INSTRUCTIONS.md
```

## ⚙ How `run-instructions` Works

1. You write a goal in `INSTRUCTIONS.md`
2. The agent reads the workspace and available papers
3. It chooses tools, runs code if needed, and reads logs/results
4. It keeps iterating until success criteria are met or the iteration budget is exhausted

While it runs, you can type guidance directly in the terminal to steer it.

Examples:

- `stop hyperparameter tuning and modify train.py`
- `focus on papers from 2024 and later`
- `fetch full text before summarizing`

## Workspace Tools

Inside `run-instructions`, the agent can use constrained tools to:

- list files
- read text files and logs
- write or append files
- replace snippets in files
- update JSON config fields
- edit Python functions
- syntax-check Python
- run Python via `uv run python`

All of this is restricted to the configured workspace root.

## ⚙ Configuration

Default config lives in [gemma_ra.yaml](./gemma_ra.yaml).

Important fields:

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

## Outputs

Each run writes:

- a Markdown artifact for humans
- a JSON artifact for tooling

By default they are saved in `./outputs`.

## Repo Layout

- [src/gemma_ra/cli.py](src/gemma_ra/cli.py): CLI entrypoints
- [src/gemma_ra/agent](src/gemma_ra/agent): orchestration
- [src/gemma_ra/sources](src/gemma_ra/sources): local PDFs and arXiv
- [src/gemma_ra/analysis](src/gemma_ra/analysis): tool loop and structured outputs
- [src/gemma_ra/core](src/gemma_ra/core): config, schemas, model client, workspace tools
- [examples](examples): runnable demos

## Limits In v0.1.0

- Ollama is the only backend
- arXiv is the only online paper source
- local PDF ingestion assumes text-extractable PDFs
- OCR and broad web indexing are out of scope for now

## Testing

```bash
uv run pytest
```
