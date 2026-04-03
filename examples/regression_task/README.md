# Regression Task Demo

This example is a tiny gradient-descent regression project meant to exercise the instruction-driven agent loop.

The agent should be able to:

- inspect the code in this folder
- run the training script with `uv run python`
- read the generated logs
- change hyperparameters in `config.json`
- rerun until the target loss is achieved

## Files

- `train.py`: deterministic regression training script
- `config.json`: tunable hyperparameters
- `INSTRUCTIONS.md`: free-form agent objective
- `logs/latest.json`: most recent metrics after running training
- `logs/history.jsonl`: append-only run history

## Manual baseline

From the repo root:

```bash
uv run python examples/regression_task/train.py
```

That will use the intentionally weak starting hyperparameters in `config.json`, so the loss should still be above the target.

## Agent demo

From the repo root:

```bash
uv run gemma-ra run-instructions \
  --instructions-file examples/regression_task/INSTRUCTIONS.md \
  --config examples/regression_task/gemma_ra.example.yaml
```

The agent is constrained to this folder through the example config.

