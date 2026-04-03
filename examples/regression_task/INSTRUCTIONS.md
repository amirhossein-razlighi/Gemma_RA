You are working inside a small regression demo project.

Your goal is to make the training succeed.

Success criteria:
- `logs/latest.json` must show `"success": true`
- `final_loss` must be less than or equal to `0.005`

Available behaviors you should use:
- inspect the files in this folder first
- read the training code
- run the training script
- read the logs
- if it failed, change hyperparameters or code
- rerun until the success criteria are reached
- when done, summarize what you changed and why it worked

Rules:
- stay inside `examples/regression_task/` only
- prefer changing `config.json` before changing the training code
- do not stop after one failed run
- finish only when the success criteria are met
- You are NOT ALLOWED to change the dataset
