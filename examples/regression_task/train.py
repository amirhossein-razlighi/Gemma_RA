from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
LOG_DIR = ROOT / "logs"
LATEST_PATH = LOG_DIR / "latest.json"
HISTORY_PATH = LOG_DIR / "history.jsonl"


def build_dataset() -> list[tuple[float, float]]:
    return [(x / 10.0, 3.5 * (x / 10.0) - 1.25) for x in range(-20, 21)]


def mean_squared_error(weight: float, bias: float, data: list[tuple[float, float]]) -> float:
    total = 0.0
    for x_value, y_value in data:
        prediction = weight * x_value + bias
        total += (prediction - y_value) ** 2
    return total / len(data)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    learning_rate = float(config["learning_rate"])
    epochs = int(config["epochs"])
    log_every = int(config["log_every"])
    target_loss = float(config["target_loss"])

    data = build_dataset()
    weight = 0.0
    bias = 0.0
    checkpoints: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        grad_weight = 0.0
        grad_bias = 0.0
        for x_value, y_value in data:
            prediction = weight * x_value + bias
            error = prediction - y_value
            grad_weight += 2.0 * error * x_value
            grad_bias += 2.0 * error

        grad_weight /= len(data)
        grad_bias /= len(data)

        weight -= learning_rate * grad_weight
        bias -= learning_rate * grad_bias

        if epoch % log_every == 0 or epoch == epochs:
            checkpoints.append(
                {
                    "epoch": epoch,
                    "loss": mean_squared_error(weight, bias, data),
                    "weight": weight,
                    "bias": bias,
                }
            )

    final_loss = mean_squared_error(weight, bias, data)
    success = final_loss <= target_loss

    result = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "target_loss": target_loss,
        "final_loss": final_loss,
        "success": success,
        "weight": weight,
        "bias": bias,
        "checkpoints": checkpoints,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(json.dumps(result, indent=2))
    with HISTORY_PATH.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(result) + "\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

