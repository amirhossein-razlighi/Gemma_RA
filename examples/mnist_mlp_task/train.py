from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
LOG_DIR = ROOT / "logs"
LATEST_PATH = LOG_DIR / "latest.json"
HISTORY_PATH = LOG_DIR / "history.jsonl"
IMAGE_SIZE = 8
NUM_CLASSES = 10


def digit_patterns() -> dict[int, torch.Tensor]:
    patterns: dict[int, list[str]] = {
        0: [
            "00111100",
            "01100110",
            "11000011",
            "11000011",
            "11000011",
            "11000011",
            "01100110",
            "00111100",
        ],
        1: [
            "00011000",
            "00111000",
            "00011000",
            "00011000",
            "00011000",
            "00011000",
            "00111100",
            "00111100",
        ],
        2: [
            "00111100",
            "01100110",
            "00000110",
            "00001100",
            "00011000",
            "00110000",
            "01100000",
            "01111110",
        ],
        3: [
            "00111100",
            "01100110",
            "00000110",
            "00011100",
            "00000110",
            "00000110",
            "01100110",
            "00111100",
        ],
        4: [
            "00001100",
            "00011100",
            "00101100",
            "01001100",
            "11111110",
            "00001100",
            "00001100",
            "00011110",
        ],
        5: [
            "01111110",
            "01100000",
            "01111100",
            "00000110",
            "00000110",
            "00000110",
            "01100110",
            "00111100",
        ],
        6: [
            "00111100",
            "01100110",
            "01100000",
            "01111100",
            "01100110",
            "01100110",
            "01100110",
            "00111100",
        ],
        7: [
            "01111110",
            "00000110",
            "00001100",
            "00011000",
            "00110000",
            "00110000",
            "00110000",
            "00110000",
        ],
        8: [
            "00111100",
            "01100110",
            "01100110",
            "00111100",
            "01100110",
            "01100110",
            "01100110",
            "00111100",
        ],
        9: [
            "00111100",
            "01100110",
            "01100110",
            "01100110",
            "00111110",
            "00000110",
            "01100110",
            "00111100",
        ],
    }
    result: dict[int, torch.Tensor] = {}
    for label, rows in patterns.items():
        result[label] = torch.tensor(
            [[float(char) for char in row] for row in rows],
            dtype=torch.float32,
        )
    return result


def build_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    patterns = digit_patterns()
    samples: list[torch.Tensor] = []
    labels: list[int] = []
    for label, pattern in patterns.items():
        for _ in range(26):
            sample = pattern.clone()
            shift_y = int(torch.randint(-1, 2, (1,), generator=generator).item())
            shift_x = int(torch.randint(-1, 2, (1,), generator=generator).item())
            sample = torch.roll(sample, shifts=(shift_y, shift_x), dims=(0, 1))
            dropout_mask = torch.rand((IMAGE_SIZE, IMAGE_SIZE), generator=generator) < 0.08
            boost_mask = torch.rand((IMAGE_SIZE, IMAGE_SIZE), generator=generator) < 0.04
            jitter = torch.randn((IMAGE_SIZE, IMAGE_SIZE), generator=generator) * 0.08
            sample = torch.clamp(sample - dropout_mask.float(), 0.0, 1.0)
            sample = torch.clamp(sample + 0.35 * boost_mask.float() + jitter, 0.0, 1.0)
            samples.append(sample.reshape(-1))
            labels.append(label)
    return torch.stack(samples), torch.tensor(labels, dtype=torch.long)


def train_validation_split(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validation_indices: list[int] = []
    train_indices: list[int] = []
    for label in range(NUM_CLASSES):
        indices = (labels == label).nonzero(as_tuple=False).flatten()
        validation_indices.extend(indices[-6:].tolist())
        train_indices.extend(indices[:-6].tolist())
    return (
        features[train_indices],
        labels[train_indices],
        features[validation_indices],
        labels[validation_indices],
    )


class DigitMLP(nn.Module):
    def __init__(self, hidden_size: int, depth: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = IMAGE_SIZE * IMAGE_SIZE
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, NUM_CLASSES))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def evaluate(model: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = nn.functional.cross_entropy(logits, labels).item()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def main() -> None:
    torch.manual_seed(11)
    config = json.loads(CONFIG_PATH.read_text())
    learning_rate = float(config["learning_rate"])
    epochs = int(config["epochs"])
    hidden_size = int(config["hidden_size"])
    depth = int(config["depth"])
    dropout = float(config["dropout"])
    weight_decay = float(config["weight_decay"])
    log_every = int(config["log_every"])
    target_accuracy = float(config["target_accuracy"])

    features, labels = build_dataset()
    train_x, train_y, validation_x, validation_y = train_validation_split(features, labels)

    model = DigitMLP(hidden_size=hidden_size, depth=depth, dropout=dropout)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    checkpoints: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs:
            train_predictions = logits.argmax(dim=1)
            train_accuracy = (train_predictions == train_y).float().mean().item()
            validation_loss, validation_accuracy = evaluate(model, validation_x, validation_y)
            checkpoints.append(
                {
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                }
            )

    final = checkpoints[-1]
    result = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "hidden_size": hidden_size,
        "depth": depth,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "target_accuracy": target_accuracy,
        "train_loss": final["train_loss"],
        "train_accuracy": final["train_accuracy"],
        "validation_loss": final["validation_loss"],
        "validation_accuracy": final["validation_accuracy"],
        "success": float(final["validation_accuracy"]) >= target_accuracy,
        "checkpoints": checkpoints,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(json.dumps(result, indent=2))
    with HISTORY_PATH.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(result) + "\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
