from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from gemma_ra.core.schemas import ArtifactRecord, TaskType


def slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    return "-".join(part for part in cleaned.split("-") if part) or "run"


def write_artifact(
    output_dir: Path,
    task: TaskType,
    task_name: str,
    title: str,
    markdown: str,
    payload: BaseModel | dict[str, Any],
) -> ArtifactRecord:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    stem = f"{timestamp.strftime('%Y%m%d-%H%M%S')}-{slugify(title)}"
    markdown_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    markdown_path.write_text(markdown)
    if isinstance(payload, BaseModel):
        serializable = payload.model_dump(mode="json")
    else:
        serializable = payload
    json_path.write_text(json.dumps(serializable, indent=2))
    return ArtifactRecord(
        task=task,
        task_name=task_name,
        generated_at=timestamp,
        markdown_path=markdown_path,
        json_path=json_path,
    )
