from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


DEFAULT_CONFIG_PATH = Path("gemma_ra.yaml")


class ArxivConfig(BaseModel):
    base_url: str = "https://export.arxiv.org/api/query"
    max_results: int = 5
    sort_by: str = "submittedDate"
    sort_order: str = "descending"


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "gemma4"
    timeout_seconds: float = 120.0


class ExecutorConfig(BaseModel):
    workspace_root: Path = Path(".")
    max_iterations: int = 20
    command_timeout_seconds: float = 120.0
    max_file_bytes: int = 200_000


class AppConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    arxiv: ArxivConfig = Field(default_factory=ArxivConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    papers_dir: Path = Path("./papers")
    output_dir: Path = Path("./outputs")


def load_config(config_path: Path | None = None) -> AppConfig:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return AppConfig()

    data = yaml.safe_load(path.read_text()) or {}
    return AppConfig.model_validate(data)
