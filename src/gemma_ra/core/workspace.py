from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from gemma_ra.core.config import ExecutorConfig


class WorkspaceExecutor:
    def __init__(self, config: ExecutorConfig) -> None:
        self.config = config
        self.root = config.workspace_root.resolve()
        self._active_processes: dict[int, subprocess.Popen[str]] = {}

    def resolve_path(self, raw_path: str) -> Path:
        raw = Path(raw_path)
        candidates: list[Path] = []

        if raw.is_absolute():
            candidates.append(raw.resolve())
        else:
            candidates.append((self.root / raw).resolve())
            candidates.append((Path.cwd() / raw).resolve())

        existing_in_root = [
            candidate
            for candidate in candidates
            if candidate.exists() and str(candidate).startswith(str(self.root))
        ]
        if existing_in_root:
            return existing_in_root[0]

        for candidate in candidates:
            if str(candidate).startswith(str(self.root)):
                return candidate

        raise ValueError(f"Path is outside allowed workspace root: {raw_path}")

    def list_files(self, directory: str = ".", pattern: str = "*") -> list[str]:
        target = self.resolve_path(directory)
        if not target.exists():
            return []
        return sorted(str(path.relative_to(self.root)) for path in target.rglob(pattern) if path.is_file())

    def read_file(self, path: str, max_bytes: int | None = None) -> str:
        target = self.resolve_path(path)
        data = target.read_bytes()
        limit = max_bytes or self.config.max_file_bytes
        return data[:limit].decode("utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> str:
        target = self.resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return str(target.relative_to(self.root))

    def update_json_field(self, path: str, field: str, value: Any) -> dict[str, Any]:
        target = self.resolve_path(path)
        payload = json.loads(target.read_text())
        payload[field] = value
        target.write_text(json.dumps(payload, indent=2) + "\n")
        return {
            "path": str(target.relative_to(self.root)),
            "field": field,
            "value": value,
        }

    def run_uv_python(
        self,
        script_path: str,
        args: list[str] | None = None,
        working_directory: str = ".",
        log_paths: list[str] | None = None,
        poll_interval_seconds: float = 0.5,
    ) -> dict[str, Any]:
        script = self.resolve_path(script_path)
        cwd = self.resolve_path(working_directory)
        if not script.exists():
            available_scripts = self.list_files(".", "*.py")
            raise FileNotFoundError(
                f"Script not found inside workspace: {script_path}. "
                f"Available Python files: {available_scripts[:20]}"
            )
        command = ["uv", "run", "python", str(script), *(args or [])]
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self._active_processes[process.pid] = process
        started_logs: list[str] = []
        deadline = time.time() + self.config.command_timeout_seconds
        while True:
            returncode = process.poll()
            if log_paths:
                for raw_path in log_paths:
                    try:
                        log_path = self.resolve_path(raw_path)
                    except ValueError:
                        continue
                    if log_path.exists():
                        relative = str(log_path.relative_to(self.root))
                        if relative not in started_logs:
                            started_logs.append(relative)
            if returncode is not None:
                break
            if time.time() >= deadline:
                self.terminate_all()
                raise TimeoutError(f"Command timed out after {self.config.command_timeout_seconds} seconds: {command}")
            time.sleep(poll_interval_seconds)
        try:
            stdout, stderr = process.communicate(timeout=1.0)
        finally:
            self._active_processes.pop(process.pid, None)
        return {
            "command": command,
            "returncode": process.returncode,
            "stdout": stdout[-8000:],
            "stderr": stderr[-8000:],
            "observed_logs": started_logs,
            "finished": True,
        }

    def terminate_all(self) -> None:
        for pid, process in list(self._active_processes.items()):
            if process.poll() is not None:
                self._active_processes.pop(pid, None)
                continue
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                process.terminate()
        deadline = time.time() + 2.0
        while self._active_processes and time.time() < deadline:
            for pid, process in list(self._active_processes.items()):
                if process.poll() is not None:
                    self._active_processes.pop(pid, None)
            time.sleep(0.1)
        for pid, process in list(self._active_processes.items()):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    process.kill()
            self._active_processes.pop(pid, None)
