from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from ast import FunctionDef, AsyncFunctionDef, ClassDef, parse
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
        normalized_pattern = pattern.strip() or "*"
        return sorted(
            str(path.relative_to(self.root))
            for path in target.rglob(normalized_pattern)
            if path.is_file()
        )

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

    def append_file(self, path: str, content: str) -> dict[str, Any]:
        target = self.resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = target.read_text() if target.exists() else ""
        prefix = ""
        if existing and not existing.endswith("\n"):
            prefix = "\n"
        target.write_text(existing + prefix + content)
        return {
            "path": str(target.relative_to(self.root)),
            "appended_chars": len(prefix + content),
        }

    def replace_in_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        count: int = 1,
    ) -> dict[str, Any]:
        if count <= 0:
            raise ValueError("count must be positive.")
        target = self.resolve_path(path)
        source = target.read_text()
        occurrences = source.count(old_text)
        if occurrences == 0:
            raise ValueError(f"Text to replace was not found in {path}.")
        updated = source.replace(old_text, new_text, count)
        replacements = min(occurrences, count)
        target.write_text(updated)
        return {
            "path": str(target.relative_to(self.root)),
            "replacements": replacements,
        }

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

    def python_syntax_check(self, path: str) -> dict[str, Any]:
        target = self.resolve_path(path)
        if target.suffix != ".py":
            raise ValueError(f"Syntax check only supports Python files: {path}")
        command = ["uv", "run", "python", "-m", "py_compile", str(target)]
        process = subprocess.run(
            command,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.config.command_timeout_seconds,
            check=False,
        )
        return {
            "path": str(target.relative_to(self.root)),
            "ok": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": process.stdout[-4000:],
            "stderr": process.stderr[-4000:],
        }

    def upsert_python_function(
        self,
        path: str,
        function_name: str,
        function_source: str,
        class_name: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_path(path)
        if target.suffix != ".py":
            raise ValueError(f"Function upsert only supports Python files: {path}")
        source = target.read_text()
        tree = parse(source)
        lines = source.splitlines()
        insertion_kind = "created"

        if class_name:
            target_node = next(
                (
                    node
                    for node in tree.body
                    if isinstance(node, ClassDef) and node.name == class_name
                ),
                None,
            )
            if target_node is None:
                raise ValueError(f'Class "{class_name}" was not found in {path}.')
            class_lines = lines[target_node.lineno - 1 : target_node.end_lineno]
            class_indent = self._indent_of_line(class_lines[0])
            body_indent = class_indent + "    "
            indented_source = self._indent_block(function_source.strip("\n"), body_indent)
            for child in target_node.body:
                if isinstance(child, (FunctionDef, AsyncFunctionDef)) and child.name == function_name:
                    start = child.lineno - 1
                    end = child.end_lineno
                    class_lines[start - (target_node.lineno - 1) : end - (target_node.lineno - 1)] = indented_source.splitlines()
                    insertion_kind = "updated"
                    break
            else:
                if class_lines and class_lines[-1].strip():
                    class_lines.append("")
                class_lines.extend(indented_source.splitlines())
            lines[target_node.lineno - 1 : target_node.end_lineno] = class_lines
        else:
            replacement_lines = function_source.strip("\n").splitlines()
            for node in tree.body:
                if isinstance(node, (FunctionDef, AsyncFunctionDef)) and node.name == function_name:
                    lines[node.lineno - 1 : node.end_lineno] = replacement_lines
                    insertion_kind = "updated"
                    break
            else:
                if lines and lines[-1].strip():
                    lines.append("")
                lines.extend(replacement_lines)

        updated = "\n".join(lines).rstrip() + "\n"
        target.write_text(updated)
        return {
            "path": str(target.relative_to(self.root)),
            "function_name": function_name,
            "class_name": class_name,
            "action": insertion_kind,
        }

    @staticmethod
    def _indent_of_line(line: str) -> str:
        return line[: len(line) - len(line.lstrip(" "))]

    @staticmethod
    def _indent_block(block: str, indent: str) -> str:
        indented_lines = []
        for line in block.splitlines():
            indented_lines.append(indent + line if line else "")
        return "\n".join(indented_lines)

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
