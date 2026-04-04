from pathlib import Path

from gemma_ra.core.config import ExecutorConfig
from gemma_ra.core.workspace import WorkspaceExecutor


def test_replace_and_append_in_workspace_file(tmp_path: Path) -> None:
    executor = WorkspaceExecutor(ExecutorConfig(workspace_root=tmp_path))
    target = tmp_path / "module.py"
    target.write_text("value = 1\n")

    replace_result = executor.replace_in_file("module.py", "value = 1", "value = 2")
    append_result = executor.append_file("module.py", "\nprint(value)\n")

    assert replace_result["replacements"] == 1
    assert append_result["appended_chars"] > 0
    assert target.read_text() == "value = 2\n\nprint(value)\n"


def test_list_files_treats_empty_pattern_as_wildcard(tmp_path: Path) -> None:
    executor = WorkspaceExecutor(ExecutorConfig(workspace_root=tmp_path))
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "train.py").write_text("print('ok')\n")

    files = executor.list_files(".", "")

    assert files == ["nested/train.py", "notes.txt"]


def test_python_syntax_check_reports_failure(tmp_path: Path) -> None:
    executor = WorkspaceExecutor(ExecutorConfig(workspace_root=tmp_path))
    target = tmp_path / "broken.py"
    target.write_text("def broken(:\n    pass\n")

    result = executor.python_syntax_check("broken.py")

    assert result["ok"] is False
    assert result["returncode"] != 0


def test_upsert_python_function_updates_and_creates(tmp_path: Path) -> None:
    executor = WorkspaceExecutor(ExecutorConfig(workspace_root=tmp_path))
    target = tmp_path / "helpers.py"
    target.write_text(
        "def old_function():\n"
        "    return 'old'\n"
        "\n"
        "class Greeter:\n"
        "    def greet(self):\n"
        "        return 'hello'\n"
    )

    update_result = executor.upsert_python_function(
        "helpers.py",
        "old_function",
        "def old_function():\n    return 'new'\n",
    )
    create_result = executor.upsert_python_function(
        "helpers.py",
        "farewell",
        "def farewell(self):\n    return 'bye'\n",
        class_name="Greeter",
    )

    contents = target.read_text()
    assert update_result["action"] == "updated"
    assert create_result["action"] == "created"
    assert "return 'new'" in contents
    assert "def farewell(self):" in contents
