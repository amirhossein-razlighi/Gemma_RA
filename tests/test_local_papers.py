from pathlib import Path

import pytest

from gemma_ra.core.exceptions import SourceError
from gemma_ra.sources.local_papers import LocalPaperSource


def test_discover_returns_pdf_files(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "notes.txt").write_text("ignore me")

    source = LocalPaperSource()

    paths = source.discover(tmp_path)

    assert paths == [tmp_path / "a.pdf"]


def test_discover_errors_for_missing_directory(tmp_path: Path) -> None:
    source = LocalPaperSource()

    with pytest.raises(SourceError):
        source.discover(tmp_path / "missing")

