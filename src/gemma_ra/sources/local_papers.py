from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from gemma_ra.core.exceptions import SourceError
from gemma_ra.core.schemas import PaperDocument, PaperMetadata


class LocalPaperSource:
    def discover(self, papers_dir: Path) -> list[Path]:
        if not papers_dir.exists():
            raise SourceError(f"Papers directory does not exist: {papers_dir}")
        return sorted(path for path in papers_dir.rglob("*.pdf") if path.is_file())

    def read_many(self, paths: list[Path]) -> list[PaperDocument]:
        return [self.read(path) for path in paths]

    def read(self, path: Path) -> PaperDocument:
        if not path.exists():
            raise SourceError(f"Paper not found: {path}")
        try:
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as exc:  # noqa: BLE001
            raise SourceError(f"Failed to read PDF {path}: {exc}") from exc

        if not text:
            raise SourceError(f"PDF did not contain extractable text: {path}")

        metadata = PaperMetadata(
            paper_id=path.stem,
            title=path.stem.replace("_", " ").replace("-", " ").title(),
            authors=[],
            abstract=None,
            local_path=path,
            source="local",
        )
        return PaperDocument(metadata=metadata, content=text, sections=[])

