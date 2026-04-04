from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import httpx

from gemma_ra.core.config import ArxivConfig
from gemma_ra.core.exceptions import SourceError
from gemma_ra.core.schemas import PaperDocument, PaperMetadata
from gemma_ra.sources.local_papers import LocalPaperSource

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivPaperSource:
    def __init__(self, config: ArxivConfig) -> None:
        self.config = config
        self.local_source = LocalPaperSource()

    def search_and_load(
        self,
        professors: list[str],
        topic: str | None,
    ) -> tuple[list[PaperDocument], list[str]]:
        papers: list[PaperDocument] = []
        notes: list[str] = []
        for professor in professors:
            found, trace = self.search_with_fallbacks(professor=professor, topic=topic)
            papers.extend(found)
            notes.extend(trace)
        return self._dedupe(papers), notes

    def search(self, professor: str, topic: str | None) -> list[PaperDocument]:
        query_parts = [f'au:"{professor}"']
        if topic:
            query_parts.append(f'all:"{topic}"')
        return self.search_query(" AND ".join(query_parts))

    def search_with_fallbacks(
        self,
        professor: str,
        topic: str | None,
    ) -> tuple[list[PaperDocument], list[str]]:
        queries: list[tuple[str, str]] = []
        for name_variant in self._name_variants(professor):
            surname = name_variant.split()[-1] if name_variant.split() else name_variant
            if topic:
                queries.append((f'author+topic ({name_variant})', f'au:"{name_variant}" AND all:"{topic}"'))
            queries.append((f'author-only ({name_variant})', f'au:"{name_variant}"'))
            if topic and surname and surname.lower() != name_variant.lower():
                queries.append((f'surname+topic ({surname})', f'au:"{surname}" AND all:"{topic}"'))
            if surname and surname.lower() != name_variant.lower():
                queries.append((f'surname-only ({surname})', f'au:"{surname}"'))
            if topic:
                queries.append((f'fulltext-name+topic ({name_variant})', f'all:"{name_variant}" AND all:"{topic}"'))
            queries.append((f'fulltext-name ({name_variant})', f'all:"{name_variant}"'))

        deduped_queries: list[tuple[str, str]] = []
        seen_queries: set[str] = set()
        for label, query in queries:
            if query in seen_queries:
                continue
            seen_queries.add(query)
            deduped_queries.append((label, query))

        combined: list[PaperDocument] = []
        notes: list[str] = []
        for label, query in deduped_queries:
            docs = self.search_query(query)
            notes.append(f'arXiv search "{label}" for "{professor}" returned {len(docs)} result(s).')
            combined.extend(docs)
            if docs and (label.startswith("author+topic") or label.startswith("author-only")):
                break

        deduped = self._dedupe(combined)
        if not deduped:
            notes.append(
                f'No arXiv papers matched professor "{professor}"'
                + (f' with topic "{topic}".' if topic else ".")
            )
        return deduped, notes

    @staticmethod
    def _name_variants(professor: str) -> list[str]:
        cleaned = " ".join(professor.strip().split())
        variants: list[str] = []
        if cleaned:
            variants.append(cleaned)
            split_camel = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cleaned)
            split_camel = " ".join(split_camel.split())
            if split_camel and split_camel not in variants:
                variants.append(split_camel)
            dehyphenated = cleaned.replace("-", " ")
            dehyphenated = " ".join(dehyphenated.split())
            if dehyphenated and dehyphenated not in variants:
                variants.append(dehyphenated)
            normalized = re.sub(r"[^A-Za-z0-9]+", " ", split_camel)
            normalized = " ".join(normalized.split())
            if normalized and normalized not in variants:
                variants.append(normalized)
        return variants

    def search_query(self, search_query: str) -> list[PaperDocument]:
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": self.config.max_results,
            "sortBy": self.config.sort_by,
            "sortOrder": self.config.sort_order,
        }
        try:
            response = httpx.get(
                self.config.base_url,
                params=params,
                timeout=30.0,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SourceError(f"Failed to query arXiv for query '{search_query}': {exc}") from exc
        return self._parse_feed(response.text)

    def fetch_pdf_document(self, metadata: PaperMetadata, download_dir: Path) -> PaperDocument:
        if metadata.pdf_url is None:
            raise SourceError(f"Paper {metadata.paper_id} does not include a PDF URL.")
        download_dir.mkdir(parents=True, exist_ok=True)
        target_path = download_dir / f"{metadata.paper_id}.pdf"
        try:
            response = httpx.get(str(metadata.pdf_url), timeout=60.0, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SourceError(f"Failed to download arXiv PDF for {metadata.paper_id}: {exc}") from exc
        target_path.write_bytes(response.content)
        document = self.local_source.read(target_path)
        document.metadata = PaperMetadata(
            paper_id=metadata.paper_id,
            title=metadata.title,
            authors=metadata.authors,
            abstract=metadata.abstract,
            published=metadata.published,
            updated=metadata.updated,
            pdf_url=metadata.pdf_url,
            local_path=target_path,
            source="arxiv_pdf",
        )
        return document

    def _parse_feed(self, xml_text: str) -> list[PaperDocument]:
        root = ElementTree.fromstring(xml_text)
        documents: list[PaperDocument] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            paper_id = entry.findtext("atom:id", namespaces=ATOM_NS) or "unknown"
            title = (entry.findtext("atom:title", namespaces=ATOM_NS) or "").strip()
            abstract = (entry.findtext("atom:summary", namespaces=ATOM_NS) or "").strip()
            authors = [
                author.findtext("atom:name", namespaces=ATOM_NS) or ""
                for author in entry.findall("atom:author", ATOM_NS)
            ]
            published = self._parse_datetime(entry.findtext("atom:published", namespaces=ATOM_NS))
            updated = self._parse_datetime(entry.findtext("atom:updated", namespaces=ATOM_NS))
            pdf_url = None
            for link in entry.findall("atom:link", ATOM_NS):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href")
                    break
            metadata = PaperMetadata(
                paper_id=paper_id.rsplit("/", maxsplit=1)[-1],
                title=title,
                authors=[author for author in authors if author],
                abstract=abstract,
                published=published,
                updated=updated,
                pdf_url=pdf_url,
                source="arxiv",
            )
            documents.append(PaperDocument(metadata=metadata, content=abstract, sections=[]))
        return documents

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    @staticmethod
    def _dedupe(documents: list[PaperDocument]) -> list[PaperDocument]:
        deduped: dict[str, PaperDocument] = {}
        for document in documents:
            deduped[document.metadata.paper_id] = document
        return list(deduped.values())
