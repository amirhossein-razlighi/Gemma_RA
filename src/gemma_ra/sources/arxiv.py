from __future__ import annotations

from datetime import datetime
from xml.etree import ElementTree

import httpx

from gemma_ra.core.config import ArxivConfig
from gemma_ra.core.exceptions import SourceError
from gemma_ra.core.schemas import PaperDocument, PaperMetadata

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivPaperSource:
    def __init__(self, config: ArxivConfig) -> None:
        self.config = config

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
        surname = professor.split()[-1] if professor.split() else professor
        queries: list[tuple[str, str]] = []
        if topic:
            queries.append(("author+topic", f'au:"{professor}" AND all:"{topic}"'))
        queries.append(("author-only", f'au:"{professor}"'))
        if topic and surname and surname.lower() != professor.lower():
            queries.append(("surname+topic", f'au:"{surname}" AND all:"{topic}"'))
        if surname and surname.lower() != professor.lower():
            queries.append(("surname-only", f'au:"{surname}"'))
        if topic:
            queries.append(("fulltext-name+topic", f'all:"{professor}" AND all:"{topic}"'))
        queries.append(("fulltext-name", f'all:"{professor}"'))

        combined: list[PaperDocument] = []
        notes: list[str] = []
        for label, query in queries:
            docs = self.search_query(query)
            notes.append(f'arXiv search "{label}" for "{professor}" returned {len(docs)} result(s).')
            combined.extend(docs)
            if docs and label in {"author+topic", "author-only"}:
                break

        deduped = self._dedupe(combined)
        if not deduped:
            notes.append(
                f'No arXiv papers matched professor "{professor}"'
                + (f' with topic "{topic}".' if topic else ".")
            )
        return deduped, notes

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
