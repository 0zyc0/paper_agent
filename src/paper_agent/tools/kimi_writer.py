from __future__ import annotations

import json

from .llm import KimiClient
from ..core.models import Paper


class KimiRelatedWorkWriter:
    def __init__(self, llm: KimiClient | None = None) -> None:
        self.llm = llm or KimiClient()

    @property
    def available(self) -> bool:
        return self.llm.available

    def draft(
        self,
        *,
        topic: str,
        papers: list[Paper],
        citation_keys: dict[str, str],
        language: str,
        writing_request: str | None = None,
        writing_plan: dict | None = None,
        evidence_notes: list[dict] | None = None,
    ) -> str:
        evidence_by_id = {
            str(item.get("paper_id") or ""): item
            for item in evidence_notes or []
            if isinstance(item, dict)
        }
        records = []
        for paper in papers:
            key = citation_keys[paper.id or paper.title]
            evidence = evidence_by_id.get(paper.id or "") or {}
            records.append(
                {
                    "citation_key": key,
                    "title": paper.title,
                    "authors": paper.authors[:8],
                    "year": paper.year,
                    "venue": paper.venue,
                    "venue_rank": paper.venue_rank,
                    "source": paper.source,
                    "url": paper.source_url,
                    "abstract": (paper.abstract or "")[:1200],
                    "evidence_level": evidence.get("evidence_level") or "metadata_or_abstract",
                    "local_fulltext_status": evidence.get("fulltext_status") or paper.fulltext_status,
                    "local_fulltext_excerpts": evidence.get("excerpts") or [],
                    "manual_upload_needed": bool(evidence.get("manual_upload_needed")),
                    "evidence_note": evidence.get("note") or "",
                }
            )
        system = (
            "You are a careful CS research writing assistant. "
            "Write only from the provided paper records. "
            "Never invent citations, papers, authors, venues, results, or years. "
            "Use LaTeX citations exactly like \\cite{citation_key}. "
            "Prioritize local_fulltext_excerpts when they are present. "
            "If a paper has only metadata or abstract evidence, use it only for high-level positioning and say it needs full-text verification when making detailed claims. "
            "Output a clean Markdown manuscript section, not a bibliography, retrieved-source list, BibTeX, URL list, or numbered catalog. "
            "Use the requested section title and concise level-3 subsections when the writing plan provides them."
        )
        target_language = "Chinese" if language.lower().startswith("zh") else "English"
        user = f"""
Write a preliminary research writing draft in {target_language}.

Research topic:
{topic}

User writing request:
{writing_request or "Write the section requested by the user. If no section is specified, write a survey-style synthesis."}

Writing plan:
{writing_plan or {}}

Allowed paper records:
{json.dumps(records, ensure_ascii=False)}

Requirements:
- Start with one level-2 Markdown heading matching the writing plan title exactly.
- Write 3 to 5 substantive academic paragraphs; use the writing-plan outline as level-3 subsections when it is useful.
- Adapt the content to the requested output type. A survey must include a taxonomy/comparison perspective; an introduction must distinguish background, limitation, and study objective; a method or experiment section must not invent an implementation or result that is absent from the evidence.
- For a survey, identify concrete method families from the supplied records, compare at least two families by their modelling focus or evidence scope, and close with specific open verification questions. Do not repeat the user's request as the topic or reuse generic sentences across subsections.
- Every factual paragraph must include citations from the allowed citation_key values. Multiple keys may be written in one \\cite{{key1,key2}} command.
- Do not cite anything outside the allowed paper records.
- Do not output "Retrieved Sources", "References", paper metadata lists, URLs, or BibTeX; those are generated separately.
- Do not simply enumerate papers. Synthesize them into connected academic paragraphs.
- Prefer claims supported by local_fulltext_excerpts. Keep abstract-only or metadata-only papers as weak supporting context.
- Do not use generic filler such as "the field is rapidly evolving" unless the provided records make the trend concrete.
- Do not mention internal system terms such as metadata, evidence pool, local_fulltext_excerpts, or manual_upload_needed in the manuscript prose. Instead, make claims conservative whenever the supporting record is weak.
"""
        return self.llm.chat_text(
            system=system,
            user=user,
            temperature=0.25,
            max_tokens=3200,
            label="related_work_writer",
        )
