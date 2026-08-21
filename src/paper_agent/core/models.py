from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import hashlib
import re
from typing import Any


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def stable_paper_id(
    title: str,
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    source_url: str | None = None,
) -> str:
    if doi:
        key = f"doi:{doi.lower().strip()}"
    elif arxiv_id:
        key = f"arxiv:{arxiv_id.lower().strip()}"
    elif source_url:
        key = f"url:{source_url.lower().strip()}"
    else:
        key = f"title:{normalize_text(title).lower()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class Paper:
    title: str
    authors: list[str]
    abstract: str | None = None
    year: int | None = None
    published_at: str | None = None
    venue: str | None = None
    source: str = "unknown"
    source_url: str | None = None
    pdf_url: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    citation_count: int | None = None
    reference_count: int | None = None
    fields_of_study: list[str] = field(default_factory=list)
    venue_rank: str | None = None
    venue_reason: str | None = None
    abstract_status: str = "none"
    local_pdf_path: str | None = None
    local_text_path: str | None = None
    fulltext_status: str = "none"
    fulltext_error: str | None = None
    fulltext_sha256: str | None = None
    fulltext_downloaded_at: str | None = None
    reading_status: str = "unread"
    importance: str = "normal"
    user_tags: list[str] = field(default_factory=list)
    excluded: bool = False
    exclusion_reason: str | None = None
    user_notes: str | None = None
    used_in_sections: list[str] = field(default_factory=list)
    relevance_score: float | None = None
    is_verified: bool = False
    id: str | None = None
    added_at: str = field(default_factory=lambda: utc_now_iso())
    updated_at: str | None = None
    retrieved_at: str = field(default_factory=lambda: utc_now_iso())

    def __post_init__(self) -> None:
        self.title = normalize_text(self.title)
        self.abstract = normalize_text(self.abstract)
        self.authors = [normalize_text(author) for author in self.authors if normalize_text(author)]
        self.reading_status = _normalize_choice(self.reading_status, {"unread", "to_read", "reading", "read"}, "unread")
        self.importance = _normalize_choice(self.importance, {"low", "normal", "high", "core"}, "normal")
        self.user_tags = _clean_unique_list(self.user_tags)
        self.used_in_sections = _clean_unique_list(self.used_in_sections)
        self.exclusion_reason = normalize_text(self.exclusion_reason)
        self.user_notes = normalize_text(self.user_notes)
        self.excluded = bool(self.excluded)
        self.abstract_status = self.abstract_status or ("available" if self.abstract else "none")
        if self.abstract:
            self.abstract_status = "available"
        self.fulltext_status = self.fulltext_status or "none"
        if self.published_at and not self.year:
            self.year = _year_from_date(self.published_at)
        self.is_verified = bool(self.doi or self.arxiv_id or self.source_url)
        self.updated_at = self.updated_at or self.added_at
        if not self.id:
            self.id = stable_paper_id(
                self.title,
                doi=self.doi,
                arxiv_id=self.arxiv_id,
                source_url=self.source_url,
            )

    @property
    def citation_label(self) -> str:
        first_author = self.authors[0].split()[-1] if self.authors else "Unknown"
        year = self.year or "n.d."
        return f"{first_author} et al., {year}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Paper":
        return cls(**data)


@dataclass
class SearchResult:
    query: str
    papers: list[Paper]
    sources: list[str]
    source_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    searched_at: str = field(default_factory=lambda: utc_now_iso())


@dataclass
class RelatedWorkDraft:
    title: str
    query: str
    content_markdown: str
    bibtex: str
    paper_ids: list[str]
    claim_map: list[dict[str, Any]]
    writing_kind: str = "related_work"
    outline: list[str] = field(default_factory=list)
    quality_report: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: utc_now_iso())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _year_from_date(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10]).year
    except ValueError:
        match = re.search(r"\b(19|20)\d{2}\b", value)
        return int(match.group(0)) if match else None


def _normalize_choice(value: str | None, allowed: set[str], default: str) -> str:
    cleaned = normalize_text(value).lower().replace("-", "_").replace(" ", "_")
    return cleaned if cleaned in allowed else default


def _clean_unique_list(values: list[str] | None) -> list[str]:
    cleaned: list[str] = []
    for value in values or []:
        item = normalize_text(str(value))
        if item and item not in cleaned:
            cleaned.append(item)
    return cleaned
