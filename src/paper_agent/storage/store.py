from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3

from ..core.models import Paper, utc_now_iso
from ..tools.search import dedupe_papers, merge_papers


class SQLitePaperStore:
    def __init__(self, path: str | Path = "data/papers.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def load_papers(self) -> list[Paper]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM papers ORDER BY COALESCE(year, 0) DESC, retrieved_at DESC").fetchall()
        return [self._paper_from_row(row) for row in rows]

    def save_papers(self, papers: list[Paper], *, query: str | None = None, topic: str | None = None) -> None:
        if not papers:
            return
        now = utc_now_iso()
        with self._connect() as conn:
            for paper in dedupe_papers(papers):
                existing = self._find_existing_paper(conn, paper)
                if existing:
                    paper = merge_papers(existing, paper)
                    paper.id = existing.id
                conn.execute(
                    """
                    INSERT INTO papers (
                        id, title, authors_json, abstract, year, published_at, venue, source,
                        source_url, pdf_url, doi, arxiv_id, citation_count, reference_count,
                        fields_json, venue_rank, venue_reason, abstract_status, local_pdf_path,
                        local_text_path, fulltext_status, fulltext_error, fulltext_sha256,
                        fulltext_downloaded_at, reading_status, importance, user_tags_json,
                        excluded, exclusion_reason, user_notes, used_in_sections_json,
                        relevance_score, is_verified, added_at, updated_at, retrieved_at,
                        last_seen_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title = excluded.title,
                        authors_json = excluded.authors_json,
                        abstract = COALESCE(NULLIF(excluded.abstract, ''), papers.abstract),
                        year = COALESCE(excluded.year, papers.year),
                        published_at = COALESCE(excluded.published_at, papers.published_at),
                        venue = COALESCE(excluded.venue, papers.venue),
                        source = excluded.source,
                        source_url = COALESCE(excluded.source_url, papers.source_url),
                        pdf_url = COALESCE(excluded.pdf_url, papers.pdf_url),
                        doi = COALESCE(excluded.doi, papers.doi),
                        arxiv_id = COALESCE(excluded.arxiv_id, papers.arxiv_id),
                        citation_count = MAX(COALESCE(excluded.citation_count, 0), COALESCE(papers.citation_count, 0)),
                        reference_count = COALESCE(excluded.reference_count, papers.reference_count),
                        fields_json = excluded.fields_json,
                        venue_rank = COALESCE(excluded.venue_rank, papers.venue_rank),
                        venue_reason = COALESCE(excluded.venue_reason, papers.venue_reason),
                        abstract_status = CASE
                            WHEN COALESCE(NULLIF(excluded.abstract, ''), papers.abstract) IS NOT NULL
                                 AND COALESCE(NULLIF(excluded.abstract, ''), papers.abstract) != '' THEN 'available'
                            ELSE COALESCE(NULLIF(excluded.abstract_status, ''), papers.abstract_status, 'none')
                        END,
                        local_pdf_path = COALESCE(excluded.local_pdf_path, papers.local_pdf_path),
                        local_text_path = COALESCE(excluded.local_text_path, papers.local_text_path),
                        fulltext_status = CASE
                            WHEN excluded.fulltext_status IS NULL OR excluded.fulltext_status = 'none' THEN papers.fulltext_status
                            ELSE excluded.fulltext_status
                        END,
                        fulltext_error = COALESCE(excluded.fulltext_error, papers.fulltext_error),
                        fulltext_sha256 = COALESCE(excluded.fulltext_sha256, papers.fulltext_sha256),
                        fulltext_downloaded_at = COALESCE(excluded.fulltext_downloaded_at, papers.fulltext_downloaded_at),
                        reading_status = COALESCE(NULLIF(excluded.reading_status, 'unread'), papers.reading_status, 'unread'),
                        importance = COALESCE(NULLIF(excluded.importance, 'normal'), papers.importance, 'normal'),
                        user_tags_json = CASE
                            WHEN papers.user_tags_json IS NOT NULL AND papers.user_tags_json != '[]' THEN papers.user_tags_json
                            ELSE excluded.user_tags_json
                        END,
                        excluded = excluded.excluded OR papers.excluded,
                        exclusion_reason = COALESCE(papers.exclusion_reason, excluded.exclusion_reason),
                        user_notes = COALESCE(papers.user_notes, excluded.user_notes),
                        used_in_sections_json = CASE
                            WHEN papers.used_in_sections_json IS NOT NULL AND papers.used_in_sections_json != '[]' THEN papers.used_in_sections_json
                            ELSE excluded.used_in_sections_json
                        END,
                        relevance_score = COALESCE(excluded.relevance_score, papers.relevance_score),
                        is_verified = excluded.is_verified OR papers.is_verified,
                        added_at = papers.added_at,
                        updated_at = excluded.updated_at,
                        last_seen_at = excluded.last_seen_at
                    """,
                    self._paper_values(paper, now),
                )
                if query or topic:
                    conn.execute(
                        """
                        INSERT INTO paper_queries (paper_id, query, topic, source, searched_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (paper.id, query or "", topic or "", paper.source, now),
                    )

    def upsert_and_return(self, papers: list[Paper]) -> list[Paper]:
        self.save_papers(papers)
        wanted_keys = set()
        for paper in papers:
            wanted_keys.update(_paper_identity_keys(paper))
        return [
            paper
            for paper in self.load_papers()
            if set(_paper_identity_keys(paper)) & wanted_keys
        ]

    def update_paper_asset_state(
        self,
        paper_id: str,
        *,
        reading_status: str | None = None,
        importance: str | None = None,
        user_tags: list[str] | None = None,
        excluded: bool | None = None,
        exclusion_reason: str | None = None,
        user_notes: str | None = None,
        used_in_sections: list[str] | None = None,
        relevance_score: float | None = None,
    ) -> Paper | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
            if not row:
                return None
            paper = self._paper_from_row(row)
            if reading_status is not None:
                paper.reading_status = reading_status
            if importance is not None:
                paper.importance = importance
            if user_tags is not None:
                paper.user_tags = user_tags
            if excluded is not None:
                paper.excluded = excluded
            if exclusion_reason is not None:
                paper.exclusion_reason = exclusion_reason
            if user_notes is not None:
                paper.user_notes = user_notes
            if used_in_sections is not None:
                paper.used_in_sections = used_in_sections
            if relevance_score is not None:
                paper.relevance_score = max(0.0, min(float(relevance_score), 1.0))
            paper.__post_init__()
            paper.updated_at = utc_now_iso()
            conn.execute(
                """
                UPDATE papers SET
                    reading_status = ?, importance = ?, user_tags_json = ?, excluded = ?,
                    exclusion_reason = ?, user_notes = ?, used_in_sections_json = ?,
                    relevance_score = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    paper.reading_status,
                    paper.importance,
                    json.dumps(paper.user_tags, ensure_ascii=False),
                    1 if paper.excluded else 0,
                    paper.exclusion_reason,
                    paper.user_notes,
                    json.dumps(paper.used_in_sections, ensure_ascii=False),
                    paper.relevance_score,
                    paper.updated_at,
                    paper.id,
                ),
            )
            return paper

    def dedupe_existing(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM papers").fetchall()
            papers = [self._paper_from_row(row) for row in rows]
            before = len(papers)
            if before <= 1:
                return {"before": before, "after": before, "merged": 0}

            parent = {paper.id: paper.id for paper in papers if paper.id}
            key_owner: dict[str, str] = {}

            def find(item_id: str) -> str:
                while parent[item_id] != item_id:
                    parent[item_id] = parent[parent[item_id]]
                    item_id = parent[item_id]
                return item_id

            def union(left: str, right: str) -> None:
                left_root = find(left)
                right_root = find(right)
                if left_root != right_root:
                    parent[right_root] = left_root

            for paper in papers:
                if not paper.id:
                    continue
                for key in _paper_identity_keys(paper):
                    owner = key_owner.get(key)
                    if owner:
                        union(owner, paper.id)
                    else:
                        key_owner[key] = paper.id

            groups: dict[str, list[Paper]] = {}
            for paper in papers:
                if paper.id:
                    groups.setdefault(find(paper.id), []).append(paper)

            merged_count = 0
            now = utc_now_iso()
            for group in groups.values():
                if len(group) <= 1:
                    continue
                ordered = sorted(group, key=_paper_quality_score, reverse=True)
                canonical = ordered[0]
                duplicate_ids = [paper.id for paper in ordered[1:] if paper.id and paper.id != canonical.id]
                for duplicate in ordered[1:]:
                    canonical = merge_papers(canonical, duplicate)
                    canonical.id = ordered[0].id
                canonical.source = _merge_sources(*(paper.source for paper in ordered))
                conn.execute(
                    """
                    UPDATE papers SET
                        title = ?, authors_json = ?, abstract = ?, year = ?, published_at = ?,
                        venue = ?, source = ?, source_url = ?, pdf_url = ?, doi = ?,
                        arxiv_id = ?, citation_count = ?, reference_count = ?, fields_json = ?,
                        venue_rank = ?, venue_reason = ?, abstract_status = ?, local_pdf_path = ?,
                        local_text_path = ?, fulltext_status = ?, fulltext_error = ?,
                        fulltext_sha256 = ?, fulltext_downloaded_at = ?, reading_status = ?,
                        importance = ?, user_tags_json = ?, excluded = ?, exclusion_reason = ?,
                        user_notes = ?, used_in_sections_json = ?, relevance_score = ?,
                        is_verified = ?, added_at = ?, updated_at = ?, last_seen_at = ?
                    WHERE id = ?
                    """,
                    (
                        canonical.title,
                        json.dumps(canonical.authors, ensure_ascii=False),
                        canonical.abstract or "",
                        canonical.year,
                        canonical.published_at,
                        canonical.venue,
                        canonical.source,
                        canonical.source_url,
                        canonical.pdf_url,
                        canonical.doi,
                        canonical.arxiv_id,
                        canonical.citation_count,
                        canonical.reference_count,
                        json.dumps(canonical.fields_of_study, ensure_ascii=False),
                        canonical.venue_rank,
                        canonical.venue_reason,
                        canonical.abstract_status or ("available" if canonical.abstract else "none"),
                        canonical.local_pdf_path,
                        canonical.local_text_path,
                        canonical.fulltext_status or "none",
                        canonical.fulltext_error,
                        canonical.fulltext_sha256,
                        canonical.fulltext_downloaded_at,
                        canonical.reading_status,
                        canonical.importance,
                        json.dumps(canonical.user_tags, ensure_ascii=False),
                        1 if canonical.excluded else 0,
                        canonical.exclusion_reason,
                        canonical.user_notes,
                        json.dumps(canonical.used_in_sections, ensure_ascii=False),
                        canonical.relevance_score,
                        1 if canonical.is_verified else 0,
                        canonical.added_at,
                        now,
                        now,
                        canonical.id,
                    ),
                )
                for duplicate_id in duplicate_ids:
                    # Generated claim maps may keep the original paper ID.
                    # Retain an alias before removing the duplicate so future
                    # lookups can still resolve that provenance reference.
                    conn.execute(
                        """
                        INSERT INTO paper_id_aliases (alias_id, paper_id)
                        VALUES (?, ?)
                        ON CONFLICT(alias_id) DO UPDATE SET paper_id = excluded.paper_id
                        """,
                        (duplicate_id, canonical.id),
                    )
                    conn.execute("UPDATE paper_id_aliases SET paper_id = ? WHERE paper_id = ?", (canonical.id, duplicate_id))
                    conn.execute("UPDATE paper_queries SET paper_id = ? WHERE paper_id = ?", (canonical.id, duplicate_id))
                    conn.execute("DELETE FROM papers WHERE id = ?", (duplicate_id,))
                merged_count += len(duplicate_ids)

            after = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        return {"before": before, "after": after, "merged": merged_count}

    def resolve_paper_id(self, paper_id: str | None) -> str | None:
        """Return the canonical ID for a current or previously merged paper."""
        if not paper_id:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM papers WHERE id = ?", (paper_id,)).fetchone()
            if row:
                return str(row["id"])
            row = conn.execute("SELECT paper_id FROM paper_id_aliases WHERE alias_id = ?", (paper_id,)).fetchone()
            return str(row["paper_id"]) if row else None

    def search_cached(
        self,
        query: str,
        *,
        from_year: int | None = None,
        to_year: int | None = None,
        limit: int = 60,
    ) -> list[Paper]:
        candidates = [
            paper
            for paper in self.load_papers()
            if _year_in_range(paper.year, from_year, to_year) and not paper.excluded
        ]
        scored = [(paper, _cache_score(paper, query)) for paper in candidates]
        relevant = [(paper, score) for paper, score in scored if score > 0]
        return [paper for paper, _ in sorted(relevant, key=lambda item: item[1], reverse=True)[:limit]]

    def stats(self) -> dict:
        with self._connect() as conn:
            paper_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            abstract_count = conn.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL AND abstract != ''").fetchone()[0]
            fulltext_count = conn.execute("SELECT COUNT(*) FROM papers WHERE fulltext_status = 'extracted'").fetchone()[0]
            to_read_count = conn.execute("SELECT COUNT(*) FROM papers WHERE reading_status = 'to_read'").fetchone()[0]
            read_count = conn.execute("SELECT COUNT(*) FROM papers WHERE reading_status = 'read'").fetchone()[0]
            core_count = conn.execute("SELECT COUNT(*) FROM papers WHERE importance = 'core'").fetchone()[0]
            excluded_count = conn.execute("SELECT COUNT(*) FROM papers WHERE excluded = 1").fetchone()[0]
            query_count = conn.execute("SELECT COUNT(*) FROM paper_queries").fetchone()[0]
        return {
            "paper_count": paper_count,
            "abstract_count": abstract_count,
            "fulltext_count": fulltext_count,
            "to_read_count": to_read_count,
            "read_count": read_count,
            "core_count": core_count,
            "excluded_count": excluded_count,
            "query_count": query_count,
        }

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors_json TEXT NOT NULL,
                    abstract TEXT,
                    year INTEGER,
                    published_at TEXT,
                    venue TEXT,
                    source TEXT NOT NULL,
                    source_url TEXT,
                    pdf_url TEXT,
                    doi TEXT,
                    arxiv_id TEXT,
                    citation_count INTEGER,
                    reference_count INTEGER,
                    fields_json TEXT NOT NULL,
                    venue_rank TEXT,
                    venue_reason TEXT,
                    abstract_status TEXT NOT NULL DEFAULT 'none',
                    local_pdf_path TEXT,
                    local_text_path TEXT,
                    fulltext_status TEXT NOT NULL DEFAULT 'none',
                    fulltext_error TEXT,
                    fulltext_sha256 TEXT,
                    fulltext_downloaded_at TEXT,
                    reading_status TEXT NOT NULL DEFAULT 'unread',
                    importance TEXT NOT NULL DEFAULT 'normal',
                    user_tags_json TEXT NOT NULL DEFAULT '[]',
                    excluded INTEGER NOT NULL DEFAULT 0,
                    exclusion_reason TEXT,
                    user_notes TEXT,
                    used_in_sections_json TEXT NOT NULL DEFAULT '[]',
                    relevance_score REAL,
                    is_verified INTEGER NOT NULL,
                    added_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                )
                """
            )
            self._ensure_paper_columns(conn)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_venue ON papers(venue)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_fulltext_status ON papers(fulltext_status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_reading_status ON papers(reading_status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_importance ON papers(importance)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_excluded ON papers(excluded)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    source TEXT NOT NULL,
                    searched_at TEXT NOT NULL,
                    FOREIGN KEY(paper_id) REFERENCES papers(id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_queries_topic ON paper_queries(topic)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_queries_paper_id ON paper_queries(paper_id)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_id_aliases (
                    alias_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    FOREIGN KEY(paper_id) REFERENCES papers(id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_aliases_canonical ON paper_id_aliases(paper_id)")

    def _ensure_paper_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(papers)").fetchall()}
        migrations = {
            "abstract_status": "ALTER TABLE papers ADD COLUMN abstract_status TEXT NOT NULL DEFAULT 'none'",
            "local_pdf_path": "ALTER TABLE papers ADD COLUMN local_pdf_path TEXT",
            "local_text_path": "ALTER TABLE papers ADD COLUMN local_text_path TEXT",
            "fulltext_status": "ALTER TABLE papers ADD COLUMN fulltext_status TEXT NOT NULL DEFAULT 'none'",
            "fulltext_error": "ALTER TABLE papers ADD COLUMN fulltext_error TEXT",
            "fulltext_sha256": "ALTER TABLE papers ADD COLUMN fulltext_sha256 TEXT",
            "fulltext_downloaded_at": "ALTER TABLE papers ADD COLUMN fulltext_downloaded_at TEXT",
            "reading_status": "ALTER TABLE papers ADD COLUMN reading_status TEXT NOT NULL DEFAULT 'unread'",
            "importance": "ALTER TABLE papers ADD COLUMN importance TEXT NOT NULL DEFAULT 'normal'",
            "user_tags_json": "ALTER TABLE papers ADD COLUMN user_tags_json TEXT NOT NULL DEFAULT '[]'",
            "excluded": "ALTER TABLE papers ADD COLUMN excluded INTEGER NOT NULL DEFAULT 0",
            "exclusion_reason": "ALTER TABLE papers ADD COLUMN exclusion_reason TEXT",
            "user_notes": "ALTER TABLE papers ADD COLUMN user_notes TEXT",
            "used_in_sections_json": "ALTER TABLE papers ADD COLUMN used_in_sections_json TEXT NOT NULL DEFAULT '[]'",
            "relevance_score": "ALTER TABLE papers ADD COLUMN relevance_score REAL",
            "added_at": "ALTER TABLE papers ADD COLUMN added_at TEXT",
            "updated_at": "ALTER TABLE papers ADD COLUMN updated_at TEXT",
        }
        for column, statement in migrations.items():
            if column not in existing:
                conn.execute(statement)
        conn.execute(
            """
            UPDATE papers
            SET abstract_status = CASE
                WHEN abstract IS NOT NULL AND abstract != '' THEN 'available'
                ELSE COALESCE(NULLIF(abstract_status, ''), 'none')
            END
            """
        )
        now = utc_now_iso()
        conn.execute("UPDATE papers SET added_at = COALESCE(added_at, retrieved_at, ?)", (now,))
        conn.execute("UPDATE papers SET updated_at = COALESCE(updated_at, last_seen_at, retrieved_at, ?)", (now,))

    def _paper_values(self, paper: Paper, now: str) -> tuple:
        return (
            paper.id,
            paper.title,
            json.dumps(paper.authors, ensure_ascii=False),
            paper.abstract or "",
            paper.year,
            paper.published_at,
            paper.venue,
            paper.source,
            paper.source_url,
            paper.pdf_url,
            paper.doi,
            paper.arxiv_id,
            paper.citation_count,
            paper.reference_count,
            json.dumps(paper.fields_of_study, ensure_ascii=False),
            paper.venue_rank,
            paper.venue_reason,
            paper.abstract_status or ("available" if paper.abstract else "none"),
            paper.local_pdf_path,
            paper.local_text_path,
            paper.fulltext_status or "none",
            paper.fulltext_error,
            paper.fulltext_sha256,
            paper.fulltext_downloaded_at,
            paper.reading_status,
            paper.importance,
            json.dumps(paper.user_tags, ensure_ascii=False),
            1 if paper.excluded else 0,
            paper.exclusion_reason,
            paper.user_notes,
            json.dumps(paper.used_in_sections, ensure_ascii=False),
            paper.relevance_score,
            1 if paper.is_verified else 0,
            paper.added_at or now,
            paper.updated_at or now,
            paper.retrieved_at or now,
            now,
        )

    def _paper_from_row(self, row: sqlite3.Row) -> Paper:
        return Paper(
            id=row["id"],
            title=row["title"],
            authors=json.loads(row["authors_json"] or "[]"),
            abstract=row["abstract"] or None,
            year=row["year"],
            published_at=row["published_at"],
            venue=row["venue"],
            source=row["source"],
            source_url=row["source_url"],
            pdf_url=row["pdf_url"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            citation_count=row["citation_count"],
            reference_count=row["reference_count"],
            fields_of_study=json.loads(row["fields_json"] or "[]"),
            venue_rank=row["venue_rank"],
            venue_reason=row["venue_reason"],
            abstract_status=row["abstract_status"] or ("available" if row["abstract"] else "none"),
            local_pdf_path=row["local_pdf_path"],
            local_text_path=row["local_text_path"],
            fulltext_status=row["fulltext_status"] or "none",
            fulltext_error=row["fulltext_error"],
            fulltext_sha256=row["fulltext_sha256"],
            fulltext_downloaded_at=row["fulltext_downloaded_at"],
            reading_status=row["reading_status"] or "unread",
            importance=row["importance"] or "normal",
            user_tags=json.loads(row["user_tags_json"] or "[]"),
            excluded=bool(row["excluded"]),
            exclusion_reason=row["exclusion_reason"],
            user_notes=row["user_notes"],
            used_in_sections=json.loads(row["used_in_sections_json"] or "[]"),
            relevance_score=row["relevance_score"],
            is_verified=bool(row["is_verified"]),
            added_at=row["added_at"] or row["retrieved_at"],
            updated_at=row["updated_at"] or row["last_seen_at"] or row["retrieved_at"],
            retrieved_at=row["retrieved_at"],
        )

    def _find_existing_paper(self, conn: sqlite3.Connection, paper: Paper) -> Paper | None:
        keys = _paper_identity_keys(paper)
        if not keys:
            return None
        rows = conn.execute("SELECT * FROM papers").fetchall()
        best_match: Paper | None = None
        best_score = -1.0
        incoming_keys = set(keys)
        for row in rows:
            candidate = self._paper_from_row(row)
            if not incoming_keys.intersection(_paper_identity_keys(candidate)) and not _same_author_year_title(paper, candidate):
                continue
            score = _paper_quality_score(candidate)
            if score > best_score:
                best_match = candidate
                best_score = score
        return best_match


class JsonPaperStore:
    def __init__(self, path: str | Path = "data/papers.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_papers(self) -> list[Paper]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return [Paper.from_dict(item) for item in data.get("papers", [])]

    def save_papers(self, papers: list[Paper]) -> None:
        merged = dedupe_papers([*self.load_papers(), *papers])
        payload = {"papers": [paper.to_dict() for paper in merged]}
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(self.path)

    def upsert_and_return(self, papers: list[Paper]) -> list[Paper]:
        self.save_papers(papers)
        saved = self.load_papers()
        wanted_ids = {paper.id for paper in papers}
        return [paper for paper in saved if paper.id in wanted_ids]


def _cache_score(paper: Paper, query: str) -> float:
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return 0.0
    title_tokens = set(_tokens(paper.title))
    abstract_tokens = set(_tokens(paper.abstract or ""))
    venue_tokens = set(_tokens(paper.venue or ""))
    overlap = query_tokens & (title_tokens | abstract_tokens | venue_tokens)
    if not overlap:
        return 0.0
    return 4.0 * len(overlap & title_tokens) + 1.5 * len(overlap & abstract_tokens) + len(overlap & venue_tokens)


def _tokens(value: str) -> list[str]:
    stopwords = {"paper", "papers", "study", "related", "work", "survey", "method", "methods", "system", "systems"}
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", value.lower())
        if len(token) > 2 and token not in stopwords
    ]


def _year_in_range(year: int | None, from_year: int | None, to_year: int | None) -> bool:
    if year is None:
        return True
    if from_year is not None and year < from_year:
        return False
    if to_year is not None and year > to_year:
        return False
    return True


def _paper_identity_keys(paper: Paper) -> list[str]:
    keys: list[str] = []
    doi = _normalize_doi(paper.doi)
    arxiv_id = _normalize_arxiv_id(paper.arxiv_id)
    title = _normalize_title_key(paper.title)
    author_year = _normalize_author_year_key(paper)
    if doi:
        keys.append(f"doi:{doi}")
    if arxiv_id:
        keys.append(f"arxiv:{arxiv_id}")
    if title:
        keys.append(f"title:{title}")
    if title and author_year:
        keys.append(f"author_year_title:{author_year}:{title}")
    return keys


def _normalize_doi(value: str | None) -> str:
    value = normalize_text_for_identity(value)
    if value.startswith("https://doi.org/"):
        value = value.removeprefix("https://doi.org/")
    if value.startswith("http://dx.doi.org/"):
        value = value.removeprefix("http://dx.doi.org/")
    return value.strip().lower().rstrip(".")


def _normalize_arxiv_id(value: str | None) -> str:
    value = normalize_text_for_identity(value)
    value = value.removeprefix("arxiv:")
    value = value.removeprefix("https://arxiv.org/abs/")
    value = value.removeprefix("http://arxiv.org/abs/")
    value = re.sub(r"v\d+$", "", value)
    return value.strip().lower().rstrip(".")


def _normalize_title_key(value: str | None) -> str:
    normalized = normalize_text_for_identity(value)
    normalized = re.sub(r"[\u2010-\u2015]", "-", normalized)
    return re.sub(r"[^a-z0-9]+", "", normalized.lower())


def _normalize_author_year_key(paper: Paper) -> str:
    if not paper.year or not paper.authors:
        return ""
    author_bits = []
    for author in paper.authors[:3]:
        normalized = re.sub(r"[^a-z0-9]+", "", normalize_text_for_identity(author).lower())
        if normalized:
            author_bits.append(normalized)
    if not author_bits:
        return ""
    return f"{paper.year}:{'-'.join(author_bits)}"


def _same_author_year_title(left: Paper, right: Paper) -> bool:
    if not left.year or left.year != right.year:
        return False
    if not _share_first_author(left, right):
        return False
    return _titles_match(left.title, right.title)


def _share_first_author(left: Paper, right: Paper) -> bool:
    if not left.authors or not right.authors:
        return False
    left_key = re.sub(r"[^a-z0-9]+", "", normalize_text_for_identity(left.authors[0]).lower())
    right_key = re.sub(r"[^a-z0-9]+", "", normalize_text_for_identity(right.authors[0]).lower())
    return bool(left_key and right_key and left_key == right_key)


def _titles_match(left: str, right: str) -> bool:
    left_key = _normalize_title_key(left)
    right_key = _normalize_title_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    shorter, longer = sorted([left_key, right_key], key=len)
    return len(shorter) >= 24 and shorter in longer


def normalize_text_for_identity(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _paper_quality_score(paper: Paper) -> float:
    score = 0.0
    if paper.fulltext_status == "extracted":
        score += 1000
    if paper.local_pdf_path:
        score += 500
    if paper.abstract:
        score += 220 + min(len(paper.abstract), 2500) / 100
    if paper.doi:
        score += 140
    if paper.arxiv_id:
        score += 120
    if paper.source_url:
        score += 40
    if paper.pdf_url:
        score += 35
    if paper.citation_count:
        score += min(paper.citation_count, 1000) / 20
    if paper.venue:
        score += 20
    if paper.authors:
        score += min(len(paper.authors), 10)
    return score


def _merge_sources(*values: str | None) -> str:
    sources: list[str] = []
    for value in values:
        for source in str(value or "").split(","):
            source = source.strip()
            if source and source not in sources:
                sources.append(source)
    return ",".join(sources) or "unknown"
