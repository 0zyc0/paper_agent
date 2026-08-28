from __future__ import annotations

from datetime import datetime, timezone
import html
import os
import re
import threading
import time
import xml.etree.ElementTree as ET

from .http_client import HttpError, get_json, get_text, with_query
from ..core.models import Paper, SearchResult, normalize_text

try:
    from .. import local_config
except ImportError:  # pragma: no cover - local config is optional
    local_config = None


# Literature search is an interactive path. A temporarily unavailable public
# source must not make the whole UI wait for several retry cycles.
SEARCH_HTTP_TIMEOUT = 8


class ArxivClient:
    api_url = "https://export.arxiv.org/api/query"
    atom_ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    def search(self, query: str, *, limit: int = 20, sort_by: str = "submittedDate") -> list[Paper]:
        search_query = _build_arxiv_query(query)
        url = with_query(
            self.api_url,
            {
                "search_query": search_query,
                "start": 0,
                "max_results": limit,
                "sortBy": sort_by,
                "sortOrder": "descending",
            },
        )
        xml_text = get_text(url, timeout=SEARCH_HTTP_TIMEOUT, retries=0)
        root = ET.fromstring(xml_text)
        papers: list[Paper] = []
        for entry in root.findall("atom:entry", self.atom_ns):
            title = _node_text(entry, "atom:title", self.atom_ns)
            abstract = _node_text(entry, "atom:summary", self.atom_ns)
            published_at = _node_text(entry, "atom:published", self.atom_ns)[:10] or None
            source_url = _node_text(entry, "atom:id", self.atom_ns)
            arxiv_id = _extract_arxiv_id(source_url)
            authors = [
                normalize_text(_node_text(author, "atom:name", self.atom_ns))
                for author in entry.findall("atom:author", self.atom_ns)
            ]
            doi = _node_text(entry, "arxiv:doi", self.atom_ns) or None
            categories = [
                category.attrib.get("term", "")
                for category in entry.findall("atom:category", self.atom_ns)
                if category.attrib.get("term")
            ]
            pdf_url = None
            for link in entry.findall("atom:link", self.atom_ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href")
                    break
            papers.append(
                Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    published_at=published_at,
                    venue="arXiv",
                    source="arxiv",
                    source_url=source_url,
                    pdf_url=pdf_url,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    fields_of_study=categories,
                )
            )
        if papers:
            return papers
        if search_query != f"all:{query}":
            return self._fallback_search(query, limit=limit, sort_by=sort_by)
        return papers

    def _fallback_search(self, query: str, *, limit: int, sort_by: str) -> list[Paper]:
        url = with_query(
            self.api_url,
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit,
                "sortBy": sort_by,
                "sortOrder": "descending",
            },
        )
        xml_text = get_text(url, timeout=SEARCH_HTTP_TIMEOUT, retries=0)
        root = ET.fromstring(xml_text)
        papers: list[Paper] = []
        for entry in root.findall("atom:entry", self.atom_ns):
            title = _node_text(entry, "atom:title", self.atom_ns)
            abstract = _node_text(entry, "atom:summary", self.atom_ns)
            published_at = _node_text(entry, "atom:published", self.atom_ns)[:10] or None
            source_url = _node_text(entry, "atom:id", self.atom_ns)
            arxiv_id = _extract_arxiv_id(source_url)
            authors = [
                normalize_text(_node_text(author, "atom:name", self.atom_ns))
                for author in entry.findall("atom:author", self.atom_ns)
            ]
            doi = _node_text(entry, "arxiv:doi", self.atom_ns) or None
            categories = [
                category.attrib.get("term", "")
                for category in entry.findall("atom:category", self.atom_ns)
                if category.attrib.get("term")
            ]
            pdf_url = None
            for link in entry.findall("atom:link", self.atom_ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href")
                    break
            papers.append(
                Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    published_at=published_at,
                    venue="arXiv",
                    source="arxiv",
                    source_url=source_url,
                    pdf_url=pdf_url,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    fields_of_study=categories,
                )
            )
        return papers


class SemanticScholarClient:
    api_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_request_at = 0.0

    def search(self, query: str, *, limit: int = 20, from_year: int | None = None) -> list[Paper]:
        fields = ",".join(
            [
                "title",
                "authors",
                "abstract",
                "year",
                "publicationDate",
                "venue",
                "url",
                "externalIds",
                "citationCount",
                "referenceCount",
                "fieldsOfStudy",
            ]
        )
        url = with_query(
            self.api_url,
            {
                "query": query,
                "limit": min(limit, 100),
                "fields": fields,
                "year": f"{from_year}-" if from_year else None,
            },
        )
        headers = {}
        api_key = self._api_key()
        if api_key:
            headers["x-api-key"] = api_key
        payload = self._get_json_with_retry(url, headers=headers, authenticated=bool(api_key))
        papers: list[Paper] = []
        for item in payload.get("data", []):
            external_ids = item.get("externalIds") or {}
            authors = [author.get("name", "") for author in item.get("authors", [])]
            doi = external_ids.get("DOI")
            arxiv_id = external_ids.get("ArXiv")
            source_url = item.get("url")
            papers.append(
                Paper(
                    title=item.get("title") or "",
                    authors=authors,
                    abstract=item.get("abstract"),
                    year=item.get("year"),
                    published_at=item.get("publicationDate"),
                    venue=item.get("venue") or "Semantic Scholar",
                    source="semantic_scholar",
                    source_url=source_url,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    citation_count=item.get("citationCount"),
                    reference_count=item.get("referenceCount"),
                    fields_of_study=item.get("fieldsOfStudy") or [],
                )
            )
        return papers

    def lookup_by_title(self, title: str, *, year: int | None = None) -> Paper | None:
        clean_title = normalize_text(title)
        if not clean_title:
            return None
        from_year = max(year - 2, 1900) if year else None
        candidates = self.search(clean_title, limit=5, from_year=from_year)
        for candidate in candidates:
            if _titles_match(clean_title, candidate.title):
                return candidate
        return None

    def _api_key(self) -> str:
        local_key = getattr(local_config, "SEMANTIC_SCHOLAR_API_KEY", "") if local_config else ""
        return os.getenv("SEMANTIC_SCHOLAR_API_KEY") or local_key

    def _get_json_with_retry(self, url: str, *, headers: dict[str, str], authenticated: bool) -> dict:
        last_error: HttpError | None = None
        for attempt in range(3):
            self._wait_for_rate_slot(authenticated=authenticated)
            try:
                return get_json(url, headers=headers, timeout=SEARCH_HTTP_TIMEOUT, retries=0)
            except HttpError as exc:
                last_error = exc
                if "HTTP 429" not in str(exc) or attempt == 2:
                    raise
                time.sleep(3.0 * (attempt + 1))
        raise last_error or HttpError(f"Semantic Scholar request failed: {url}")

    def _wait_for_rate_slot(self, *, authenticated: bool) -> None:
        min_interval = 0.25 if authenticated else 1.25
        with self._lock:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_request_at = time.monotonic()


class OpenAlexClient:
    api_url = "https://api.openalex.org/works"

    def search(
        self,
        query: str,
        *,
        limit: int = 30,
        from_year: int | None = None,
        to_year: int | None = None,
    ) -> list[Paper]:
        filters = []
        if from_year:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year:
            filters.append(f"to_publication_date:{to_year}-12-31")
        params = {
            "search": query,
            "per-page": min(max(limit, 10), 200),
            "filter": ",".join(filters) if filters else None,
            "sort": "publication_date:desc",
        }
        api_key = self._api_key()
        if api_key:
            params["api_key"] = api_key
        mailto = self._mailto()
        if mailto:
            params["mailto"] = mailto
        payload = get_json(
            with_query(self.api_url, params),
            timeout=SEARCH_HTTP_TIMEOUT,
            retries=0,
        )
        papers: list[Paper] = []
        for item in payload.get("results", []):
            paper = self._paper_from_work(item)
            if paper and _year_in_range(paper.year, from_year, to_year):
                papers.append(paper)
        return papers

    def _paper_from_work(self, item: dict) -> Paper | None:
        title = normalize_text(item.get("title") or item.get("display_name") or "")
        if not title:
            return None
        ids = item.get("ids") or {}
        doi = normalize_text(item.get("doi") or ids.get("doi") or "") or None
        if doi and doi.startswith("https://doi.org/"):
            doi = doi.removeprefix("https://doi.org/")
        authors = [
            normalize_text((authorship.get("author") or {}).get("display_name") or "")
            for authorship in item.get("authorships", [])
            if isinstance(authorship, dict)
        ]
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}
        host_venue = item.get("host_venue") or {}
        venue = normalize_text(source.get("display_name") or host_venue.get("display_name") or "OpenAlex")
        source_url = normalize_text(primary_location.get("landing_page_url") or ids.get("openalex") or "") or None
        open_access = item.get("open_access") or {}
        pdf_url = normalize_text(open_access.get("oa_url") or "") or None
        fields = [
            normalize_text(concept.get("display_name") or "")
            for concept in item.get("concepts", [])
            if isinstance(concept, dict)
        ]
        fields.extend(
            normalize_text(topic.get("display_name") or "")
            for topic in item.get("topics", [])
            if isinstance(topic, dict)
        )
        return Paper(
            title=title,
            authors=authors,
            abstract=_openalex_abstract(item.get("abstract_inverted_index")),
            year=item.get("publication_year"),
            published_at=item.get("publication_date"),
            venue=venue,
            source="openalex",
            source_url=source_url,
            pdf_url=pdf_url,
            doi=doi,
            citation_count=item.get("cited_by_count"),
            reference_count=item.get("referenced_works_count"),
            fields_of_study=[field for field in fields if field],
        )

    def _mailto(self) -> str:
        local_mailto = getattr(local_config, "OPENALEX_MAILTO", "") if local_config else ""
        return os.getenv("OPENALEX_MAILTO") or local_mailto

    def _api_key(self) -> str:
        local_api_key = getattr(local_config, "OPENALEX_API_KEY", "") if local_config else ""
        return os.getenv("OPENALEX_API_KEY") or local_api_key


class DblpClient:
    api_url = "https://dblp.org/search/publ/api"

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        from_year: int | None = None,
        to_year: int | None = None,
        target_venues: list[str] | None = None,
    ) -> list[Paper]:
        # A venue suffix is useful for a small explicit target such as ICLR, but
        # DBLP does not consistently index the same venue spelling in every hit.
        # Always retain the plain topic query and let VenuePolicy perform the
        # definitive filter after retrieval. Rank filters may expand to many
        # venues, so querying every name here would be slow and brittle.
        queries = [query]
        if target_venues and len(target_venues) <= 3:
            for venue in target_venues:
                variants = _dblp_venue_query_variants(venue)
                if variants:
                    venue_query = f"{query} {variants[0]}"
                    if venue_query not in queries:
                        queries.append(venue_query)
        papers: list[Paper] = []
        per_query_limit = max(limit, 20)
        for search_query in queries:
            url = with_query(
                self.api_url,
                {
                    "q": search_query,
                    "format": "json",
                    "h": min(per_query_limit, 1000),
                },
            )
            payload = self._get_json_with_retry(url)
            hits = payload.get("result", {}).get("hits", {}).get("hit", [])
            if isinstance(hits, dict):
                hits = [hits]
            for hit in hits:
                info = hit.get("info") or {}
                paper = self._paper_from_info(info)
                if not paper:
                    continue
                if not _year_in_range(paper.year, from_year, to_year):
                    continue
                papers.append(paper)
        return dedupe_papers(papers)

    def _get_json_with_retry(self, url: str) -> dict:
        last_error: HttpError | None = None
        for attempt in range(3):
            try:
                return get_json(url, timeout=SEARCH_HTTP_TIMEOUT, retries=0)
            except HttpError as exc:
                last_error = exc
                if "HTTP 503" not in str(exc) or attempt == 2:
                    raise
                time.sleep(1.0 + attempt)
        raise last_error or HttpError(f"DBLP request failed: {url}")

    def _paper_from_info(self, info: dict) -> Paper | None:
        title = _strip_dblp_markup(info.get("title") or "")
        if not title:
            return None
        year = _safe_year(info.get("year"))
        authors_node = info.get("authors") if isinstance(info.get("authors"), dict) else {}
        authors = _parse_dblp_authors(authors_node.get("author"))
        venue = normalize_text(_first_text(info.get("venue") or info.get("journal") or info.get("booktitle")) or "DBLP")
        doi = normalize_text(_first_text(info.get("doi"))) or None
        source_url = normalize_text(_first_text(info.get("ee") or info.get("url"))) or None
        dblp_key = normalize_text(_first_text(info.get("key"))) or None
        if not source_url and dblp_key:
            source_url = f"https://dblp.org/rec/{dblp_key}"
        return Paper(
            title=title,
            authors=authors,
            abstract=None,
            year=year,
            venue=venue,
            source="dblp",
            source_url=source_url,
            doi=doi,
            fields_of_study=["cs"],
        )


class GoogleScholarClient:
    api_url = "https://serpapi.com/search.json"

    @property
    def available(self) -> bool:
        local_serpapi_key = getattr(local_config, "SERPAPI_API_KEY", "") if local_config else ""
        return bool(os.getenv("SERPAPI_API_KEY") or local_serpapi_key)

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        from_year: int | None = None,
        to_year: int | None = None,
        target_venues: list[str] | None = None,
    ) -> list[Paper]:
        local_serpapi_key = getattr(local_config, "SERPAPI_API_KEY", "") if local_config else ""
        api_key = os.getenv("SERPAPI_API_KEY") or local_serpapi_key
        if not api_key:
            return []
        url = with_query(
            self.api_url,
            {
                "engine": "google_scholar",
                "q": query,
                "num": min(limit, 20),
                "as_ylo": from_year,
                "as_yhi": to_year,
                "api_key": api_key,
            },
        )
        payload = get_json(url, timeout=SEARCH_HTTP_TIMEOUT, retries=0)
        papers: list[Paper] = []
        for item in payload.get("organic_results", []):
            title = normalize_text(item.get("title") or "")
            if not title:
                continue
            publication_info = item.get("publication_info") or {}
            summary = normalize_text(publication_info.get("summary") or "")
            year = _safe_year(summary) or _safe_year(item.get("snippet"))
            authors = _parse_scholar_authors(summary)
            venue = _infer_scholar_venue(summary, target_venues)
            papers.append(
                Paper(
                    title=title,
                    authors=authors,
                    abstract=normalize_text(item.get("snippet")) or None,
                    year=year,
                    venue=venue,
                    source="google_scholar",
                    source_url=normalize_text(item.get("link")) or None,
                    pdf_url=_scholar_resource_pdf_url(item),
                    citation_count=_safe_int_from_text(
                        ((item.get("inline_links") or {}).get("cited_by") or {}).get("total")
                    ),
                    fields_of_study=["cs"],
                )
            )
        return papers

    def lookup_by_title(self, title: str, *, year: int | None = None) -> Paper | None:
        """Find a title-matched Scholar result and retain only its explicit PDF resource."""
        clean_title = normalize_text(title)
        if not clean_title or not self.available:
            return None
        candidates = self.search(
            f'"{clean_title}"',
            limit=6,
            from_year=max(year - 2, 1900) if year else None,
            to_year=year + 2 if year else None,
        )
        for candidate in candidates:
            if candidate.pdf_url and _titles_match(clean_title, candidate.title):
                return candidate
        return None


class PaperSearchService:
    def __init__(self) -> None:
        self.arxiv = ArxivClient()
        self.semantic_scholar = SemanticScholarClient()
        self.openalex = OpenAlexClient()
        self.dblp = DblpClient()
        self.google_scholar = GoogleScholarClient()

    def search(
        self,
        query: str,
        *,
        limit: int = 30,
        sources: list[str] | None = None,
        recent_years: int | None = 3,
        from_year: int | None = None,
        to_year: int | None = None,
        target_venues: list[str] | None = None,
    ) -> SearchResult:
        selected_sources = sources or ["openalex", "arxiv", "semantic_scholar", "dblp", "google_scholar"]
        per_source_limit = max(limit, 10)
        papers: list[Paper] = []
        errors: list[str] = []
        source_status: dict[str, dict[str, object]] = {}
        current_year = datetime.now(timezone.utc).year
        inferred_from_year = current_year - recent_years + 1 if recent_years else None
        from_year = from_year or inferred_from_year
        to_year = to_year or current_year

        def collect(source: str, fetch) -> None:
            try:
                found = fetch()
                papers.extend(found)
                source_status[source] = {"status": "ok", "count": len(found), "error": ""}
            except Exception as exc:
                error = str(exc)
                errors.append(f"{source}: {error}")
                source_status[source] = {"status": "error", "count": 0, "error": error[:500]}

        if "arxiv" in selected_sources:
            collect("arxiv", lambda: self.arxiv.search(query, limit=per_source_limit))

        if "semantic_scholar" in selected_sources:
            collect(
                "semantic_scholar",
                lambda: self.semantic_scholar.search(query, limit=per_source_limit, from_year=from_year),
            )

        if "openalex" in selected_sources:
            collect(
                "openalex",
                lambda: self.openalex.search(query, limit=per_source_limit, from_year=from_year, to_year=to_year),
            )

        if "dblp" in selected_sources:
            collect(
                "dblp",
                lambda: self.dblp.search(
                    query,
                    limit=per_source_limit,
                    from_year=from_year,
                    to_year=to_year,
                    target_venues=target_venues,
                ),
            )

        if "google_scholar" in selected_sources:
            if not self.google_scholar.available:
                source_status["google_scholar"] = {
                    "status": "unavailable",
                    "count": 0,
                    "error": "未配置 SERPAPI_API_KEY，Google Scholar 查询未执行。",
                }
            else:
                collect(
                    "google_scholar",
                    lambda: self.google_scholar.search(
                        query,
                        limit=per_source_limit,
                        from_year=from_year,
                        to_year=to_year,
                        target_venues=target_venues,
                    ),
                )

        deduped = [
            paper
            for paper in dedupe_papers(papers)
            if _year_in_range(paper.year, from_year, to_year)
        ]
        if not deduped and errors:
            raise RuntimeError("Paper search failed: " + " | ".join(errors))
        return SearchResult(
            query=query,
            papers=deduped,
            sources=selected_sources,
            source_status=source_status,
        )

    def enrich_missing_metadata(self, papers: list[Paper], *, max_items: int = 10) -> dict[str, object]:
        """Best-effort abstract/identifier enrichment for DBLP-style metadata-only hits."""
        stats: dict[str, object] = {"attempted": 0, "enriched": 0, "failures": []}
        for paper in papers:
            if stats["attempted"] >= max_items:
                break
            if paper.abstract:
                continue
            stats["attempted"] = int(stats["attempted"]) + 1
            try:
                enriched = self.semantic_scholar.lookup_by_title(paper.title, year=paper.year)
            except Exception as exc:
                error = str(exc)[:300]
                failures = stats["failures"]
                if isinstance(failures, list) and error not in failures:
                    failures.append(error)
                if "429" in error or "Too Many Requests" in error:
                    break
                continue
            if not enriched:
                continue
            had_abstract = bool(paper.abstract)
            merge_papers(paper, enriched)
            if paper.abstract and not had_abstract:
                stats["enriched"] = int(stats["enriched"]) + 1
        return stats


def dedupe_papers(papers: list[Paper]) -> list[Paper]:
    by_key: dict[str, Paper] = {}
    for paper in papers:
        key = _paper_key(paper)
        existing = by_key.get(key)
        if not existing:
            by_key[key] = paper
            continue
        by_key[key] = merge_papers(existing, paper)
    return list(by_key.values())


def merge_papers(left: Paper, right: Paper) -> Paper:
    # Third-party aggregators often truncate authors and venue strings with an
    # ellipsis. Prefer the richer record so degraded metadata does not replace
    # a usable citation when the same work is later seen from another source.
    if _metadata_text_score(right.title) > _metadata_text_score(left.title):
        left.title = right.title
    if _authors_score(right.authors) > _authors_score(left.authors):
        left.authors = right.authors
    if right.abstract and (not left.abstract or len(right.abstract) > len(left.abstract)):
        left.abstract = right.abstract
    if _metadata_text_score(right.venue) > _metadata_text_score(left.venue):
        left.venue = right.venue
    left.source_url = left.source_url or right.source_url
    left.pdf_url = left.pdf_url or right.pdf_url
    left.doi = left.doi or right.doi
    left.arxiv_id = left.arxiv_id or right.arxiv_id
    left.citation_count = max(left.citation_count or 0, right.citation_count or 0) or None
    left.reference_count = left.reference_count or right.reference_count
    left.fields_of_study = sorted(set(left.fields_of_study + right.fields_of_study))
    left.abstract_status = "available" if left.abstract else (left.abstract_status or right.abstract_status or "none")
    left.local_pdf_path = left.local_pdf_path or right.local_pdf_path
    left.local_text_path = left.local_text_path or right.local_text_path
    left.fulltext_status = left.fulltext_status if left.fulltext_status != "none" else (right.fulltext_status or "none")
    left.fulltext_error = left.fulltext_error or right.fulltext_error
    left.fulltext_sha256 = left.fulltext_sha256 or right.fulltext_sha256
    left.fulltext_downloaded_at = left.fulltext_downloaded_at or right.fulltext_downloaded_at
    left.reading_status = _merge_reading_status(left.reading_status, right.reading_status)
    left.importance = _merge_importance(left.importance, right.importance)
    left.user_tags = sorted(set(left.user_tags + right.user_tags))
    left.excluded = left.excluded or right.excluded
    left.exclusion_reason = left.exclusion_reason or right.exclusion_reason
    left.user_notes = left.user_notes or right.user_notes
    left.used_in_sections = sorted(set(left.used_in_sections + right.used_in_sections))
    left.relevance_score = max(left.relevance_score or 0, right.relevance_score or 0) or None
    left.added_at = min(left.added_at, right.added_at)
    left.updated_at = max(left.updated_at or left.added_at, right.updated_at or right.added_at)
    left.is_verified = bool(left.doi or left.arxiv_id or left.source_url)
    left.source = _merge_sources(left.source, right.source)
    return left


def _metadata_text_score(value: str | None) -> float:
    text = normalize_text(value)
    if not text:
        return 0.0
    penalty = 10_000.0 if "…" in text or "..." in text else 0.0
    return len(text) - penalty


def _authors_score(authors: list[str]) -> float:
    if not authors:
        return 0.0
    text = " ".join(authors)
    penalty = 10_000.0 if "…" in text or "..." in text else 0.0
    return len(authors) * 100 + len(text) - penalty


def _merge_sources(*values: str | None) -> str:
    """Keep provenance compact when the same paper is found repeatedly."""
    sources: list[str] = []
    for value in values:
        for source in str(value or "").split(","):
            source = source.strip()
            if source and source not in sources:
                sources.append(source)
    return ",".join(sources) or "unknown"


def _merge_reading_status(left: str, right: str) -> str:
    order = {"unread": 0, "to_read": 1, "reading": 2, "read": 3}
    return left if order.get(left, 0) >= order.get(right, 0) else right


def _merge_importance(left: str, right: str) -> str:
    order = {"low": 0, "normal": 1, "high": 2, "core": 3}
    return left if order.get(left, 1) >= order.get(right, 1) else right


def _paper_key(paper: Paper) -> str:
    if paper.doi:
        return f"doi:{paper.doi.lower()}"
    if paper.arxiv_id:
        return f"arxiv:{paper.arxiv_id.lower()}"
    return "title:" + re.sub(r"[^a-z0-9]+", "", paper.title.lower())


def _titles_match(left: str, right: str) -> bool:
    left_key = re.sub(r"[^a-z0-9]+", "", left.lower())
    right_key = re.sub(r"[^a-z0-9]+", "", right.lower())
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    shorter, longer = sorted([left_key, right_key], key=len)
    return len(shorter) >= 24 and shorter in longer


def _openalex_abstract(value) -> str | None:
    if not isinstance(value, dict) or not value:
        return None
    positioned: list[tuple[int, str]] = []
    for token, positions in value.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            try:
                positioned.append((int(position), str(token)))
            except (TypeError, ValueError):
                continue
    if not positioned:
        return None
    return normalize_text(" ".join(token for _, token in sorted(positioned)))


def _node_text(node: ET.Element, path: str, ns: dict[str, str]) -> str:
    found = node.find(path, ns)
    return normalize_text(found.text if found is not None else "")


def _extract_arxiv_id(url: str | None) -> str | None:
    if not url:
        return None
    value = url.rstrip("/").split("/")[-1]
    return value if value else None


def _parse_dblp_authors(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [normalize_text(value)]
    if isinstance(value, dict):
        return [normalize_text(value.get("text") or value.get("#text") or "")]
    authors = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                authors.append(normalize_text(item))
            elif isinstance(item, dict):
                authors.append(normalize_text(item.get("text") or item.get("#text") or ""))
    return [author for author in authors if author]


def _first_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            text = _first_text(item)
            if text:
                return text
        return None
    if isinstance(value, dict):
        return value.get("text") or value.get("#text") or value.get("href")
    return str(value)


def _strip_dblp_markup(value: str) -> str:
    return normalize_text(html.unescape(re.sub(r"<[^>]+>", "", value)))


def _safe_year(value) -> int | None:
    try:
        if value is None:
            return None
        text = str(value)
        match = re.search(r"\b(19|20)\d{2}\b", text)
        if match:
            return int(match.group(0))
        return int(text[:4])
    except ValueError:
        return None


def _safe_int_from_text(value) -> int | None:
    if isinstance(value, int):
        return value
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def _parse_scholar_authors(summary: str) -> list[str]:
    if not summary:
        return []
    first_part = summary.split(" - ")[0]
    authors = [normalize_text(author) for author in re.split(r",| and ", first_part)]
    return [author for author in authors if author and not re.search(r"\d{4}", author)][:8]


def _scholar_resource_pdf_url(item: dict) -> str | None:
    """Return only Scholar's explicit public PDF resource link, when present."""
    resources = item.get("resources") or []
    if isinstance(resources, dict):
        resources = [resources]
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        file_format = normalize_text(resource.get("file_format") or "").lower()
        link = normalize_text(resource.get("link") or "")
        if file_format == "pdf" and link.startswith(("https://", "http://")):
            return link

    primary_link = normalize_text(item.get("link") or "")
    if primary_link.lower().split("?", 1)[0].endswith(".pdf") and primary_link.startswith(("https://", "http://")):
        return primary_link
    return None


def _infer_scholar_venue(summary: str, target_venues: list[str] | None) -> str:
    for venue in target_venues or []:
        if re.search(r"(?<![a-z0-9])" + re.escape(venue) + r"(?![a-z0-9])", summary, flags=re.I):
            return venue
    if " - " in summary:
        parts = [normalize_text(part) for part in summary.split(" - ") if normalize_text(part)]
        if len(parts) >= 2:
            venue_part = re.sub(r",?\s*(19|20)\d{2}.*$", "", parts[1]).strip()
            if venue_part:
                return venue_part[:120]
    return "Google Scholar"


def _year_in_range(year: int | None, from_year: int | None, to_year: int | None) -> bool:
    if year is None:
        return True
    if from_year is not None and year < from_year:
        return False
    if to_year is not None and year > to_year:
        return False
    return True


def _build_arxiv_query(query: str) -> str:
    terms = _important_terms(query)
    if not terms:
        return f"all:{query}"
    if len(terms) <= 3:
        return " AND ".join(f"all:{term}" for term in terms)
    core_terms = terms[:2]
    optional_terms = terms[2:6]
    return "(" + " AND ".join(f"all:{term}" for term in core_terms) + ") AND (" + " OR ".join(f"all:{term}" for term in optional_terms) + ")"


def _important_terms(query: str) -> list[str]:
    stopwords = {
        "and",
        "for",
        "the",
        "with",
        "from",
        "using",
        "based",
        "study",
        "paper",
        "related",
        "work",
    }
    terms = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", query.lower()):
        if len(token) <= 2 or token in stopwords:
            continue
        terms.append(token)
    return terms


def _dblp_venue_query_variants(venue: str) -> list[str]:
    key = re.sub(r"[^a-z0-9]+", "", venue.lower())
    variants = DBLP_VENUE_QUERY_ALIASES.get(key, [venue])
    return list(dict.fromkeys(variants))


DBLP_VENUE_QUERY_ALIASES = {
    "tpami": [
        "T-PAMI",
        "TPAMI",
        "IEEE Trans. Pattern Anal. Mach. Intell.",
        "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    ],
    "tkde": [
        "TKDE",
        "IEEE Trans. Knowl. Data Eng.",
        "IEEE Transactions on Knowledge and Data Engineering",
    ],
    "tifs": [
        "TIFS",
        "IEEE Trans. Inf. Forensics Secur.",
        "IEEE Transactions on Information Forensics and Security",
    ],
    "tnnls": [
        "TNNLS",
        "IEEE Trans. Neural Networks Learn. Syst.",
        "IEEE Transactions on Neural Networks and Learning Systems",
    ],
    "jmlr": ["JMLR", "Journal of Machine Learning Research"],
    "cacm": ["Commun. ACM", "Communications of the ACM"],
}
