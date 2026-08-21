from __future__ import annotations

from datetime import datetime, timezone
import math
import re

from .models import Paper


def rank_papers(papers: list[Paper], query: str, *, limit: int = 20) -> list[Paper]:
    scored = [(paper, _score(paper, query)) for paper in papers]
    relevant = [(paper, score) for paper, score in scored if score > 0]
    pool = relevant if relevant else scored
    return [paper for paper, _ in sorted(pool, key=lambda item: item[1], reverse=True)[:limit]]


def split_classic_and_recent(papers: list[Paper], *, recent_years: int = 3) -> tuple[list[Paper], list[Paper]]:
    current_year = datetime.now(timezone.utc).year
    cutoff = current_year - recent_years + 1
    recent = [paper for paper in papers if paper.year and paper.year >= cutoff]
    classic = [paper for paper in papers if paper not in recent]
    classic = sorted(classic, key=lambda paper: paper.citation_count or 0, reverse=True)
    recent = sorted(recent, key=lambda paper: (paper.year or 0, paper.citation_count or 0), reverse=True)
    return classic, recent


def _score(paper: Paper, query: str) -> float:
    query_tokens = set(_tokens(query))
    title_tokens = set(_tokens(paper.title))
    abstract_tokens = set(_tokens(paper.abstract or ""))
    if not query_tokens:
        return 0.0

    overlap_count = len(query_tokens & (title_tokens | abstract_tokens))
    if overlap_count == 0:
        return -1.0
    if len(query_tokens) >= 4 and overlap_count < 2:
        return -1.0

    title_overlap = len(query_tokens & title_tokens) / len(query_tokens)
    abstract_overlap = len(query_tokens & abstract_tokens) / len(query_tokens)
    citation_score = math.log1p(paper.citation_count or 0) / 10
    recency_score = _recency_score(paper.year) * 0.35
    verified_bonus = 0.2 if paper.is_verified else 0
    return 4.0 * title_overlap + 2.0 * abstract_overlap + citation_score + recency_score + verified_bonus


def _tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", value.lower()) if len(token) > 2]


def _recency_score(year: int | None) -> float:
    if not year:
        return 0.0
    current_year = datetime.now(timezone.utc).year
    age = max(current_year - year, 0)
    return max(0.0, 1.0 - age / 10)
