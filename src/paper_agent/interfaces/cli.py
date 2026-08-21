from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..core.intent import IntentAnalyzer
from ..core.rank import rank_papers
from ..core.related_work import RelatedWorkGenerator
from ..tools.search import PaperSearchService, dedupe_papers
from ..storage.store import JsonPaperStore
from ..tools.venues import VenuePolicy


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "search":
            return run_search(args)
        if args.command == "related-work":
            return run_related_work(args)
        if args.command == "investigate":
            return run_investigate(args)
        if args.command == "web":
            return run_web(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    parser.print_help()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search papers and draft a citation-safe Related Work section.")
    subparsers = parser.add_subparsers(dest="command")

    search = subparsers.add_parser("search", help="Search papers and save metadata locally.")
    _add_search_args(search)
    search.add_argument("--json", action="store_true", help="Print raw JSON results.")

    related = subparsers.add_parser("related-work", help="Search papers and generate a Related Work draft.")
    _add_search_args(related)
    related.add_argument("--language", choices=["en", "zh"], default="en")
    related.add_argument("--output", default="outputs/related_work.md")
    related.add_argument("--bib", default="outputs/references.bib")
    related.add_argument("--claim-map", default="outputs/claim_map.json")
    related.add_argument("--use-kimi", action="store_true", help="Use Kimi to draft when KIMI_API_KEY is configured.")

    investigate = subparsers.add_parser(
        "investigate",
        help="Infer a CS research direction, search target venues, and draft Related Work.",
    )
    investigate.add_argument("--request", required=True, help="Natural-language research request from the user.")
    investigate.add_argument("--limit", type=int, default=24, help="Number of papers to keep after ranking.")
    investigate.add_argument("--recent-years", type=int, default=None)
    investigate.add_argument("--from-year", type=int, default=None)
    investigate.add_argument("--to-year", type=int, default=None)
    investigate.add_argument("--venue", action="append", default=[], help="Target conference or journal. Can be repeated.")
    investigate.add_argument(
        "--sources",
        nargs="+",
        default=["arxiv", "semantic_scholar", "dblp", "google_scholar"],
        choices=["arxiv", "semantic_scholar", "dblp", "google_scholar"],
    )
    investigate.add_argument("--language", choices=["en", "zh"], default="zh")
    investigate.add_argument("--store", default="data/papers.json")
    investigate.add_argument("--venue-config", default="config/venues.json")
    investigate.add_argument("--output", default="outputs/investigation_related_work.md")
    investigate.add_argument("--bib", default="outputs/investigation_references.bib")
    investigate.add_argument("--intent", default="outputs/intent.json")
    investigate.add_argument("--claim-map", default="outputs/investigation_claim_map.json")
    investigate.add_argument("--use-kimi", action="store_true", default=True)
    investigate.add_argument("--no-kimi", action="store_false", dest="use_kimi")

    web = subparsers.add_parser("web", help="Run the small web UI.")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    return parser


def _add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query", required=True, help="Research topic or paper search query.")
    parser.add_argument("--limit", type=int, default=20, help="Number of papers to keep after ranking.")
    parser.add_argument("--recent-years", type=int, default=3, help="Recent-paper window used by sources that support it.")
    parser.add_argument("--from-year", type=int, default=None)
    parser.add_argument("--to-year", type=int, default=None)
    parser.add_argument("--venue", action="append", default=[], help="Target conference or journal. Can be repeated.")
    parser.add_argument("--venue-config", default="config/venues.json")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["arxiv", "semantic_scholar", "dblp", "google_scholar"],
        choices=["arxiv", "semantic_scholar", "dblp", "google_scholar"],
    )
    parser.add_argument("--store", default="data/papers.json", help="Local JSON paper index path.")


def run_search(args: argparse.Namespace) -> int:
    papers = _search_and_rank(args)
    store = JsonPaperStore(args.store)
    store.save_papers(papers)
    if args.json:
        print(json.dumps([paper.to_dict() for paper in papers], ensure_ascii=False, indent=2))
    else:
        _print_papers(papers)
        print(f"\nSaved {len(papers)} papers to {args.store}")
    return 0


def run_related_work(args: argparse.Namespace) -> int:
    papers = _search_and_rank(args)
    store = JsonPaperStore(args.store)
    store.save_papers(papers)

    draft = RelatedWorkGenerator().generate(
        query=args.query,
        papers=papers,
        language=args.language,
        use_llm=args.use_kimi,
    )
    _write_text(args.output, draft.content_markdown)
    _write_text(args.bib, draft.bibtex)
    _write_text(
        args.claim_map,
        json.dumps(
            {
                "query": draft.query,
                "generated_at": draft.generated_at,
                "paper_ids": draft.paper_ids,
                "claim_map": draft.claim_map,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )

    print(f"Generated Related Work draft: {args.output}")
    print(f"Generated BibTeX references: {args.bib}")
    print(f"Generated claim map: {args.claim_map}")
    print(f"Saved {len(papers)} papers to {args.store}")
    return 0


def run_investigate(args: argparse.Namespace) -> int:
    intent = IntentAnalyzer().analyze(args.request)
    service = PaperSearchService()
    all_papers = []
    target_venues = args.venue or intent.target_venues
    target_venue_ranks = intent.target_venue_ranks
    recent_years = args.recent_years if args.recent_years is not None else intent.recent_years
    from_year = args.from_year if args.from_year is not None else intent.from_year
    to_year = args.to_year if args.to_year is not None else intent.to_year
    if recent_years is None and from_year is None:
        recent_years = 3

    source_plan = _source_query_plan(intent)
    candidate_limit = max(args.limit, 200) if target_venue_ranks else max(args.limit, 30)
    for source in args.sources:
        for query in source_plan.get(source, intent.queries):
            try:
                result = service.search(
                    query,
                    limit=candidate_limit,
                    sources=[source],
                    recent_years=recent_years,
                    from_year=from_year,
                    to_year=to_year,
                    target_venues=target_venues,
                )
                all_papers.extend(result.papers)
            except RuntimeError:
                continue

    deduped = dedupe_papers(all_papers)
    venue_policy = VenuePolicy(args.venue_config)
    target_papers = venue_policy.filter(
        deduped,
        target_venues=target_venues,
        target_venue_ranks=target_venue_ranks,
    )
    ranked = rank_papers(target_papers, intent.normalized_topic, limit=args.limit)

    store = JsonPaperStore(args.store)
    store.save_papers(ranked)

    draft = RelatedWorkGenerator().generate(
        query=intent.normalized_topic,
        papers=ranked,
        language=args.language,
        use_llm=args.use_kimi,
    )
    _write_text(args.output, draft.content_markdown)
    _write_text(args.bib, draft.bibtex)
    _write_text(
        args.intent,
        json.dumps(
            {
                **intent.to_dict(),
                "effective_target_venues": target_venues,
                "effective_target_venue_ranks": target_venue_ranks,
                "effective_recent_years": recent_years,
                "effective_from_year": from_year,
                "effective_to_year": to_year,
                "total_retrieved": len(deduped),
                "target_venue_papers": len(target_papers),
                "selected_papers": len(ranked),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    _write_text(
        args.claim_map,
        json.dumps(
            {
                "query": draft.query,
                "generated_at": draft.generated_at,
                "paper_ids": draft.paper_ids,
                "claim_map": draft.claim_map,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )

    print(f"Intent source: {intent.source}")
    print(f"Detected topic: {intent.normalized_topic}")
    print(f"CS area: {intent.cs_area}")
    if target_venues:
        print(f"Target venues: {', '.join(target_venues)}")
    if target_venue_ranks:
        print(f"Target venue ranks: {', '.join(target_venue_ranks)}")
    if from_year or to_year:
        print(f"Year range: {from_year or '...'}-{to_year or '...'}")
    elif recent_years:
        print(f"Recent years: {recent_years}")
    print(f"Search queries: {', '.join(intent.queries)}")
    if intent.source_queries:
        print(f"DBLP queries: {', '.join(intent.source_queries.get('dblp', []))}")
        print(f"arXiv queries: {', '.join(intent.source_queries.get('arxiv', []))}")
        print(f"Google Scholar queries: {', '.join(intent.source_queries.get('google_scholar', []))}")
    print(f"Retrieved papers: {len(deduped)}")
    print(f"Target-venue papers: {len(target_papers)}")
    print(f"Selected papers: {len(ranked)}")
    print(f"Generated Related Work draft: {args.output}")
    print(f"Generated BibTeX references: {args.bib}")
    print(f"Generated intent report: {args.intent}")
    return 0


def run_web(args: argparse.Namespace) -> int:
    from .web_app import run_web as start_web

    start_web(args.host, args.port)
    return 0


def _search_and_rank(args: argparse.Namespace):
    service = PaperSearchService()
    result = service.search(
        args.query,
        limit=max(args.limit, 20),
        sources=args.sources,
        recent_years=args.recent_years,
        from_year=getattr(args, "from_year", None),
        to_year=getattr(args, "to_year", None),
        target_venues=getattr(args, "venue", None),
    )
    papers = result.papers
    target_venues = getattr(args, "venue", None)
    if target_venues:
        venue_policy = VenuePolicy(getattr(args, "venue_config", "config/venues.json"))
        papers = venue_policy.filter(papers, target_venues=target_venues)
    return rank_papers(papers, args.query, limit=args.limit)


def _write_text(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _print_papers(papers) -> None:
    for index, paper in enumerate(papers, start=1):
        authors = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors += ", et al."
        print(f"{index}. {paper.title}")
        print(f"   {authors} ({paper.year or 'n.d.'}) | {paper.venue or paper.source} | {paper.source}")
        print(f"   {paper.source_url or paper.pdf_url or 'no URL'}")


def _source_query_plan(intent) -> dict[str, list[str]]:
    fallback = intent.queries or [intent.normalized_topic]
    source_queries = intent.source_queries or {}
    plan = {
        "dblp": source_queries.get("dblp") or fallback,
        "arxiv": source_queries.get("arxiv") or fallback,
        "semantic_scholar": fallback,
        "google_scholar": source_queries.get("google_scholar") or fallback,
    }
    return {source: list(dict.fromkeys(query for query in queries if query))[:4] for source, queries in plan.items()}


if __name__ == "__main__":
    raise SystemExit(main())
