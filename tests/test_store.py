from paper_agent.core.models import Paper
from paper_agent.storage.store import SQLitePaperStore


def test_sqlite_store_caches_abstracts_for_similar_queries(tmp_path):
    store = SQLitePaperStore(tmp_path / "papers.sqlite")
    paper = Paper(
        title="Dynamic Recommendation with Temporal User Interests",
        authors=["Ada Lovelace"],
        abstract="This paper studies dynamic recommender systems with temporal user preference drift.",
        year=2025,
        venue="KDD",
        source="semantic_scholar",
        source_url="https://example.org/dynamic-rec",
    )

    store.save_papers([paper], query="dynamic recommender systems", topic="dynamic recommender systems")

    cached = store.search_cached("temporal dynamic recommendation", from_year=2024, to_year=2026)

    assert cached
    assert cached[0].abstract == paper.abstract
    assert store.stats()["abstract_count"] == 1


def test_sqlite_store_preserves_fulltext_cache_fields(tmp_path):
    store = SQLitePaperStore(tmp_path / "papers.sqlite")
    paper = Paper(
        title="Cached Full Text Paper",
        authors=["Ada Lovelace"],
        abstract="A cached paper.",
        year=2026,
        source="arxiv",
        source_url="https://example.org/cached",
        pdf_url="https://example.org/cached.pdf",
        local_pdf_path=str(tmp_path / "paper_files" / "cached.pdf"),
        local_text_path=str(tmp_path / "paper_texts" / "cached.json"),
        fulltext_status="extracted",
        fulltext_sha256="abc123",
        fulltext_downloaded_at="2026-07-24T00:00:00Z",
    )

    store.save_papers([paper])
    loaded = store.load_papers()[0]

    assert loaded.abstract_status == "available"
    assert loaded.local_pdf_path == paper.local_pdf_path
    assert loaded.local_text_path == paper.local_text_path
    assert loaded.fulltext_status == "extracted"
    assert loaded.fulltext_sha256 == "abc123"
    assert store.stats()["fulltext_count"] == 1


def test_sqlite_store_preserves_paper_asset_state_across_refreshes(tmp_path):
    store = SQLitePaperStore(tmp_path / "papers.sqlite")
    first = Paper(
        title="Evidence-Aware Dynamic Recommendation",
        authors=["Ada Lovelace"],
        abstract="A first abstract.",
        year=2026,
        source="openalex",
        source_url="https://example.org/openalex",
    )
    store.save_papers([first])
    saved = store.load_papers()[0]

    updated = store.update_paper_asset_state(
        saved.id,
        reading_status="read",
        importance="core",
        user_tags=["debiasing", "dynamic rec"],
        used_in_sections=["related_work", "method_survey"],
        relevance_score=0.91,
    )
    assert updated is not None

    refreshed = Paper(
        title="Evidence-Aware Dynamic Recommendation.",
        authors=["Ada Lovelace"],
        abstract="A richer refreshed abstract.",
        year=2026,
        source="dblp",
        source_url="https://dblp.org/rec/conf/example",
    )
    store.save_papers([refreshed])
    loaded = store.load_papers()[0]

    assert loaded.abstract == "A richer refreshed abstract."
    assert loaded.reading_status == "read"
    assert loaded.importance == "core"
    assert loaded.user_tags == ["debiasing", "dynamic rec"]
    assert loaded.used_in_sections == ["related_work", "method_survey"]
    assert loaded.relevance_score == 0.91
    assert store.stats()["core_count"] == 1
    assert store.stats()["read_count"] == 1


def test_sqlite_store_tracks_excluded_papers_as_assets(tmp_path):
    store = SQLitePaperStore(tmp_path / "papers.sqlite")
    paper = Paper(
        title="Off-topic Dynamic Recommendation",
        authors=["Ada Lovelace"],
        year=2025,
        source="google_scholar",
        source_url="https://example.org/off-topic",
    )
    store.save_papers([paper])
    saved = store.load_papers()[0]

    store.update_paper_asset_state(
        saved.id,
        excluded=True,
        exclusion_reason="只讨论普通推荐，不涉及动态去偏。",
        importance="low",
    )
    loaded = store.load_papers()[0]

    assert loaded.excluded is True
    assert loaded.exclusion_reason == "只讨论普通推荐，不涉及动态去偏。"
    assert loaded.importance == "low"
    assert store.stats()["excluded_count"] == 1


def test_sqlite_store_excludes_rejected_papers_from_cache_hits(tmp_path):
    store = SQLitePaperStore(tmp_path / "papers.sqlite")
    keep = Paper(
        title="Relevant Dynamic Recommendation",
        authors=["Ada Lovelace"],
        abstract="A dynamic recommendation paper.",
        year=2026,
        source="openalex",
        source_url="https://example.org/keep",
    )
    reject = Paper(
        title="Rejected Dynamic Recommendation",
        authors=["Alan Turing"],
        abstract="A dynamic recommendation paper that the user rejected.",
        year=2026,
        source="openalex",
        source_url="https://example.org/reject",
    )
    store.save_papers([keep, reject])
    saved = store.load_papers()
    rejected_id = next(paper.id for paper in saved if paper.title.startswith("Rejected"))
    store.update_paper_asset_state(rejected_id, excluded=True)

    cached = store.search_cached("dynamic recommendation", from_year=2025, to_year=2026)

    assert {paper.title for paper in cached} == {"Relevant Dynamic Recommendation"}


def test_sqlite_store_merges_same_title_across_sources(tmp_path):
    store = SQLitePaperStore(tmp_path / "papers.sqlite")
    dblp_paper = Paper(
        title="Enhancing New-item Fairness in Dynamic Recommender Systems.",
        authors=["A"],
        year=2025,
        venue="SIGIR",
        source="dblp",
        source_url="https://dblp.org/rec/conf/sigir/example",
    )
    scholar_paper = Paper(
        title="Enhancing New-item Fairness in Dynamic Recommender Systems",
        authors=["A", "B"],
        abstract="This work studies new-item fairness in dynamic recommender systems.",
        year=2025,
        source="google_scholar",
        source_url="https://example.org/scholar",
    )

    store.save_papers([dblp_paper])
    store.save_papers([scholar_paper])
    papers = store.load_papers()

    assert len(papers) == 1
    assert papers[0].abstract == scholar_paper.abstract
    assert set(papers[0].source.split(",")) == {"dblp", "google_scholar"}


def test_sqlite_store_merges_author_year_title_variants(tmp_path):
    store = SQLitePaperStore(tmp_path / "papers.sqlite")
    short_title = Paper(
        title="Temporal Debiasing for Dynamic Recommendation",
        authors=["Ada Lovelace", "Alan Turing"],
        year=2026,
        source="dblp",
        source_url="https://dblp.org/rec/conf/example",
    )
    long_title = Paper(
        title="Temporal Debiasing for Dynamic Recommendation: A Benchmark Study",
        authors=["Ada Lovelace", "Grace Hopper"],
        abstract="A benchmark study with a richer abstract.",
        year=2026,
        source="openalex",
        source_url="https://openalex.org/W123",
    )

    store.save_papers([short_title])
    store.save_papers([long_title])
    papers = store.load_papers()

    assert len(papers) == 1
    assert papers[0].abstract == "A benchmark study with a richer abstract."
    assert set(papers[0].source.split(",")) == {"dblp", "openalex"}


def test_sqlite_store_dedupes_existing_rows_and_repoints_queries(tmp_path):
    store = SQLitePaperStore(tmp_path / "papers.sqlite")
    first = Paper(
        title="Task-Aware Retrieval Augmentation for Dynamic Recommendation",
        authors=["A"],
        year=2026,
        source="semantic_scholar",
        source_url="https://example.org/semantic",
    )
    second = Paper(
        title="Task-Aware Retrieval Augmentation for Dynamic Recommendation.",
        authors=["A", "B"],
        abstract="A richer abstract.",
        year=2026,
        source="dblp",
        source_url="https://dblp.org/rec/conf/example",
    )
    second.id = "manual_duplicate_id"
    store.save_papers([first], query="dynamic recommendation", topic="rec")
    with store._connect() as conn:
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
            """,
            store._paper_values(second, "2026-07-24T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO paper_queries (paper_id, query, topic, source, searched_at) VALUES (?, ?, ?, ?, ?)",
            (second.id, "task aware recommendation", "rec", second.source, "2026-07-24T00:00:00Z"),
        )

    result = store.dedupe_existing()
    papers = store.load_papers()

    assert result["merged"] == 1
    assert len(papers) == 1
    assert papers[0].abstract == second.abstract
    with store._connect() as conn:
        query_paper_ids = {row[0] for row in conn.execute("SELECT paper_id FROM paper_queries").fetchall()}
    assert query_paper_ids == {papers[0].id}
