from paper_agent.core.models import Paper
from paper_agent.core.assistant_engine import ResearchAssistantEngine
from paper_agent.core.systematic_review import review_snapshot, write_review_export
from paper_agent.storage.store import SQLitePaperStore


def _paper(title: str, year: int) -> Paper:
    return Paper(
        title=title,
        authors=["Researcher"],
        abstract=f"Abstract for {title}.",
        year=year,
        venue="ICLR",
        source="openalex",
        source_url=f"https://example.org/{year}",
    )


def test_review_protocol_and_screening_are_project_scoped_and_persistent(tmp_path):
    store = SQLitePaperStore(tmp_path / "papers.sqlite")
    store.create_project("Review", project_id="review")
    included, excluded = _paper("Included Study", 2025), _paper("Excluded Study", 2024)
    store.save_papers([included, excluded])
    store.replace_project_papers("review", [included.id, excluded.id])
    protocol = store.upsert_review_protocol(
        "review",
        {
            "research_question": "What debiasing methods work?",
            "inclusion_criteria": ["Peer reviewed", "Recommender systems"],
            "exclusion_criteria": ["Not empirical"],
            "search_strategy": "OpenAlex, DBLP, 2024-2026",
        },
    )
    store.set_review_screening("review", included.id, stage="full_text", decision="include")
    store.set_review_screening("review", excluded.id, stage="title_abstract", decision="exclude", reason="Wrong domain")

    snapshot = review_snapshot(
        papers=[included, excluded], protocol=protocol, screenings=store.list_review_screenings("review")
    )

    assert snapshot["summary"] == {
        "identified": 2,
        "screened": 2,
        "pending": 0,
        "title_abstract_excluded": 1,
        "full_text_excluded": 0,
        "included": 1,
    }
    assert snapshot["rows"][1]["reason"] == "Wrong domain"


def test_systematic_review_exports_evidence_table_and_prisma_summary(tmp_path):
    snapshot = review_snapshot(
        papers=[_paper("A Study", 2026)],
        protocol={"research_question": "RQ", "inclusion_criteria": ["Relevant"], "exclusion_criteria": [], "search_strategy": "OpenAlex"},
        screenings=[],
    )
    csv_path = tmp_path / "evidence.csv"
    prisma_path = tmp_path / "prisma.md"

    write_review_export(snapshot=snapshot, format="evidence_csv", path=csv_path)
    write_review_export(snapshot=snapshot, format="prisma_markdown", path=prisma_path)

    assert "A Study" in csv_path.read_text(encoding="utf-8-sig")
    text = prisma_path.read_text(encoding="utf-8")
    assert "PRISMA Flow Summary" in text
    assert "Identified records: 1" in text


def test_included_screening_decisions_constrain_the_writing_evidence_pool(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite")
    included, excluded = _paper("Included Study", 2025), _paper("Excluded Study", 2024)
    engine.store.save_papers([included, excluded])
    engine.state.papers = [included, excluded]
    engine.state.evidence_paper_ids = [included.id, excluded.id]
    engine.store.set_review_screening("default", included.id, stage="full_text", decision="include")
    engine.store.set_review_screening("default", excluded.id, stage="full_text", decision="exclude", reason="Out of scope")

    assert [paper.title for paper in engine._evidence_papers()] == ["Included Study"]
