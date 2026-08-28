from pathlib import Path

from paper_agent.core import assistant_engine as assistant_engine_module
from paper_agent.core.assistant_engine import ResearchAssistantEngine, _search_answer_summary, _search_source_summary, _source_query_plan
from paper_agent.core.agent_runtime import IterativeAgentRuntime
from paper_agent.core.intent import ActionIntent, IntentAnalyzer, RequestAnalysis, ResearchIntent
from paper_agent.core.models import Paper, SearchResult


class FakeIntentAnalyzer:
    class LLM:
        available = False
        model = "fake"

    llm = LLM()

    def analyze_action(self, request, *, mode="auto", has_papers=False):
        return ActionIntent(request, "document", "test", 1.0, "fake")


def test_search_source_summary_explains_partial_failures():
    summary = _search_source_summary(
        {
            "dblp": {"queries": 3, "successful_queries": 0, "empty_queries": 1, "failures": ["HTTP 503 Service Unavailable"]},
            "arxiv": {"queries": 3, "successful_queries": 3, "empty_queries": 0, "failures": []},
            "semantic_scholar": {"queries": 3, "successful_queries": 0, "empty_queries": 0, "failures": ["HTTP 429 Too Many Requests"]},
            "google_scholar": {"queries": 3, "successful_queries": 0, "empty_queries": 3, "failures": ["未配置 SERPAPI_API_KEY，Google Scholar 查询未执行。"]},
        }
    )

    assert "DBLP 0/3 个查询有结果，1 个空结果，1 个失败：HTTP 503 Service Unavailable" in summary
    assert "arXiv 3/3 个查询有结果" in summary
    assert "Semantic Scholar 0/3 个查询有结果，0 个空结果，1 个失败：HTTP 429 Too Many Requests" in summary
    assert "Google Scholar 未配置" in summary


def test_empty_search_response_discloses_the_english_queries_that_were_attempted():
    intent = ResearchIntent(
        original_request="调研动态去偏推荐系统近三年进展",
        normalized_topic="debiasing dynamic recommender systems",
        cs_area="AI",
        queries=["debiasing temporal recommendation"],
        source_queries={"dblp": ["debiasing dynamic rec"]},
        recent_years=3,
    )

    answer = _search_answer_summary(intent, [], "")

    assert "英文检索式" in answer
    assert "debiasing dynamic rec" in answer


class FakeFollowUpIntentAnalyzer(FakeIntentAnalyzer):
    def analyze_action(self, request, *, mode="auto", has_papers=False):
        return ActionIntent(request, "answer", "follow-up", 1.0, "fake")


class FakePdfAnswerIntentAnalyzer(FakeIntentAnalyzer):
    def analyze_action(self, request, *, mode="auto", has_papers=False):
        return ActionIntent(request, "answer", "read uploaded PDF", 1.0, "fake")


def test_source_query_plan_keeps_more_dynamic_recommender_variants():
    intent = ResearchIntent(
        original_request="查近三年动态推荐系统论文",
        normalized_topic="dynamic recommender systems",
        cs_area="AI",
        queries=["dynamic recommender systems", "dynamic recommendation"],
        keywords=["dynamic", "recommender", "systems"],
        source_queries={
            "dblp": [
                "dynamic rec",
                "dynamic recommender",
                "dynamic recommender systems",
                "dynamic recommendation",
                "time-aware recommendation",
                "temporal recommendation",
            ]
        },
        recent_years=3,
    )

    plan = _source_query_plan(intent)

    assert len(plan["dblp"]) == 6
    assert "dynamic rec" in plan["dblp"]
    assert "temporal recommendation" in plan["dblp"] or "time-aware recommendation" in plan["dblp"]


def test_source_query_plan_adds_openalex_facet_boolean_query_for_compound_topics():
    intent = ResearchIntent(
        original_request="搜索近三年去偏动态推荐系统论文",
        normalized_topic="debiasing in dynamic recommender systems",
        cs_area="AI",
        queries=["debiasing dynamic recommender systems"],
        keywords=["debiasing", "dynamic", "recommender", "fairness"],
        recent_years=3,
    )

    plan = _source_query_plan(intent)

    assert "openalex" in plan
    combined = " ".join(plan["openalex"]).lower()
    assert "fairness" in combined or "debias" in combined
    assert "dynamic" in combined
    assert "recommender" in combined or "recommendation" in combined


def test_simple_greeting_uses_model_owned_free_chat_route(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")

    class ChatLlm:
        available = True
        model = "fake-kimi"

        def chat_json(self, **kwargs):
            return {
                "category": "chat",
                "subtask": "general_chat",
                "deliverable": "none",
                "evidence_scope": "none",
                "tool_plan": ["free_chat"],
                "needs_fresh_literature": False,
                "reason": "普通自由交流。",
                "confidence": 0.99,
                "normalized_topic": "",
                "cs_area": "Interdisciplinary CS",
                "keywords": [],
                "queries": [],
                "source_queries": {},
                "target_venues": [],
                "target_venue_ranks": [],
                "recent_years": None,
                "from_year": None,
                "to_year": None,
            }

        def chat_text(self, **kwargs):
            return "你好，我在。"

    llm = ChatLlm()
    engine.llm = llm
    engine.intent_analyzer = IntentAnalyzer(llm=llm)

    events = list(engine.handle_stream("hello"))

    assert next(event["content"] for event in events if event["type"] == "answer") == "你好，我在。"
    assert engine.state.last_tool_plan == ["free_chat"]
    action = next(event["action"] for event in events if event["type"] == "action")
    assert action["source"] == "kimi"


def test_search_results_are_checked_by_llm_for_topic_relevance(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    relevant = Paper(
        title="Dynamic Recommender Systems with Temporal User Interest",
        authors=["Ada Lovelace"],
        abstract="A dynamic recommendation method for time-varying user preference.",
        year=2025,
        venue="arXiv",
        source="arxiv",
        arxiv_id="2501.00001",
    )
    off_topic = Paper(
        title="Dynamic Recommender Systems for Urban Parking Recommendation",
        authors=["Alan Turing"],
        abstract="A dynamic recommendation system for parking allocation.",
        year=2025,
        venue="arXiv",
        source="arxiv",
        arxiv_id="2501.00002",
    )

    class FakeSearchService:
        def search(self, query, *, sources, **kwargs):
            return SearchResult(
                query=query,
                papers=[relevant, off_topic],
                sources=sources,
                source_status={sources[0]: {"status": "ok", "count": 2, "error": ""}},
            )

    class FakeLlm:
        available = True

        def chat_json(self, **kwargs):
            return {
                "relevant_ids": [relevant.id],
                "borderline_ids": [],
                "rejected_ids": [off_topic.id],
                "reason": "仅第一篇直接讨论动态推荐系统。",
            }

    intent = ResearchIntent(
        original_request="查近三年动态推荐系统论文",
        normalized_topic="dynamic recommender systems",
        cs_area="AI",
        queries=["dynamic recommender systems", "dynamic recommendation"],
        keywords=["dynamic", "recommender", "systems"],
        source_queries={"arxiv": ["dynamic recommender systems"]},
        recent_years=3,
    )
    engine.search_service = FakeSearchService()
    engine.llm = FakeLlm()

    papers = engine._search_for_intent(intent, limit=10)

    assert [paper.id for paper in papers] == [relevant.id]
    assert engine._last_search_debug["llm_relevance"]["used"] is True
    assert engine._last_search_debug["llm_relevance"]["rejected_count"] >= 1


def test_compound_debiasing_dynamic_recommender_search_requires_all_topic_modifiers(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    papers = [
        Paper(
            title="GUIDER: Uncertainty Guided Dynamic Re-ranking for Large Language Models Based Recommender Systems.",
            authors=["A"],
            year=2026,
            venue="AAAI",
            source="dblp",
            source_url="https://example.org/guider",
        ),
        Paper(
            title="Calibration in Dynamic Recommender Systems.",
            authors=["B"],
            year=2026,
            venue="WSDM",
            source="dblp",
            source_url="https://example.org/calibration",
        ),
        Paper(
            title="Enhancing New-item Fairness in Dynamic Recommender Systems.",
            authors=["C"],
            year=2025,
            venue="SIGIR",
            source="dblp",
            source_url="https://example.org/new-item-fairness",
        ),
        Paper(
            title="Ensuring User-side Fairness in Dynamic Recommender Systems.",
            authors=["D"],
            year=2024,
            venue="WWW",
            source="dblp",
            source_url="https://example.org/user-side-fairness",
        ),
    ]

    class FakeSearchService:
        def search(self, query, *, sources, **kwargs):
            return SearchResult(
                query=query,
                papers=papers,
                sources=sources,
                source_status={sources[0]: {"status": "ok", "count": len(papers), "error": ""}},
            )

        def enrich_missing_metadata(self, papers, *, max_items=12):
            return {"attempted": 0, "enriched": 0, "failures": []}

    class FakeLlm:
        available = False

    intent = ResearchIntent(
        original_request="搜索近三年去偏动态推荐系统论文",
        normalized_topic="debiasing in dynamic recommender systems",
        cs_area="AI",
        queries=["debiasing dynamic recommender systems", "fairness dynamic recommender systems"],
        keywords=["debiasing", "dynamic", "recommender", "fairness"],
        source_queries={"dblp": ["debiasing dynamic recommender systems", "fairness dynamic recommender systems"]},
        target_venue_ranks=["CCF-A", "CCF-B"],
        recent_years=3,
    )
    engine.search_service = FakeSearchService()
    engine.llm = FakeLlm()

    results = engine._search_for_intent(intent, limit=10)

    assert [paper.title for paper in results] == [
        "Enhancing New-item Fairness in Dynamic Recommender Systems.",
        "Ensuring User-side Fairness in Dynamic Recommender Systems.",
    ]
    assert engine._last_search_debug["topic_constraints"]["used"] is True
    assert engine._last_search_debug["topic_constraints"]["kept_count"] == 2


def test_document_action_uses_existing_papers_without_research(tmp_path):
    engine = ResearchAssistantEngine(
        store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs", upload_dir=tmp_path / "uploads"
    )
    engine.intent_analyzer = FakeIntentAnalyzer()
    engine.state.papers = [
        Paper(
            title="Dynamic Recommender Systems",
            authors=["Ada Lovelace"],
            abstract="A paper about dynamic recommendation.",
            year=2025,
            source="arxiv",
            source_url="https://example.org/paper",
        )
    ]

    def fail_search(*args, **kwargs):
        raise AssertionError("document generation should not trigger paper search when evidence exists")

    engine._search_for_intent = fail_search
    engine._generate_document = lambda request: {
        "markdown": {"name": "draft.md", "url": "/outputs/draft.md", "kind": "markdown"},
        "bibtex": {"name": "draft.bib", "url": "/outputs/draft.bib", "kind": "bibtex"},
        "claim_map": {"name": "draft.json", "url": "/outputs/draft.json", "kind": "json"},
        "files": [],
    }

    events = list(engine.handle_stream("根据已选论文生成 related work"))

    assert [event["type"] for event in events].count("intent") == 0
    assert any(event["type"] == "document" for event in events)


def test_followup_question_uses_current_papers_without_search_or_document(tmp_path):
    engine = ResearchAssistantEngine(
        store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs", upload_dir=tmp_path / "uploads"
    )
    engine.intent_analyzer = FakeFollowUpIntentAnalyzer()
    engine.state.papers = [
        Paper(
            title="Dynamic Recommender Systems",
            authors=["Ada Lovelace"],
            abstract="A paper about dynamic recommendation.",
            year=2025,
            source="arxiv",
            source_url="https://example.org/paper",
        )
    ]

    def fail_search(*args, **kwargs):
        raise AssertionError("follow-up question should not trigger paper search")

    def fail_document(*args, **kwargs):
        raise AssertionError("follow-up question should not generate a new document")

    engine._search_for_intent = fail_search
    engine._generate_document = fail_document

    events = list(engine.handle_stream("related work呢，怎么没有生成"))

    assert [event["type"] for event in events].count("intent") == 0
    assert not any(event["type"] == "document" for event in events)
    assert any(event["type"] == "answer" for event in events)


def test_followup_generated_file_question_points_to_document_outputs(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    engine.intent_analyzer = FakeFollowUpIntentAnalyzer()
    engine.state.papers = [
        Paper(
            title="Dynamic Recommender Systems",
            authors=["Ada Lovelace"],
            abstract="A paper about dynamic recommendation.",
            year=2025,
            source="arxiv",
            source_url="https://example.org/paper",
        )
    ]
    engine.state.generated_files = [
        {"name": "draft.md", "url": "/outputs/draft.md", "kind": "markdown"},
        {"name": "draft.bib", "url": "/outputs/draft.bib", "kind": "bibtex"},
    ]
    engine.state.last_generated_document = {
        "query": "dynamic recommender systems",
        "preview": "这是 related work 的开头。",
        "markdown": {"name": "draft.md", "url": "/outputs/draft.md", "kind": "markdown"},
        "bibtex": {"name": "draft.bib", "url": "/outputs/draft.bib", "kind": "bibtex"},
        "claim_map": {"name": "draft.json", "url": "/outputs/draft.json", "kind": "json"},
        "files": [],
    }

    events = list(engine.handle_stream("related work 在哪"))
    answer = next(event["content"] for event in events if event["type"] == "answer")

    assert "draft.md" in answer
    assert "/outputs/draft.md" in answer
    assert "related work" in answer.lower()


def test_debug_event_marks_existing_evidence_when_no_search_runs(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    engine.intent_analyzer = FakeFollowUpIntentAnalyzer()
    engine.state.papers = [
        Paper(
            title="Dynamic Recommender Systems",
            authors=["Ada Lovelace"],
            abstract="A paper about dynamic recommendation.",
            year=2025,
            source="arxiv",
            source_url="https://example.org/paper",
        )
    ]

    events = list(engine.handle_stream("这些论文里哪篇最值得先读"))
    debug = next(event["debug"] for event in events if event["type"] == "debug")

    assert debug["action"] == "answer"
    assert debug["search"]["used_existing_evidence"] is True


def test_engine_updates_paper_asset_state_and_excludes_from_evidence(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    keep = Paper(
        title="Useful Dynamic Recommendation",
        authors=["Ada Lovelace"],
        abstract="Useful evidence.",
        year=2026,
        source="openalex",
        source_url="https://example.org/useful",
    )
    reject = Paper(
        title="Rejected Dynamic Recommendation",
        authors=["Alan Turing"],
        abstract="Rejected evidence.",
        year=2026,
        source="openalex",
        source_url="https://example.org/rejected",
    )
    engine.store.save_papers([keep, reject])
    engine.state.papers = engine.store.load_papers()
    rejected_id = next(paper.id for paper in engine.state.papers if paper.title.startswith("Rejected"))

    result = engine.update_paper_asset_state(
        rejected_id,
        {
            "excluded": True,
            "exclusion_reason": "不符合当前课题。",
            "reading_status": "read",
            "importance": "low",
            "user_tags": ["排除"],
        },
    )
    evidence_titles = {paper.title for paper in engine._evidence_papers()}

    assert result["ok"] is True
    assert result["paper"]["excluded"] is True
    assert result["paper"]["reading_status"] == "read"
    assert evidence_titles == {"Useful Dynamic Recommendation"}


class FakePdfKimi:
    available = True

    def __init__(self):
        self.user = ""
        self.calls = []

    def chat_text(self, *, system, user, **kwargs):
        self.user = user
        self.calls.append(kwargs)
        return "该论文的主要贡献是一个可复现实验基准。[^sample-paper.pdf, p. 1^]【回答完毕】"


def test_uploaded_pdf_question_uses_relevant_page_context_and_kimi(tmp_path):
    engine = ResearchAssistantEngine(
        store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs", upload_dir=tmp_path / "uploads"
    )
    document = engine.upload_pdf(
        _text_pdf_bytes("The contribution is a reproducible experimental benchmark."),
        filename="sample-paper.pdf",
        session_id="default",
    )
    engine.llm = FakePdfKimi()

    answer = engine._answer_from_uploaded_pdfs("What is the contribution?")

    assert "【sample-paper.pdf，第 1 页】" in answer
    assert "sample-paper.pdf" in engine.llm.user
    assert "reproducible experimental benchmark" in engine.llm.user
    assert document["page_count"] == 1
    assert document["char_count"] > 0


def test_paper_fulltext_manager_extracts_cached_local_pdf(tmp_path):
    from paper_agent.core.fulltext import PaperFullTextManager

    manager = PaperFullTextManager(tmp_path)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(_text_pdf_bytes("The method uses a temporal debiasing objective."))
    paper = Paper(
        title="Temporal Debiasing for Dynamic Recommendation",
        authors=["Ada Lovelace"],
        source="arxiv",
        source_url="https://example.org/fulltext",
        local_pdf_path=str(pdf_path),
    )

    result = manager.ensure_text(paper)

    assert result.chunks
    assert paper.fulltext_status == "extracted"
    assert paper.local_text_path
    assert "temporal debiasing objective" in manager.read_chunks(paper)[0].text


def test_cached_pdf_keeps_library_status_when_text_extraction_fails(tmp_path):
    from paper_agent.core import fulltext
    from paper_agent.core.fulltext import PaperFullTextManager

    manager = PaperFullTextManager(tmp_path)
    pdf_path = manager.pdf_dir / "paper.pdf"
    pdf_path.write_bytes(_text_pdf_bytes("Readable if parser is installed."))
    paper = Paper(
        title="Cached PDF Without Parser",
        authors=["Ada Lovelace"],
        source="arxiv",
        local_pdf_path=str(pdf_path),
    )
    original_extract = fulltext.extract_pdf_text
    fulltext.extract_pdf_text = lambda data: (_ for _ in ()).throw(fulltext.PdfExtractionError("parser missing"))
    try:
        result = manager.cache_pdf(paper, extract_text=True)
    finally:
        fulltext.extract_pdf_text = original_extract

    assert result.chunks == []
    assert paper.local_pdf_path == str(pdf_path)
    assert paper.fulltext_status == "pdf_cached"
    assert "parser missing" in (paper.fulltext_error or "")


def test_uploaded_pdf_defaults_to_personal_library_folder(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")

    document = engine.upload_pdf(_text_pdf_bytes("A user uploaded library paper."), filename="paper.pdf")
    stored_path = engine.state.uploaded_documents[0].path

    assert document["local_pdf_path"] == stored_path
    assert (engine.fulltext_manager.pdf_dir / Path(stored_path).name).exists()
    assert Path(stored_path).parent == engine.fulltext_manager.pdf_dir


def test_download_paper_pdf_updates_personal_library_and_payload(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    paper = Paper(
        title="Open PDF for Personal Library",
        authors=["Ada Lovelace"],
        abstract="A paper with an open PDF.",
        year=2026,
        source="arxiv",
        pdf_url="https://example.org/paper.pdf",
    )
    engine.state.papers = [paper]
    engine.store.save_papers([paper])

    def fake_cache_pdf(target, *, extract_text=True):
        pdf_path = engine.fulltext_manager.pdf_dir / f"{target.id}.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n% cached test file\n")
        target.local_pdf_path = str(pdf_path)
        target.fulltext_status = "pdf_cached"
        target.fulltext_error = None
        return type("Result", (), {"summary": "已下载 PDF 到个人文献库。"})()

    engine.fulltext_manager.cache_pdf = fake_cache_pdf

    result = engine.download_paper_pdf(paper.id)
    restored = [item for item in engine.store.load_papers() if item.id == paper.id][0]
    served_path = engine.cached_paper_pdf_path(paper.id)

    assert result["ok"] is True
    assert result["paper"]["local_pdf_url"].startswith("/api/library/paper/file")
    assert Path(restored.local_pdf_path).parent == engine.fulltext_manager.pdf_dir
    assert served_path == Path(restored.local_pdf_path)


def test_download_paper_pdf_failure_prompts_manual_upload(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    paper = Paper(
        title="Closed PDF for Manual Upload",
        authors=["Ada Lovelace"],
        year=2026,
        source="dblp",
        source_url="https://example.org/closed",
    )
    engine.state.papers = [paper]

    result = engine.download_paper_pdf(paper.id)

    assert result["ok"] is False
    assert "上传原文" in result["message"]
    assert result["paper"]["manual_upload_needed"] is True
    assert result["paper"]["fulltext_status"] == "none"


def test_find_open_pdf_saves_explicit_scholar_resource(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    paper = Paper(
        title="Scholar Resource Lookup",
        authors=["Ada Lovelace"],
        year=2026,
        source="dblp",
        source_url="https://example.org/metadata",
    )
    engine.state.papers = [paper]
    engine.store.save_papers([paper])

    class ScholarWithPdf:
        def lookup_by_title(self, title, *, year=None):
            assert title == paper.title
            assert year == paper.year
            return Paper(
                title=title,
                authors=[],
                source="google_scholar",
                source_url="https://scholar.google.com/example",
                pdf_url="https://repository.example.org/paper.pdf",
            )

    engine.search_service.google_scholar = ScholarWithPdf()
    result = engine.find_open_pdf(paper.id)
    restored = next(item for item in engine.store.load_papers() if item.id == paper.id)

    assert result["ok"] is True
    assert result["paper"]["pdf_url"] == "https://repository.example.org/paper.pdf"
    assert restored.pdf_url == "https://repository.example.org/paper.pdf"


def test_open_local_library_folder_uses_managed_pdf_directory(tmp_path, monkeypatch):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    captured = {}

    monkeypatch.setattr(assistant_engine_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        assistant_engine_module.subprocess,
        "run",
        lambda command, check, timeout: captured.update({"command": command, "check": check, "timeout": timeout}),
    )

    result = engine.open_local_library_folder()

    assert result["ok"] is True
    assert result["folder"] == str((tmp_path / "paper_files" / "pdf").resolve())
    assert captured == {
        "command": ["open", str((tmp_path / "paper_files" / "pdf").resolve())],
        "check": True,
        "timeout": 10,
    }


def test_upload_pdf_can_attach_fulltext_to_retrieved_paper(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    paper = Paper(
        title="Manual Upload Evidence Paper",
        authors=["Ada Lovelace"],
        year=2026,
        source="dblp",
        source_url="https://example.org/manual",
    )
    engine.state.papers = [paper]
    engine.store.save_papers([paper])

    document = engine.upload_pdf(
        _text_pdf_bytes("Manual upload provides the full method and experiment evidence."),
        filename="manual.pdf",
        paper_id=paper.id,
    )
    restored = [item for item in engine.store.load_papers() if item.id == paper.id][0]
    chunks = engine.fulltext_manager.read_chunks(restored)

    assert document["linked_paper_id"] == paper.id
    assert document["linked_paper"]["fulltext_status"] == "extracted"
    assert restored.local_pdf_path
    assert restored.local_text_path
    assert "full method" in chunks[0].text


def test_manual_pdf_upload_resolves_a_merged_paper_alias(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    paper = Paper(
        title="Alias-safe Manual PDF Attachment",
        authors=["Ada Lovelace"],
        year=2026,
        source="openalex",
        source_url="https://example.org/alias-safe",
    )
    engine.store.save_papers([paper])
    with engine.store._connect() as conn:
        conn.execute(
            "INSERT INTO paper_id_aliases (alias_id, paper_id) VALUES (?, ?)",
            ("legacy-paper-id", paper.id),
        )

    document = engine.upload_pdf(
        _text_pdf_bytes("Manual binding must resolve the library record after a merge."),
        filename="alias-safe.pdf",
        paper_id="legacy-paper-id",
    )

    assert document["match"]["status"] == "manual"
    assert document["linked_paper_id"] == paper.id
    assert document["linked_paper"]["fulltext_status"] == "extracted"


def test_upload_pdf_automatically_matches_a_known_paper_from_first_page(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    paper = Paper(
        title="Automatic Matching for Debiased Recommendation",
        authors=["Ada Lovelace"],
        year=2026,
        source="openalex",
        source_url="https://example.org/automatic-match",
    )
    engine.state.papers = [paper]
    engine.store.save_papers([paper])

    document = engine.upload_pdf(
        _text_pdf_bytes("Automatic Matching for Debiased Recommendation This paper studies local full text matching."),
        filename="downloaded-paper.pdf",
    )

    assert document["match"]["status"] == "matched"
    assert document["match"]["confidence"] >= 0.82
    assert document["linked_paper_id"] == paper.id
    assert document["linked_paper"]["fulltext_status"] == "extracted"


def test_upload_pdf_ignores_generic_section_heading_when_matching_title(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    generic_heading = Paper(
        title="Introduction",
        authors=[],
        year=2024,
        source="google_scholar",
        source_url="https://example.org/introduction",
    )
    target = Paper(
        title="SPRec: Self-Play to Debias LLM-based Recommendation",
        authors=["Chongming Gao"],
        year=2025,
        source="arxiv",
        source_url="https://arxiv.org/abs/2412.09243",
    )
    engine.state.papers = [generic_heading, target]
    engine.store.save_papers([generic_heading, target])

    document = engine.upload_pdf(
        _text_pdf_bytes(
            "SPRec: Self-Play to Debias LLM-based Recommendation\n"
            "Introduction\nThis paper introduces a self-play framework for debiased recommendation."
        ),
        filename="3696410.3714524.pdf",
    )

    assert document["match"]["status"] == "matched"
    assert document["linked_paper_id"] == target.id


def test_upload_pdf_leaves_low_confidence_document_unlinked(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    paper = Paper(
        title="Automatic Matching for Debiased Recommendation",
        authors=["Ada Lovelace"],
        year=2026,
        source="openalex",
        source_url="https://example.org/automatic-match",
    )
    engine.state.papers = [paper]
    engine.store.save_papers([paper])

    document = engine.upload_pdf(
        _text_pdf_bytes("An unrelated vision benchmark with classification experiments."),
        filename="vision-benchmark.pdf",
    )

    assert document["match"]["status"] == "unmatched"
    assert "linked_paper_id" not in document


def test_match_local_library_pdfs_binds_previously_unlinked_upload(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    engine.upload_pdf(
        _text_pdf_bytes("Library Reconciliation for Recommendation Papers This paper provides a local PDF matching test."),
        filename="unknown-download.pdf",
    )
    paper = Paper(
        title="Library Reconciliation for Recommendation Papers",
        authors=["Ada Lovelace"],
        year=2026,
        source="openalex",
        source_url="https://example.org/library-reconciliation",
    )
    engine.state.papers = [paper]
    engine.store.save_papers([paper])

    summary = engine.match_local_library_pdfs()
    restored = next(item for item in engine.store.load_papers() if item.id == paper.id)

    assert summary["scanned"] == 1
    assert summary["matched"] == 1
    assert restored.fulltext_status == "extracted"


def test_writing_evidence_prefers_local_fulltext_chunks(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    pdf_path = tmp_path / "evidence.pdf"
    pdf_path.write_bytes(_text_pdf_bytes("Local full text says the method optimizes fairness in dynamic recommendation."))
    paper = Paper(
        title="Local Fulltext for Writing",
        authors=["Ada Lovelace"],
        abstract="Short abstract only.",
        year=2026,
        source="arxiv",
        local_pdf_path=str(pdf_path),
    )
    engine.state.papers = [paper]

    notes = engine._prepare_writing_evidence([paper], "写 related work")

    assert notes[0]["evidence_level"] == "local_fulltext"
    assert notes[0]["excerpt_count"] >= 1
    assert "optimizes fairness" in notes[0]["excerpts"][0]["text"]
    assert paper.fulltext_status == "extracted"


def test_generated_document_keeps_markdown_preview_and_quality_report(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    paper = Paper(
        title="Evidence-Aware Drafting",
        authors=["Ada Lovelace"],
        abstract="The work studies evidence-aware academic drafting.",
        year=2026,
        venue="ACL",
        source="fixture",
        source_url="https://example.org/evidence",
    )
    engine.state.papers = [paper]
    engine.state.evidence_paper_ids = [paper.id]
    engine.state.last_intent = ResearchIntent(
        original_request="写一段引言",
        normalized_topic="evidence aware drafting",
        cs_area="AI",
    )

    document = engine._generate_document("基于当前证据写一段引言")

    assert document["title"] == "引言"
    assert document["preview_markdown"].startswith("## 引言")
    assert "quality_report" in document
    assert document["quality_report"]["writing_kind"] == "introduction"


def test_agent_fulltext_tool_reads_existing_local_cache(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(_text_pdf_bytes("The experiments compare fairness and recommendation accuracy."))
    paper = Paper(
        title="Fairness Experiments in Dynamic Recommendation",
        authors=["Ada Lovelace"],
        source="arxiv",
        source_url="https://example.org/fulltext",
        local_pdf_path=str(pdf_path),
    )
    engine.state.papers = [paper]
    engine.llm = type("OfflineLlm", (), {"available": False})()
    engine.agent_runtime = IterativeAgentRuntime(
        FakeAgentPlanner(
            [
                {
                    "kind": "tool",
                    "tool": "paper_fulltext_read",
                    "arguments": {"paper_id": paper.id, "question": "实验比较了什么？"},
                }
            ]
        )
    )

    events = list(engine.handle_stream("详细分析这篇论文的实验"))
    answer = next(event["content"] for event in events if event["type"] == "answer")

    assert "experiments compare fairness" in answer
    assert paper.fulltext_status == "extracted"
    assert engine.store.stats()["fulltext_count"] == 1


def test_session_state_survives_engine_restart(tmp_path):
    store_path = tmp_path / "papers.sqlite"
    session_path = tmp_path / "sessions.json"
    output_dir = tmp_path / "outputs"
    upload_dir = tmp_path / "uploads"
    engine = ResearchAssistantEngine(
        store_path=store_path,
        output_dir=output_dir,
        upload_dir=upload_dir,
        session_store_path=session_path,
    )
    paper = Paper(
        title="Persistent Recommender Evidence",
        authors=["Ada Lovelace"],
        abstract="A persistent paper.",
        year=2026,
        source="dblp",
        source_url="https://example.org/persistent",
    )
    session = engine._session_state("project-a")
    session.papers = [paper]
    session.evidence_paper_ids = [paper.id]
    session.last_intent = ResearchIntent(
        original_request="检索动态推荐系统",
        normalized_topic="dynamic recommender systems",
        cs_area="AI",
        queries=["dynamic recommender systems"],
    )
    session.generated_files = [{"name": "draft.md", "url": "/outputs/draft.md", "kind": "markdown"}]
    session.last_generated_document = {"title": "Related Work", "markdown": session.generated_files[0]}
    session.conversation_history = [{"role": "user", "content": "检索动态推荐系统"}]
    engine.upload_pdf(_text_pdf_bytes("Persistent PDF text."), filename="paper.pdf", session_id="project-a")
    engine._persist_sessions()
    # SQLite is the source of truth. The compatibility JSON mirror must not
    # be required to restore a project after the service restarts.
    session_path.unlink()

    restored = ResearchAssistantEngine(
        store_path=store_path,
        output_dir=output_dir,
        upload_dir=upload_dir,
        session_store_path=session_path,
    )
    restored_session = restored._session_state("project-a")

    assert restored_session.papers[0].title == "Persistent Recommender Evidence"
    assert restored_session.evidence_paper_ids == [paper.id]
    assert restored_session.last_intent.normalized_topic == "dynamic recommender systems"
    assert restored_session.generated_files[0]["name"] == "draft.md"
    assert restored_session.uploaded_documents[0].name == "paper.pdf"
    assert restored_session.uploaded_documents[0].chunks[0].text
    assert restored.snapshot(session_id="project-a")["messages"] == [{"role": "user", "content": "检索动态推荐系统"}]


def test_generated_document_is_saved_as_an_editable_versioned_draft(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    paper = Paper(
        title="Evidence Grounded Recommendation", authors=["Ada Lovelace"],
        abstract="This paper studies evidence-grounded recommendation.", year=2026, source="dblp",
        source_url="https://example.org/evidence",
    )
    engine.state.papers = [paper]
    engine.state.evidence_paper_ids = [paper.id]
    document = engine._generate_document("基于当前证据写一段引言", writing_kind="introduction")

    assert document["draft_id"]
    draft = engine.get_draft(document["draft_id"])
    assert draft is not None
    assert draft["writing_kind"] == "introduction"
    assert draft["content_markdown"]
    updated = engine.update_draft(document["draft_id"], {"content_markdown": draft["content_markdown"] + "\n\n补充段落"})
    assert updated["version"] == 2
    assert len(engine.draft_versions(document["draft_id"])) == 2


def test_draft_export_revision_and_restore_keep_an_auditable_history(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    paper = Paper(
        title="Evidence Grounded Recommendation", authors=["Ada Lovelace"], year=2026,
        source="dblp", source_url="https://example.org/evidence",
    )
    engine.store.save_papers([paper])
    draft = engine.store.create_draft(
        "default",
        {
            "title": "Evidence Report",
            "writing_kind": "research_report",
            "content_markdown": "# Introduction\n\nOriginal evidence claim \\cite{lovelace2026evidence}.",
            "bibtex": "@article{lovelace2026evidence, title={Evidence Grounded Recommendation}, year={2026}}",
            "paper_ids": [paper.id],
        },
    )

    class RevisionLLM:
        available = True

        @staticmethod
        def chat_text(**kwargs):
            assert "Original evidence claim" in kwargs["user"]
            return "Revised evidence claim \\cite{lovelace2026evidence}."

    engine.llm = RevisionLLM()
    revised = engine.revise_draft(
        draft["id"],
        selected_text="Original evidence claim \\cite{lovelace2026evidence}.",
        instruction="改为更正式的学术表达",
    )

    assert revised["ok"] is True
    assert revised["draft"]["version"] == 2
    assert "Revised evidence claim" in revised["draft"]["content_markdown"]
    restored = engine.restore_draft_version(draft["id"], version=1)
    assert restored["ok"] is True
    assert restored["draft"]["version"] == 3
    assert "Original evidence claim" in restored["draft"]["content_markdown"]
    exported = engine.export_draft(draft["id"], format="docx")
    assert exported["ok"] is True
    assert (tmp_path / "outputs" / exported["file"]["name"]).exists()


def test_reset_session_clears_persisted_session_state(tmp_path):
    store_path = tmp_path / "papers.sqlite"
    session_path = tmp_path / "sessions.json"
    engine = ResearchAssistantEngine(store_path=store_path, session_store_path=session_path)
    session = engine._session_state("project-a")
    session.papers = [Paper(title="Temporary Evidence", authors=["A"], year=2026, source="dblp")]
    engine._persist_sessions()

    engine.reset_session("project-a")
    restored = ResearchAssistantEngine(store_path=store_path, session_store_path=session_path)

    assert restored._session_state("project-a").papers == []


def test_pdf_overview_samples_the_full_document_instead_of_only_the_opening():
    from paper_agent.core.pdf_reader import PdfChunk, relevant_chunks

    chunks = [PdfChunk(text=f"page {index}", page=index + 1, index=index) for index in range(40)]
    selected = relevant_chunks("解读一下全文", chunks, limit=6)

    assert [chunk.index for chunk in selected] == [0, 8, 16, 23, 31, 39]


def test_full_pdf_analysis_reads_all_chunks_then_caches_the_synthesis(tmp_path):
    from paper_agent.core.pdf_reader import PdfChunk

    engine = ResearchAssistantEngine(
        store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs", upload_dir=tmp_path / "uploads"
    )
    engine.upload_pdf(_text_pdf_bytes("Opening page."), filename="sample-paper.pdf")
    engine.state.uploaded_documents[0].chunks = [
        PdfChunk(text="A" * 8_000, page=1, index=0),
        PdfChunk(text="B" * 8_000, page=2, index=1),
        PdfChunk(text="C" * 8_000, page=3, index=2),
    ]
    engine.llm = FakePdfKimi()

    answer = engine._answer_from_uploaded_pdfs("解析一下这篇文章")
    cached_answer = engine._answer_from_uploaded_pdfs("解析一下这篇文章")

    assert "主要贡献" in answer
    assert cached_answer == answer
    labels = [call["label"] for call in engine.llm.calls]
    assert labels.count("pdf_full_map") == 3
    assert labels[-1] == "pdf_full_synthesis"
    assert "第 1 页" in engine.llm.user or "全文阅读笔记" in engine.llm.user


def test_uploaded_pdf_turn_routes_to_answer_in_auto_mode(tmp_path):
    engine = ResearchAssistantEngine(
        store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs", upload_dir=tmp_path / "uploads"
    )
    engine.intent_analyzer = FakePdfAnswerIntentAnalyzer()
    engine.upload_pdf(_text_pdf_bytes("The method uses contrastive learning."), filename="paper.pdf")
    engine._answer_from_uploaded_pdfs = lambda question: "PDF answer"

    events = list(engine.handle_stream("What method does this paper use?"))

    action = next(event["action"] for event in events if event["type"] == "action")
    answer = next(event["content"] for event in events if event["type"] == "answer")
    assert action["action"] == "answer"
    assert answer == "PDF answer"


def test_uploaded_pdf_summary_request_does_not_start_literature_search(tmp_path):
    engine = ResearchAssistantEngine(
        store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs", upload_dir=tmp_path / "uploads"
    )
    engine.intent_analyzer = FakeIntentAnalyzer()
    engine.upload_pdf(_text_pdf_bytes("This paper introduces a compact model."), filename="paper.pdf")
    engine._answer_from_uploaded_pdfs = lambda question: "PDF summary"

    events = list(engine.handle_stream("请总结这篇论文"))

    assert not any(event["type"] == "intent" for event in events)
    assert next(event["content"] for event in events if event["type"] == "answer") == "PDF summary"


class FakeCompositeIntentAnalyzer(FakeIntentAnalyzer):
    def analyze_request(
        self,
        request,
        *,
        mode="auto",
        has_papers=False,
        has_documents=False,
        has_generated_document=False,
        conversation_context="",
    ):
        self.last_action_trace = {"requested": "fake", "used": "fake", "fallback": False, "error": ""}
        self.last_research_trace = {}
        return RequestAnalysis(
            ActionIntent(request, "document", "先解析上传论文，再生成 related work。", 1.0, "fake"),
            tools=["pdf_read", "write_document"],
        )


def test_composite_pdf_read_and_related_work_runs_in_order_without_search(tmp_path):
    engine = ResearchAssistantEngine(
        store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs", upload_dir=tmp_path / "uploads"
    )
    engine.intent_analyzer = FakeCompositeIntentAnalyzer()
    engine.llm = FakePdfKimi()
    engine.upload_pdf(_text_pdf_bytes("The paper compares prior approaches."), filename="paper.pdf")

    events = list(engine.handle_stream("解析上传的论文，并生成 related work"))

    types = [event["type"] for event in events]
    assert "papers" not in types
    assert types.index("answer") < types.index("document")
    document = next(event for event in events if event["type"] == "document")
    assert document["markdown"]["name"].startswith("uploaded_pdf_document_")
    assert (tmp_path / "outputs" / document["markdown"]["name"]).exists()


class FakeDocumentInspectPlanner(FakeIntentAnalyzer):
    def analyze_request(self, request, **kwargs):
        self.last_action_trace = {"requested": "fake", "used": "fake", "fallback": False, "error": ""}
        self.last_research_trace = {}
        return RequestAnalysis(
            ActionIntent(request, "answer", "检查上一份文档。", 1.0, "fake"),
            tools=["document_inspect"],
        )


def test_document_inspection_reads_the_actual_last_generated_file(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    engine.intent_analyzer = FakeDocumentInspectPlanner()
    engine.llm = type("OfflineKimi", (), {"available": False})()
    draft_path = tmp_path / "outputs" / "draft.md"
    draft_path.write_text("# Related Work\n\nThis is the generated section.", encoding="utf-8")
    engine.state.last_generated_document = {
        "query": "dynamic recommender systems",
        "preview": "Related Work",
        "markdown": {"name": "draft.md", "url": "/outputs/draft.md", "kind": "markdown"},
    }

    events = list(engine.handle_stream("生成的内容是空的吗"))
    answer = next(event["content"] for event in events if event["type"] == "answer")

    assert "不是空文件" in answer
    assert "draft.md" in answer


def test_conversation_context_keeps_previous_turns_and_document_state(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    engine._remember_conversation("user", "请生成一份动态推荐系统的 related work")
    engine._remember_conversation("assistant", "已经生成文件 draft.md")
    engine.state.last_generated_document = {
        "query": "dynamic recommender systems",
        "preview": "A related work draft about temporal recommendation.",
        "markdown": {"name": "draft.md"},
    }

    context = engine._conversation_context()

    assert "动态推荐系统" in context
    assert "draft.md" in context
    assert "Latest generated document" in context


class FakeChatIntentAnalyzer(FakeIntentAnalyzer):
    def analyze_action(self, request, *, mode="auto", has_papers=False):
        return ActionIntent(request, "chat", "test", 1.0, "fake")


class FakeAgentPlanner:
    available = True

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.requests = []

    def chat_json(self, *, user, **kwargs):
        self.requests.append(user)
        return self.decisions.pop(0)


class FailingAgentPlanner:
    available = True

    def chat_json(self, **kwargs):
        raise RuntimeError("planner timeout")


def test_iterative_agent_searches_then_writes_from_the_observation(tmp_path):
    engine = ResearchAssistantEngine(
        store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs", upload_dir=tmp_path / "uploads"
    )
    planner = FakeAgentPlanner(
        [
            {
                "kind": "tool",
                "tool": "paper_search",
                "arguments": {
                    "normalized_topic": "dynamic recommender systems",
                    "queries": ["dynamic recommendation", "dynamic rec"],
                    "recent_years": 2,
                    "target_venues": ["ICLR"],
                },
                "reason": "先获得可引用论文。",
            },
            {
                "kind": "tool",
                "tool": "write_document",
                "arguments": {"request": "形成 related work 和 BibTeX"},
                "reason": "检索已完成，可以写作。",
            },
        ]
    )
    engine.agent_runtime = IterativeAgentRuntime(planner)
    paper = Paper(
        title="Dynamic Recommendation at Scale",
        authors=["Ada Lovelace"],
        abstract="Temporal recommendation with changing user preferences.",
        year=2025,
        venue="ICLR",
        source="dblp",
        source_url="https://example.org/paper",
    )
    captured = {}

    def search(intent):
        captured["intent"] = intent
        return [paper]

    engine._search_for_intent = search
    engine._confirm_writing_request = lambda _request: {"should_write": True, "reason": "测试中的明确写作请求。"}
    engine._generate_document = lambda request: {
        "title": "Related Work",
        "query": "dynamic recommender systems",
        "preview": "A grounded related work draft.",
        "markdown": {"name": "draft.md", "url": "/outputs/draft.md", "kind": "markdown"},
        "bibtex": {"name": "draft.bib", "url": "/outputs/draft.bib", "kind": "bibtex"},
        "claim_map": None,
        "files": [],
    }

    events = list(engine.handle_stream("查近两年 ICLR 动态推荐系统论文并写 related work"))

    assert captured["intent"].normalized_topic == "dynamic recommender systems"
    assert captured["intent"].target_venues == ["ICLR"]
    assert engine.state.last_tool_plan == ["paper_search", "write_document"]
    assert [event["type"] for event in events].index("papers") < [event["type"] for event in events].index("document")
    assert len(planner.requests) == 2
    assert "Dynamic Recommendation at Scale" in planner.requests[1]


def test_agent_refuses_an_unrequested_document_after_a_plain_literature_search(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    engine.agent_runtime = IterativeAgentRuntime(
        FakeAgentPlanner(
            [
                {
                    "kind": "tool",
                    "tool": "paper_search",
                    "arguments": {"normalized_topic": "object detection", "display_topic": "目标检测", "queries": ["object detection"], "recent_years": 3},
                },
                {
                    "kind": "tool",
                    "tool": "write_document",
                    "arguments": {"deliverable": "outline"},
                    "reason": "错误地把调研理解为大纲写作。",
                },
            ]
        )
    )
    engine._search_for_intent = lambda _intent: [
        Paper(title="A detector", authors=["Ada Lovelace"], abstract="Object detection research.", year=2025, source="dblp")
    ]
    engine._confirm_writing_request = lambda _request: {"should_write": False, "reason": "用户只要求调研进展。"}
    engine._generate_document = lambda _request: (_ for _ in ()).throw(AssertionError("不应生成文稿"))

    events = list(engine.handle_stream("调研一下近三年目标检测领域进展"))

    assert any(event["type"] == "papers" for event in events)
    assert not any(event["type"] == "document" for event in events)
    assert "没有生成文稿" in next(event["content"] for event in events if event["type"] == "answer")


def test_current_evidence_survey_writing_bypasses_agent_search(tmp_path):
    engine = ResearchAssistantEngine(
        store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs", upload_dir=tmp_path / "uploads"
    )

    class FakeSurveyWriteLlm:
        available = True
        model = "fake-kimi"

        def chat_json(self, **kwargs):
            return {
                "category": "document_writing",
                "subtask": "current_evidence_survey",
                "deliverable": "survey",
                "evidence_scope": "current_evidence",
                "tool_plan": ["write_document"],
                "action": "document",
                "reason": "用户要基于已调研文献写综述章节。",
                "confidence": 0.9,
                "normalized_topic": "survey method introduction chapter",
                "cs_area": "Interdisciplinary CS",
                "keywords": ["survey", "method", "chapter"],
                "queries": ["survey method introduction chapter"],
                "source_queries": {"dblp": ["survey method introduction chapter"]},
                "target_venues": [],
                "target_venue_ranks": [],
                "recent_years": None,
                "from_year": None,
                "to_year": None,
            }

    engine.intent_analyzer = IntentAnalyzer(llm=FakeSurveyWriteLlm())
    engine.agent_runtime = IterativeAgentRuntime(FailingAgentPlanner())
    engine.state.papers = [
        Paper(
            title="Dynamic Recommender Systems",
            authors=["Ada Lovelace"],
            abstract="A paper about dynamic recommendation.",
            year=2025,
            source="dblp",
            source_url="https://example.org/paper",
        )
    ]
    engine._search_for_intent = lambda intent: (_ for _ in ()).throw(AssertionError("不应重新检索 survey"))
    engine._generate_document = lambda request: {
        "markdown": {"name": "survey.md", "url": "/outputs/survey.md", "kind": "markdown"},
        "bibtex": {"name": "survey.bib", "url": "/outputs/survey.bib", "kind": "bibtex"},
        "claim_map": None,
        "files": [],
    }
    engine._answer_about_generated_outputs = lambda question: "已生成综述章节。"

    events = list(engine.handle_stream("按现在已调研的文献，写一篇survey综述方法介绍章节"))

    assert engine.state.last_tool_plan == ["write_document"]
    assert not any(event["type"] == "papers" for event in events)
    assert any(event["type"] == "document" for event in events)
    assert next(event["content"] for event in events if event["type"] == "answer") == "已生成综述章节。"


def test_agent_planner_failure_falls_back_to_legacy_search(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    engine.agent_runtime = IterativeAgentRuntime(FailingAgentPlanner())
    intent = ResearchIntent(
        original_request="搜索近三年去偏动态推荐系统",
        normalized_topic="debiasing dynamic recommender systems",
        cs_area="AI",
        queries=["debiasing dynamic recommender systems"],
        source_queries={
            "dblp": ["debiasing dynamic recommender systems"],
            "arxiv": ["debiasing dynamic recommender systems"],
            "semantic_scholar": ["debiasing dynamic recommender systems"],
            "google_scholar": ["debiasing dynamic recommender systems"],
        },
        recent_years=3,
        source="test",
    )
    engine.intent_analyzer.analyze_request = lambda *args, **kwargs: RequestAnalysis(
        action=ActionIntent("搜索近三年去偏动态推荐系统", "search", "fallback", 1.0, "test"),
        research=intent,
        tools=["paper_search"],
    )
    paper = Paper(
        title="Debiasing Dynamic Recommendation",
        authors=["Ada Lovelace"],
        abstract="A paper about debiasing dynamic recommendation.",
        year=2025,
        source="dblp",
        source_url="https://example.org/paper",
    )
    engine._search_for_intent = lambda research_intent: [paper]

    events = list(engine.handle_stream("搜索近三年去偏动态推荐系统"))

    assert any(event["type"] == "papers" for event in events)
    assert any("备用意图识别流程" in event.get("message", "") for event in events if event["type"] == "status")
    assert engine.state.last_tool_plan == ["paper_search"]
    assert engine.state.conversation_history.count({"role": "user", "content": "搜索近三年去偏动态推荐系统"}) == 1


def test_agent_free_chat_decision_is_not_reclassified_by_keyword_rules(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    engine.agent_runtime = IterativeAgentRuntime(
        FakeAgentPlanner(
            [
                {
                    "kind": "tool",
                    "tool": "free_chat",
                    "arguments": {"message": "搜索近三年去偏动态推荐系统"},
                    "reason": "误判为普通聊天。",
                }
            ]
        )
    )
    events = list(engine.handle_stream("搜索近三年去偏动态推荐系统"))

    assert not any(event["type"] == "papers" for event in events)
    assert engine.state.last_tool_plan == ["free_chat"]
    assert any(event["type"] == "answer" for event in events)


def test_agent_free_chat_forwards_model_deltas_before_the_final_answer(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")

    class StreamingChatLlm:
        available = True
        model = "fake-kimi"

        def stream_text(self, **kwargs):
            assert kwargs["label"] == "free_chat"
            yield "## Hello\n\n"
            yield "This reply arrives in pieces."

    engine.llm = StreamingChatLlm()
    engine.agent_runtime = IterativeAgentRuntime(
        FakeAgentPlanner(
            [{"kind": "tool", "tool": "free_chat", "arguments": {"message": "hello"}}]
        )
    )

    events = list(engine.handle_stream("hello"))

    assert [event["type"] for event in events].count("answer_start") == 1
    assert [event["delta"] for event in events if event["type"] == "answer_delta"] == [
        "## Hello\n\n",
        "This reply arrives in pieces.",
    ]
    assert next(event["content"] for event in events if event["type"] == "answer") == "## Hello\n\nThis reply arrives in pieces."


def test_iterative_agent_uses_current_evidence_for_followup_without_search(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    planner = FakeAgentPlanner(
        [
            {"kind": "tool", "tool": "evidence_answer", "arguments": {"question": "哪篇值得先读"}},
            {"kind": "final", "answer": "应先读 Dynamic Recommender Systems。"},
        ]
    )
    engine.agent_runtime = IterativeAgentRuntime(planner)
    engine.state.papers = [
        Paper(
            title="Dynamic Recommender Systems",
            authors=["Ada Lovelace"],
            abstract="A paper about dynamic recommendation.",
            year=2025,
            source="arxiv",
            source_url="https://example.org/paper",
        )
    ]
    engine._answer_from_papers = lambda question: "现有证据答复"
    engine._search_for_intent = lambda intent: (_ for _ in ()).throw(AssertionError("不应再次检索"))

    events = list(engine.handle_stream("哪篇值得先读？"))

    assert engine.state.last_tool_plan == ["evidence_answer"]
    assert not any(event["type"] == "papers" for event in events)
    assert next(event["content"] for event in events if event["type"] == "answer") == "现有证据答复"
    assert len(planner.requests) == 1


def test_iterative_agent_stops_duplicate_tool_calls(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    repeated_plan = {
        "normalized_topic": "dynamic recommender systems",
        "queries": ["dynamic recommender systems"],
        "source_queries": {},
        "recent_years": 2,
    }
    planner = FakeAgentPlanner(
        [
            {"kind": "tool", "tool": "paper_search", "arguments": repeated_plan},
            {"kind": "tool", "tool": "paper_search", "arguments": repeated_plan},
        ]
    )
    engine.agent_runtime = IterativeAgentRuntime(planner)
    paper = Paper(
        title="Dynamic Recommender Systems",
        authors=["Ada Lovelace"],
        abstract="A paper about dynamic recommendation.",
        year=2025,
        source="arxiv",
        source_url="https://example.org/paper",
    )
    calls = {"search": 0}

    def search_once(intent):
        calls["search"] += 1
        return [paper]

    engine._search_for_intent = search_once

    events = list(engine.handle_stream("搜索动态推荐系统并生成 related work"))

    assert calls["search"] == 1
    assert engine.state.last_tool_plan == ["paper_search"]
    assert any("重复调用 paper_search" in event.get("message", "") for event in events if event["type"] == "status")
    assert "已完成检索" in next(event["content"] for event in events if event["type"] == "answer")


def test_iterative_agent_pure_search_finishes_after_search_result(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    planner = FakeAgentPlanner(
        [
            {
                "kind": "tool",
                "tool": "paper_search",
                "arguments": {
                    "normalized_topic": "dynamic recommender systems",
                    "queries": ["dynamic recommender systems"],
                    "source_queries": {},
                    "recent_years": 2,
                },
            },
            {"kind": "final", "answer": "检索完成，已返回当前结果。"},
        ]
    )
    engine.agent_runtime = IterativeAgentRuntime(planner)
    paper = Paper(
        title="Dynamic Recommender Systems",
        authors=["Ada Lovelace"],
        abstract="A dynamic recommendation paper.",
        year=2025,
        source="arxiv",
        source_url="https://example.org/paper",
    )
    engine._search_for_intent = lambda intent: [paper]

    events = list(engine.handle_stream("搜索近两年动态推荐系统论文"))
    answer = next(event["content"] for event in events if event["type"] == "answer")

    assert len(planner.requests) == 2
    assert engine.state.last_tool_plan == ["paper_search"]
    assert answer == "检索完成，已返回当前结果。"


def test_search_stops_repeating_source_after_rate_limit(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")

    class RateLimitedSearchService:
        def __init__(self):
            self.calls = []

        def search(self, query, *, sources, **kwargs):
            self.calls.append((sources[0], query))
            if sources[0] == "arxiv":
                raise RuntimeError("arxiv: HTTP 429 for Rate exceeded.")
            return SearchResult(
                query=query,
                papers=[],
                sources=sources,
                source_status={sources[0]: {"status": "ok", "count": 0, "error": ""}},
            )

        def enrich_missing_metadata(self, papers, *, max_items=10):
            return {"attempted": 0, "enriched": 0, "failures": []}

    engine.search_service = RateLimitedSearchService()
    engine.llm = type("OfflineLlm", (), {"available": False})()
    intent = ResearchIntent(
        original_request="搜索动态推荐系统",
        normalized_topic="dynamic recommender systems",
        cs_area="AI",
        queries=["dynamic recommender systems"],
        source_queries={"arxiv": ["q1", "q2", "q3"]},
        recent_years=2,
    )

    engine._search_for_intent(intent, limit=10)

    arxiv_calls = [call for call in engine.search_service.calls if call[0] == "arxiv"]
    assert len(arxiv_calls) == 1


def test_discovery_profile_repairs_a_legacy_request_echo_topic(tmp_path):
    engine = ResearchAssistantEngine(store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs")
    request = "调研一下最近三年去偏推荐系统工作进展"
    engine.state.last_intent = ResearchIntent(
        original_request=request,
        normalized_topic=request,
        cs_area="Interdisciplinary CS",
    )
    engine.state.research_topics = [request]
    engine.intent_analyzer.repair_echoed_topic = lambda _request, _topic: {
        "normalized_topic": "debiasing recommender systems",
        "display_topic": "去偏推荐系统",
        "keywords": ["debiasing", "recommendation"],
        "queries": ["debiasing recommender systems"],
        "cs_area": "AI",
    }

    profile = engine._discovery_profile(engine.state)

    assert profile["primary_topic"] == "debiasing recommender systems"
    assert profile["display_topic"] == "去偏推荐系统"
    assert engine.state.research_topics == ["debiasing recommender systems"]


def _text_pdf_bytes(text):
    from io import BytesIO

    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    content = DecodedStreamObject()
    content.set_data(f"BT\n/F1 12 Tf\n72 720 Td\n({text}) Tj\nET\n".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(content)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
