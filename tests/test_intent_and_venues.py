from paper_agent.core.intent import IntentAnalyzer
from paper_agent.core.models import Paper
from paper_agent.tools.search import DblpClient, GoogleScholarClient, OpenAlexClient, PaperSearchService, SemanticScholarClient
from paper_agent.tools.venues import VenuePolicy


def test_llm_intent_uses_structured_topic_and_time_range_without_rule_rewrite():
    class EchoingLlm:
        available = True

        def chat_json(self, **kwargs):
            return {
                "category": "literature_search",
                "subtask": "paper_search",
                "deliverable": "none",
                "evidence_scope": "fresh_search",
                "tool_plan": ["paper_search"],
                "action": "search",
                "reason": "检索论文。",
                "confidence": 0.9,
                "normalized_topic": "debiasing recommender systems",
                "cs_area": "Interdisciplinary CS",
                "keywords": ["debiasing", "recommender systems"],
                "queries": ["debiasing recommender systems"],
                "source_queries": {},
                "target_venues": [],
                "target_venue_ranks": [],
                "recent_years": 3,
                "from_year": None,
                "to_year": None,
            }

    analysis = IntentAnalyzer(llm=EchoingLlm()).analyze_request(
        "调研一下最近三年去偏推荐系统工作进展", has_papers=False
    )

    assert analysis.research is not None
    assert analysis.research.normalized_topic == "debiasing recommender systems"
    assert analysis.research.recent_years == 3
    assert "debias" in " ".join(analysis.research.queries).lower()


def test_action_intent_without_kimi_keeps_safe_free_chat_instead_of_keyword_routing():
    action = IntentAnalyzer().analyze_action("帮我基于这些论文生成 related work 和 bib 文件", has_papers=True)

    assert action.action == "chat"


def test_action_intent_without_kimi_does_not_classify_sections_with_rules():
    action = IntentAnalyzer().analyze_action("基于这些论文写一段引言", has_papers=True)

    assert action.action == "chat"


def test_action_intent_without_kimi_does_not_classify_followups_with_rules():
    action = IntentAnalyzer().analyze_action("related work呢，怎么没有生成", has_papers=True)

    assert action.action == "chat"


class FakeUnifiedIntentLlm:
    available = True

    def __init__(self):
        self.calls = []

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "category": "document_writing",
            "subtask": "search_then_bibtex",
            "deliverable": "bibtex",
            "evidence_scope": "fresh_search",
            "tool_plan": ["paper_search", "write_document"],
            "action": "document",
            "reason": "用户需要检索并生成引用文件。",
            "confidence": 0.96,
            "normalized_topic": "dynamic recommender systems",
            "cs_area": "AI",
            "keywords": ["dynamic", "recommendation", "recommender"],
            "queries": ["dynamic recommender systems", "dynamic rec"],
            "source_queries": {
                "dblp": ["dynamic rec"],
                "arxiv": ["dynamic recommender systems"],
                "semantic_scholar": ["dynamic recommendation"],
                "google_scholar": ["dynamic rec"],
            },
            "target_venues": ["ICLR"],
            "target_venue_ranks": [],
            "recent_years": 2,
            "from_year": None,
            "to_year": None,
        }


def test_unified_request_analysis_uses_one_llm_call_for_routing_and_search_intent():
    llm = FakeUnifiedIntentLlm()
    analysis = IntentAnalyzer(llm=llm).analyze_request(
        "查近两年 ICLR 上动态推荐系统的论文并生成 bib", has_papers=False
    )

    assert len(llm.calls) == 1
    assert llm.calls[0]["label"] == "intent_request"
    assert analysis.action.action == "document"
    assert analysis.research is not None
    assert analysis.research.normalized_topic == "dynamic recommender systems"
    assert analysis.research.target_venues == ["ICLR"]


class FakeFollowUpPlannerLlm(FakeUnifiedIntentLlm):
    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "category": "evidence_qa",
            "subtask": "current_evidence_question",
            "deliverable": "answer",
            "evidence_scope": "current_evidence",
            "tool_plan": ["evidence_answer"],
            "action": "answer",
            "reason": "基于当前论文池回答追问。",
            "confidence": 0.95,
            "normalized_topic": "",
            "cs_area": "AI",
            "keywords": [],
            "queries": [],
            "source_queries": {},
            "target_venues": [],
            "target_venue_ranks": [],
            "recent_years": None,
            "from_year": None,
            "to_year": None,
        }


def test_contextual_question_uses_llm_routing_with_current_evidence():
    llm = FakeFollowUpPlannerLlm()
    analysis = IntentAnalyzer(llm=llm).analyze_request("这些论文的主要区别是什么", has_papers=True)

    assert analysis.action.action == "answer"
    assert analysis.research is None
    assert analysis.tools == ["evidence_answer"]
    assert len(llm.calls) == 1


def test_kimi_preserves_free_chat_even_when_a_paper_pool_exists():
    class ChatRouteLlm:
        available = True

        def chat_json(self, **_kwargs):
            return {
                "category": "chat",
                "subtask": "general_chat",
                "deliverable": "none",
                "evidence_scope": "none",
                "tool_plan": ["free_chat"],
                "reason": "普通自由交流。",
            }

    analysis = IntentAnalyzer(llm=ChatRouteLlm()).analyze_request("hello", has_papers=True)

    assert analysis.category == "chat"
    assert analysis.tools == ["free_chat"]
    assert analysis.research is None


class FakeSurveyWritingPlanLlm(FakeUnifiedIntentLlm):
    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "category": "document_writing",
            "subtask": "current_evidence_survey",
            "deliverable": "survey",
            "action": "document",
            "reason": "用户要求基于已有文献写综述章节。",
            "confidence": 0.95,
            "evidence_scope": "current_evidence",
            "tool_plan": ["write_document"],
            "normalized_topic": "survey method introduction chapter",
            "cs_area": "Interdisciplinary CS",
            "keywords": ["survey", "method", "chapter"],
            "queries": ["survey method introduction chapter"],
            "source_queries": {
                "openalex": ["survey method introduction chapter"],
                "dblp": ["survey"],
                "arxiv": ["survey method"],
                "semantic_scholar": ["survey method"],
                "google_scholar": ["survey method introduction"],
            },
            "target_venues": [],
            "target_venue_ranks": [],
            "recent_years": None,
            "from_year": None,
            "to_year": None,
        }


def test_current_evidence_survey_writing_does_not_trigger_search():
    llm = FakeSurveyWritingPlanLlm()
    analysis = IntentAnalyzer(llm=llm).analyze_request(
        "按现在已调研的文献，写一篇survey综述方法介绍章节",
        has_papers=True,
    )

    assert analysis.category == "document_writing"
    assert analysis.action.action == "document"
    assert analysis.tools == ["write_document"]
    assert analysis.research is None


class FakeWrongSurveyCategoryLlm(FakeSurveyWritingPlanLlm):
    def chat_json(self, **kwargs):
        data = super().chat_json(**kwargs)
        data["category"] = "document_writing"
        return data


class FakeIntroductionWritingLlm(FakeSurveyWritingPlanLlm):
    def chat_json(self, **kwargs):
        data = super().chat_json(**kwargs)
        data["deliverable"] = "introduction"
        data["subtask"] = "current_evidence_introduction"
        return data


def test_current_evidence_writing_uses_the_model_owned_route():
    llm = FakeWrongSurveyCategoryLlm()
    analysis = IntentAnalyzer(llm=llm).analyze_request(
        "按现在已调研的文献，写一篇survey综述方法介绍章节",
        has_papers=True,
    )

    assert analysis.category == "document_writing"
    assert analysis.tools == ["write_document"]
    assert analysis.research is None


def test_current_evidence_introduction_uses_explicit_model_scope():
    llm = FakeIntroductionWritingLlm()
    analysis = IntentAnalyzer(llm=llm).analyze_request(
        "基于当前证据池生成 introduction",
        has_papers=True,
    )

    assert analysis.category == "document_writing"
    assert analysis.deliverable == "introduction"
    assert analysis.evidence_scope == "current_evidence"
    assert analysis.search_required is False
    assert analysis.tools == ["write_document"]
    assert analysis.research is None


def test_current_retrieval_results_follow_model_plan():
    analysis = IntentAnalyzer(llm=FakeSurveyWritingPlanLlm()).analyze_request(
        "根据当前检索结果生成 introduction",
        has_papers=True,
    )

    assert analysis.category == "document_writing"
    assert analysis.tools == ["write_document"]
    assert analysis.search_required is False


def test_current_library_writing_follows_model_plan_not_keywords():
    analysis = IntentAnalyzer(llm=FakeSurveyWritingPlanLlm()).analyze_request(
        "我想基于目前文献库写一篇调研",
        has_papers=True,
    )

    assert analysis.category == "document_writing"
    assert analysis.evidence_scope == "current_evidence"
    assert analysis.tools == ["write_document"]
    assert analysis.search_required is False


class FakeNewTopicWritingLlm(FakeUnifiedIntentLlm):
    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "category": "document_writing",
            "subtask": "new_topic_survey",
            "deliverable": "survey",
            "evidence_scope": "fresh_search",
            "tool_plan": ["paper_search", "write_document"],
            "needs_fresh_literature": True,
            "topic_relation": "new_topic",
            "reason": "用户切换到目标检测这一新方向。",
            "normalized_topic": "object detection",
            "queries": ["object detection"],
            "source_queries": {},
        }


def test_kimi_can_request_fresh_search_for_a_named_new_writing_topic():
    analysis = IntentAnalyzer(llm=FakeNewTopicWritingLlm()).analyze_request(
        "请写一篇目标检测调研",
        has_papers=True,
        conversation_context="当前论文池主题：动态推荐系统",
    )

    assert analysis.tools == ["paper_search", "write_document"]
    assert analysis.search_required is True
    assert analysis.research is not None
    assert analysis.research.normalized_topic == "object detection"


def test_research_topic_from_agent_plan_is_not_rewritten_by_regex_rules():
    intent = IntentAnalyzer().research_from_plan(
        "按现在已调研的文献，写一篇survey综述方法介绍章节",
        {
            "normalized_topic": "survey method introduction chapter",
            "queries": ["survey method introduction chapter"],
            "source_queries": {"dblp": ["survey method introduction chapter"]},
        },
    )

    combined = " ".join([intent.normalized_topic, *intent.queries, *intent.source_queries["dblp"]]).lower()
    assert "survey method introduction chapter" in combined


def test_agent_request_echo_is_repaired_by_llm_before_becoming_a_research_topic():
    class TopicRepairLlm:
        available = True

        def chat_json(self, **kwargs):
            assert kwargs["label"] == "topic_repair"
            return {
                "normalized_topic": "debiasing recommender systems",
                "display_topic": "去偏推荐系统",
                "keywords": ["debiasing", "recommendation", "recommender systems"],
                "queries": ["debiasing recommender systems", "debiasing recommendation"],
                "cs_area": "AI",
            }

    request = "调研一下最近三年去偏推荐系统工作进展"
    intent = IntentAnalyzer(llm=TopicRepairLlm()).research_from_plan(
        request,
        {"normalized_topic": request, "queries": [request]},
    )

    assert intent.normalized_topic == "debiasing recommender systems"
    assert intent.display_topic == "去偏推荐系统"
    assert intent.queries == ["debiasing recommender systems", "debiasing recommendation"]


def test_agent_plan_with_chinese_search_terms_is_recompiled_by_kimi_before_retrieval():
    class QueryRepairLlm:
        available = True

        def chat_json(self, **kwargs):
            assert kwargs["label"] == "search_query_repair"
            return {
                "normalized_topic": "debiasing dynamic recommender systems",
                "display_topic": "动态去偏推荐系统",
                "keywords": ["debiasing", "dynamic recommendation", "recommender systems"],
                "queries": ["debiasing dynamic recommender systems", "debiasing temporal recommendation"],
                "source_queries": {
                    "openalex": ["debiasing dynamic recommender systems"],
                    "dblp": ["debiasing dynamic rec"],
                    "arxiv": ["debiasing sequential recommendation"],
                    "semantic_scholar": ["unbiased dynamic recommendation"],
                    "google_scholar": ["bias mitigation temporal recommender systems"],
                },
                "cs_area": "AI",
                "recent_years": 3,
                "from_year": None,
                "to_year": None,
            }

    intent = IntentAnalyzer(llm=QueryRepairLlm()).research_from_plan(
        "调研一下动态去偏推荐系统近三年研究进展",
        {
            "normalized_topic": "动态去偏推荐系统",
            "keywords": ["动态", "去偏", "推荐系统"],
            "queries": ["动态去偏推荐系统"],
            "source_queries": {"dblp": ["动态去偏推荐系统"]},
        },
    )

    assert intent.normalized_topic == "debiasing dynamic recommender systems"
    assert all(query.isascii() for query in intent.queries)
    assert intent.source_queries["dblp"] == ["debiasing dynamic rec"]
    assert intent.recent_years == 3


class FakePdfPlanLlm(FakeUnifiedIntentLlm):
    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "category": "document_writing",
            "subtask": "uploaded_pdf_writing",
            "deliverable": "related_work",
            "evidence_scope": "uploaded_pdf",
            "tool_plan": ["pdf_read", "write_document"],
            "action": "document",
            "reason": "先解读上传论文，再生成 related work。",
            "confidence": 0.98,
            "tools": ["pdf_read", "write_document"],
            "normalized_topic": "uploaded paper",
            "cs_area": "AI",
            "keywords": [],
            "queries": [],
            "source_queries": {},
            "target_venues": [],
            "target_venue_ranks": [],
            "recent_years": None,
            "from_year": None,
            "to_year": None,
        }


def test_kimi_can_plan_pdf_read_then_document_generation():
    llm = FakePdfPlanLlm()
    analysis = IntentAnalyzer(llm=llm).analyze_request(
        "解析上传的论文，并生成 related work", has_documents=True
    )

    assert analysis.action.action == "document"
    assert analysis.tools == ["pdf_read", "write_document"]
    assert analysis.research is None


class FakeIncompletePdfPlanLlm(FakeUnifiedIntentLlm):
    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return {"action": "answer", "reason": "解析上传论文", "confidence": "high"}


def test_incomplete_kimi_pdf_plan_falls_back_to_safe_free_chat():
    llm = FakeIncompletePdfPlanLlm()
    analysis = IntentAnalyzer(llm=llm).analyze_request("解析一下这篇文章", has_documents=True)

    assert analysis.action.action == "chat"
    assert analysis.action.source == "kimi"
    assert analysis.tools == ["free_chat"]
    assert analysis.research is None


def test_venue_policy_accepts_cs_arxiv_and_rejects_non_cs_arxiv():
    policy = VenuePolicy(path="missing-config.json")
    cs_paper = Paper(
        title="A CS Paper",
        authors=["A"],
        venue="arXiv",
        source="arxiv",
        arxiv_id="2601.00001",
        fields_of_study=["cs.CL"],
    )
    math_paper = Paper(
        title="A Math Paper",
        authors=["B"],
        venue="arXiv",
        source="arxiv",
        arxiv_id="2601.00002",
        fields_of_study=["math.PR"],
    )

    assert policy.decide(cs_paper).accepted
    assert not policy.decide(math_paper).accepted


def test_venue_policy_filters_explicit_target_venue():
    policy = VenuePolicy(path="missing-config.json")
    acl = Paper(title="ACL Paper", authors=["A"], venue="ACL", source="dblp", source_url="https://dblp.org")
    kdd = Paper(title="KDD Paper", authors=["B"], venue="KDD", source="dblp", source_url="https://dblp.org")

    accepted = policy.filter([acl, kdd], target_venues=["ACL"])

    assert [paper.title for paper in accepted] == ["ACL Paper"]


def test_venue_policy_filters_by_rank():
    policy = VenuePolicy(path="missing-config.json")
    cvpr = Paper(title="CVPR Paper", authors=["A"], venue="CVPR", source="dblp", source_url="https://dblp.org")
    iclr = Paper(title="ICLR Paper", authors=["B"], venue="ICLR", source="dblp", source_url="https://dblp.org")

    accepted = policy.filter([cvpr, iclr], target_venue_ranks=["CCF-A"])

    assert [paper.title for paper in accepted] == ["CVPR Paper"]


def test_venue_policy_suggests_area_scoped_rank_venues():
    policy = VenuePolicy(path="missing-config.json")

    venues = policy.venues_for_ranks(["CCF-A", "CCF-B"], "CV")

    assert "CVPR" in venues
    assert "ICCV" in venues
    assert len(venues) < 14


def test_dblp_info_parsing():
    paper = DblpClient()._paper_from_info(
        {
            "title": "A Study of Retrieval-Augmented Generation.",
            "authors": {"author": [{"text": "Ada Lovelace"}, {"text": "Alan Turing"}]},
            "venue": "ACL",
            "year": "2024",
            "doi": "10.0000/example",
            "ee": "https://example.org/paper",
            "key": "conf/acl/example2024",
        }
    )

    assert paper is not None
    assert paper.title == "A Study of Retrieval-Augmented Generation."
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.venue == "ACL"
    assert paper.year == 2024


def test_dblp_keeps_base_query_when_an_explicit_venue_is_requested():
    client = DblpClient()
    requested_urls = []

    def fake_get_json(url):
        requested_urls.append(url)
        return {"result": {"hits": {"hit": []}}}

    client._get_json_with_retry = fake_get_json
    client.search("dynamic rec", target_venues=["ICLR"])

    assert any("q=dynamic+rec&" in url for url in requested_urls)
    assert any("q=dynamic+rec+ICLR" in url for url in requested_urls)


def test_search_service_reports_partial_source_failures_without_dropping_other_sources():
    paper = Paper(title="Reliable Result", authors=["A"], year=2025, source="arxiv", arxiv_id="2501.00001")

    class SuccessfulSource:
        def search(self, *args, **kwargs):
            return [paper]

    class FailedSource:
        def search(self, *args, **kwargs):
            raise RuntimeError("rate limited")

    class EmptyScholar:
        available = False

        def search(self, *args, **kwargs):
            raise AssertionError("未配置时不应请求 Google Scholar")

    service = PaperSearchService()
    service.arxiv = SuccessfulSource()
    service.semantic_scholar = FailedSource()
    service.dblp = SuccessfulSource()
    service.google_scholar = EmptyScholar()

    result = service.search("dynamic recommendation", sources=["arxiv", "semantic_scholar", "dblp", "google_scholar"])

    assert result.papers
    assert result.source_status["arxiv"]["status"] == "ok"
    assert result.source_status["dblp"]["status"] == "ok"
    assert result.source_status["semantic_scholar"]["status"] == "error"
    assert result.source_status["google_scholar"]["status"] == "unavailable"


def test_search_service_enriches_dblp_metadata_with_semantic_scholar_abstract():
    paper = Paper(
        title="Enhancing New-item Fairness in Dynamic Recommender Systems.",
        authors=["A"],
        year=2025,
        venue="SIGIR",
        source="dblp",
        source_url="https://dblp.org/rec/example",
    )
    enriched = Paper(
        title="Enhancing New-item Fairness in Dynamic Recommender Systems",
        authors=["A"],
        abstract="This paper studies fairness in dynamic recommender systems.",
        year=2025,
        venue="SIGIR",
        source="semantic_scholar",
        source_url="https://semanticscholar.org/example",
        citation_count=7,
    )

    class FakeSemanticScholar:
        def lookup_by_title(self, title, *, year=None):
            return enriched

    service = PaperSearchService()
    service.semantic_scholar = FakeSemanticScholar()

    stats = service.enrich_missing_metadata([paper])

    assert stats["attempted"] == 1
    assert stats["enriched"] == 1
    assert paper.abstract == enriched.abstract
    assert paper.citation_count == 7


def test_openalex_work_parser_reconstructs_abstract_and_metadata():
    work = {
        "title": "Fairness in Dynamic Recommender Systems",
        "publication_year": 2025,
        "publication_date": "2025-05-01",
        "doi": "https://doi.org/10.1000/example",
        "ids": {"openalex": "https://openalex.org/W123"},
        "abstract_inverted_index": {
            "This": [0],
            "paper": [1],
            "studies": [2],
            "fairness": [3],
            "in": [4],
            "dynamic": [5],
            "recommendation": [6],
        },
        "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
        "primary_location": {
            "landing_page_url": "https://example.org/paper",
            "source": {"display_name": "SIGIR"},
        },
        "open_access": {"oa_url": "https://example.org/paper.pdf"},
        "cited_by_count": 12,
        "referenced_works_count": 30,
        "concepts": [{"display_name": "Recommender system"}],
    }

    paper = OpenAlexClient()._paper_from_work(work)

    assert paper is not None
    assert paper.abstract == "This paper studies fairness in dynamic recommendation"
    assert paper.doi == "10.1000/example"
    assert paper.venue == "SIGIR"
    assert paper.pdf_url == "https://example.org/paper.pdf"


def test_openalex_client_sends_api_key_and_mailto(monkeypatch):
    from paper_agent.tools import search as search_module

    calls = []

    def fake_get_json(url, *, timeout=25, **kwargs):
        calls.append({"url": url, "timeout": timeout, "kwargs": kwargs})
        return {"results": []}

    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-test-key")
    monkeypatch.setenv("OPENALEX_MAILTO", "researcher@example.com")
    monkeypatch.setattr(search_module, "get_json", fake_get_json)

    papers = OpenAlexClient().search("dynamic recommender systems")

    assert papers == []
    assert "api_key=openalex-test-key" in calls[0]["url"]
    assert "mailto=researcher%40example.com" in calls[0]["url"]


def test_google_scholar_client_keeps_explicit_pdf_resource(monkeypatch):
    from paper_agent.tools import search as search_module

    monkeypatch.setenv("SERPAPI_API_KEY", "serpapi-test-key")
    monkeypatch.setattr(
        search_module,
        "get_json",
        lambda *args, **kwargs: {
            "organic_results": [
                {
                    "title": "A Public Recommendation Paper",
                    "link": "https://publisher.example.org/article",
                    "publication_info": {"summary": "Ada Lovelace - SIGIR, 2025"},
                    "resources": [
                        {"title": "publisher.example.org", "file_format": "HTML", "link": "https://publisher.example.org/html"},
                        {"title": "repository.example.org", "file_format": "PDF", "link": "https://repository.example.org/paper.pdf"},
                    ],
                }
            ]
        },
    )

    paper = GoogleScholarClient().search("public recommendation paper")[0]

    assert paper.source_url == "https://publisher.example.org/article"
    assert paper.pdf_url == "https://repository.example.org/paper.pdf"


def test_semantic_scholar_retries_429_and_uses_local_config_key(monkeypatch):
    from paper_agent.tools import search as search_module

    calls = []
    sleeps = []

    def fake_get_json(url, *, headers=None, timeout=25, retries=2):
        calls.append({"url": url, "headers": headers or {}, "timeout": timeout, "retries": retries})
        if len(calls) == 1:
            raise search_module.HttpError('HTTP 429 for Semantic Scholar: {"message": "Too Many Requests"}')
        return {
            "data": [
                {
                    "title": "Rate Limited Result",
                    "authors": [{"name": "Ada Lovelace"}],
                    "abstract": "A recovered result.",
                    "year": 2026,
                    "publicationDate": "2026-01-01",
                    "venue": "Semantic Scholar",
                    "url": "https://example.org/paper",
                    "externalIds": {"DOI": "10.0000/test"},
                }
            ]
        }

    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.setattr(search_module.local_config, "SEMANTIC_SCHOLAR_API_KEY", "local-semantic-key", raising=False)
    monkeypatch.setattr(search_module, "get_json", fake_get_json)
    monkeypatch.setattr(search_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    papers = SemanticScholarClient().search("dynamic recommender systems")

    assert papers[0].title == "Rate Limited Result"
    assert calls[0]["headers"]["x-api-key"] == "local-semantic-key"
    assert len(calls) == 2
    assert any(seconds >= 3.0 for seconds in sleeps)
