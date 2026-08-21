from paper_agent.core.models import Paper
from paper_agent.core.rank import rank_papers
from paper_agent.core.related_work import RelatedWorkGenerator, writing_plan_for
from paper_agent.core import related_work as related_work_module


def test_related_work_contains_real_citation_keys():
    papers = [
        Paper(
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            authors=["Patrick Lewis", "Ethan Perez"],
            abstract="Retrieval-augmented generation combines parametric and non-parametric memory.",
            year=2020,
            venue="NeurIPS",
            source="fixture",
            source_url="https://example.com/rag",
            citation_count=1000,
        ),
        Paper(
            title="Medical Question Answering with Evidence Retrieval",
            authors=["Jane Smith"],
            abstract="This paper studies evidence retrieval for medical question answering.",
            year=2024,
            venue="arXiv",
            source="fixture",
            source_url="https://example.com/medical-rag",
            citation_count=12,
        ),
    ]

    ranked = rank_papers(papers, "retrieval augmented generation medical question answering", limit=2)
    draft = RelatedWorkGenerator().generate(
        query="retrieval augmented generation medical question answering",
        papers=ranked,
        language="en",
    )

    assert "\\cite{" in draft.content_markdown
    assert "Retrieved Sources" not in draft.content_markdown
    assert "Source:" not in draft.content_markdown
    assert "@article" in draft.bibtex
    assert all(claim["paper_ids"] for claim in draft.claim_map)


def test_related_work_rejects_llm_source_list_and_falls_back_to_narrative(monkeypatch):
    class SourceListWriter:
        available = True

        def draft(self, **kwargs):
            return (
                "Retrieved Sources 1. lewis2020retrieval Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks "
                "(2020). Source: fixture. URL: https://example.com/rag"
            )

    monkeypatch.setattr(related_work_module, "KimiRelatedWorkWriter", SourceListWriter)
    papers = [
        Paper(
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            authors=["Patrick Lewis"],
            abstract="Retrieval-augmented generation combines parametric and non-parametric memory.",
            year=2020,
            venue="NeurIPS",
            source="fixture",
            source_url="https://example.com/rag",
            citation_count=1000,
        ),
        Paper(
            title="Evidence Retrieval for Medical QA",
            authors=["Jane Smith"],
            abstract="Evidence retrieval improves medical question answering.",
            year=2024,
            venue="ACL",
            source="fixture",
            source_url="https://example.com/medical",
            citation_count=20,
        ),
    ]

    draft = RelatedWorkGenerator().generate(
        query="retrieval augmented generation",
        papers=papers,
        language="en",
        use_llm=True,
    )

    assert draft.content_markdown.startswith("## Related Work")
    assert "Retrieved Sources" not in draft.content_markdown
    assert "Source:" not in draft.content_markdown
    assert "\\cite{" in draft.content_markdown


def test_related_work_accepts_llm_narrative_without_appending_sources(monkeypatch):
    class NarrativeWriter:
        available = True

        def draft(self, **kwargs):
            return (
                "## Related Work\n\n"
                "Retrieval-augmented generation connects parametric generation with external evidence, "
                "which helps position medical question answering as an evidence-grounded application \\cite{lewis2020retrieval}."
            )

    monkeypatch.setattr(related_work_module, "KimiRelatedWorkWriter", NarrativeWriter)
    paper = Paper(
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors=["Patrick Lewis"],
        abstract="Retrieval-augmented generation combines parametric and non-parametric memory.",
        year=2020,
        venue="NeurIPS",
        source="fixture",
        source_url="https://example.com/rag",
    )

    draft = RelatedWorkGenerator().generate(
        query="retrieval augmented generation",
        papers=[paper],
        language="en",
        use_llm=True,
    )

    assert "evidence-grounded application" in draft.content_markdown
    assert "Retrieved Sources" not in draft.content_markdown
    assert "Source:" not in draft.content_markdown


def test_writer_uses_requested_section_plan_and_passes_topic_to_llm(monkeypatch):
    captured = {}

    class SurveyWriter:
        available = True

        def draft(self, **kwargs):
            captured.update(kwargs)
            return (
                "## 任意标题\n\n"
                "### 范围与分类框架\n\n"
                "该方向的现有研究可以按问题设定与建模路线组织 \\cite{lewis2020retrieval}。\n\n"
                "### 比较、挑战与研究机会\n\n"
                "当前证据支持对不同路线进行保守比较 \\cite{lewis2020retrieval}。"
            )

    monkeypatch.setattr(related_work_module, "KimiRelatedWorkWriter", SurveyWriter)
    paper = Paper(
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors=["Patrick Lewis"],
        abstract="Retrieval-augmented generation combines parametric and non-parametric memory.",
        year=2020,
        venue="NeurIPS",
        source="fixture",
        source_url="https://example.com/rag",
    )

    draft = RelatedWorkGenerator().generate(
        query="retrieval augmented generation",
        papers=[paper],
        language="zh",
        use_llm=True,
        writing_request="基于当前论文写一篇 survey 综述",
    )

    assert captured["topic"] == "retrieval augmented generation"
    assert captured["writing_plan"]["kind"] == "survey"
    assert draft.title == "研究综述"
    assert draft.content_markdown.startswith("## 研究综述")
    assert draft.writing_kind == "survey"
    assert draft.quality_report["citation_uses"] >= 2


def test_current_library_research_request_uses_survey_plan_and_evidence_led_fallback():
    papers = [
        Paper(
            title="Counterfactual Learning for Debiased Recommendation",
            authors=["A"],
            abstract="We use propensity estimation and counterfactual learning to correct exposure bias in recommendation.",
            year=2023,
            venue="RecSys",
            source="fixture",
            source_url="https://example.com/counterfactual",
        ),
        Paper(
            title="Fair Exposure in Recommender Systems",
            authors=["B"],
            abstract="This work studies fairness and exposure bias for recommender systems.",
            year=2024,
            venue="WWW",
            source="fixture",
            source_url="https://example.com/fairness",
        ),
    ]

    draft = RelatedWorkGenerator().generate(
        query="debiasing recommender systems",
        papers=papers,
        language="zh",
        writing_request="我想基于目前文献库写一篇调研",
        use_llm=False,
    )

    assert writing_plan_for("我想基于目前文献库写一篇调研").kind == "survey"
    assert draft.writing_kind == "survey"
    assert "因果与反事实校正" in draft.content_markdown
    assert "公平性、曝光与偏差校正" in draft.content_markdown
    assert "Counterfactual Learning for Debiased Recommendation" in draft.content_markdown
    assert "\u56f4\u7ed5 debiasing recommender systems\uff0c\u672c\u8282\u57fa\u4e8e\u5f53\u524d\u8bc1\u636e\u6574\u7406" not in draft.content_markdown


def test_rank_papers_filters_zero_overlap_results():
    papers = [
        Paper(
            title="RAG Evaluation for Question Answering",
            authors=["A"],
            year=2025,
            venue="ACL",
            source="fixture",
            source_url="https://example.com/rag",
        ),
        Paper(
            title="Byte-Level Grammatical Error Correction",
            authors=["B"],
            year=2025,
            venue="ACL",
            source="fixture",
            source_url="https://example.com/gec",
        ),
    ]

    ranked = rank_papers(papers, "rag", limit=10)

    assert [paper.title for paper in ranked] == ["RAG Evaluation for Question Answering"]
