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
    assert all(claim["citation_keys"] for claim in draft.claim_map)
    assert all(claim["papers"] for claim in draft.claim_map)


def test_bibtex_omits_truncated_source_fields_and_recovers_doi():
    paper = Paper(
        title="Debiasing Recommendation with Incomplete Source Metadata",
        authors=["Ada Lovelace", "Grace Hopper…"],
        abstract="The paper studies exposure bias in recommendation.",
        year=2026,
        venue="Proceedings of the ACM …",
        source="dblp",
        source_url="https://dl.acm.org/doi/abs/10.1145/1234567.8901234",
    )

    draft = RelatedWorkGenerator().generate(
        query="debiasing recommendation",
        papers=[paper],
        language="en",
    )

    assert "…" not in draft.bibtex
    assert "Grace Hopper" not in draft.bibtex
    assert "doi = {10.1145/1234567.8901234}" in draft.bibtex
    assert "journal =" not in draft.bibtex
    assert draft.quality_report["citation_metadata_issues"]
    assert draft.quality_report["draft_source"] == "fallback"


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
                "### Problem framing\n\n"
                "Retrieval-augmented generation connects parametric generation with external evidence, "
                "which helps position medical question answering as an evidence-grounded application. "
                "The retrieved material can constrain generation when the task requires claims to remain traceable to external sources \\cite{lewis2020retrieval}.\n\n"
                "### Research positioning\n\n"
                "This framing makes evidence retrieval a useful lens for comparing knowledge-intensive tasks, because it distinguishes what the model stores from what the system can verify at inference time. "
                "For medical question answering, this distinction is especially important when the response must be checked against a supporting record rather than treated as an unsupported generated statement \\cite{lewis2020retrieval}.\n\n"
                "The resulting comparison should therefore focus on the retrieval target, how retrieved evidence is incorporated, and how the final answer remains attributable to that evidence. "
                "These dimensions provide a manuscript-level rationale for comparing later evidence-grounded systems \\cite{lewis2020retrieval}."
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


def test_related_work_rejects_heading_only_model_output(monkeypatch):
    class HeadingOnlyWriter:
        available = True

        def draft(self, **kwargs):
            return "## Related Work\n\n### Problem framing\n\n### Method families\n\n### Positioning\n"

    monkeypatch.setattr(related_work_module, "KimiRelatedWorkWriter", HeadingOnlyWriter)
    paper = Paper(
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors=["Patrick Lewis"],
        abstract="Retrieval-augmented generation combines parametric and non-parametric memory.",
        year=2020,
        venue="NeurIPS",
        source="fixture",
    )

    draft = RelatedWorkGenerator().generate(
        query="retrieval augmented generation",
        papers=[paper],
        language="en",
        use_llm=True,
    )

    assert draft.quality_report["draft_source"] == "fallback"
    assert "Research Lineage and Problem Framing" in draft.content_markdown
    assert "### Method families\n\n###" not in draft.content_markdown


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
    assert captured["writing_plan"]["skill"] == "academic-writing"
    assert captured["writing_plan"]["resource"] == "resources/survey.md"
    assert "# 综述" in captured["writing_plan"]["instruction"]
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


def test_report_fallback_is_evidence_led_and_not_the_generic_section_template():
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
        Paper(
            title="Dynamic Feedback Debiasing for Sequential Recommendation",
            authors=["C"],
            abstract="The paper models feedback loops in dynamic sequential recommendation.",
            year=2025,
            venue="SIGIR",
            source="fixture",
            source_url="https://example.com/dynamic",
        ),
    ]

    draft = RelatedWorkGenerator().generate(
        query="debiasing recommender systems",
        papers=papers,
        language="zh",
        writing_request="结合目前的文献库和研究方向，写一篇研究报告",
        use_llm=False,
    )

    assert writing_plan_for("结合目前的文献库和研究方向，写一篇研究报告").kind == "report"
    assert draft.writing_kind == "report"
    assert "### 研究范围与证据边界" in draft.content_markdown
    assert "### 主要研究路线" in draft.content_markdown
    assert "#### 因果与反事实校正" in draft.content_markdown
    assert "#### 公平性、曝光与偏差校正" in draft.content_markdown
    assert "反事实/倾向性假设" in draft.content_markdown
    assert "优化目标与可能的效用权衡" in draft.content_markdown
    assert "本节基于当前证据整理可确认的研究线索" not in draft.content_markdown
    assert draft.content_markdown.count("围绕 debiasing recommender systems") == 1


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
