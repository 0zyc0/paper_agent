from __future__ import annotations

"""Deterministic guardrails around the LLM's semantic routing decision.

The model is useful for ambiguous research language.  It should not, however,
be able to turn an explicit "use the current evidence" request into an
expensive external search.  This module separates observable request signals
from the final executable tool plan.
"""

from dataclasses import dataclass
import re

from .models import normalize_text


CATEGORIES = {
    "chat",
    "literature_search",
    "evidence_qa",
    "paper_reading",
    "document_writing",
    "document_inspection",
    "discovery",
}

DELIVERABLES = {
    "none",
    "survey",
    "related_work",
    "introduction",
    "method_section",
    "experiment_section",
    "summary",
    "bibtex",
    "markdown",
    "report",
    "answer",
    "outline",
}


@dataclass(frozen=True)
class RoutingSignals:
    uses_current_evidence: bool
    requests_fresh_search: bool
    mentions_uploaded_pdf: bool
    inspects_generated_document: bool
    requests_writing: bool
    requests_discovery: bool
    has_papers: bool
    has_documents: bool
    has_generated_document: bool

    @property
    def evidence_scope(self) -> str:
        if self.uses_current_evidence and self.requests_fresh_search:
            return "mixed"
        if self.uses_current_evidence:
            return "current_evidence"
        if self.mentions_uploaded_pdf and not self.requests_fresh_search:
            return "uploaded_pdf"
        if self.requests_fresh_search:
            return "fresh_search"
        return "none"


@dataclass(frozen=True)
class RouteDecision:
    category: str
    subtask: str
    deliverable: str
    evidence_scope: str
    tools: list[str]
    search_required: bool
    reason: str


def detect_routing_signals(
    request: str,
    *,
    has_papers: bool,
    has_documents: bool,
    has_generated_document: bool,
) -> RoutingSignals:
    text = normalize_text(request).lower()
    compact = re.sub(r"\s+", "", text)
    current_markers = (
        "当前证据池", "当前文献池", "当前论文池", "当前结果", "本地文献库", "目前文献库", "现有文献库", "我的文献库",
        "按现在", "已调研", "已经调研", "已有", "已选", "当前", "这些论文",
        "这些文献", "基于这些", "基于当前", "根据这些", "根据当前",
        "current evidence", "current papers", "existing evidence", "retrieved papers",
    )
    writing_markers = (
        "生成", "写", "撰写", "起草", "形成", "导出", "章节", "引言", "摘要", "方法", "实验", "讨论", "结论",
        "综述", "相关工作", "大纲", "报告", "总结", "bibtex", ".bib", "markdown",
        "introduction", "abstract", "method", "methodology", "experiment", "evaluation", "discussion", "conclusion",
        "related work", "survey", "review", "outline", "section", "chapter", "report", "summary",
    )
    explicit_refresh_markers = (
        "重新检索", "重新搜索", "重新查", "重搜", "再搜", "补充检索", "补充搜索", "扩展检索", "新增文献",
        "最新论文", "最新文献", "fresh", "refresh", "new literature", "latest papers",
    )
    direct_search_markers = (
        "搜索", "检索", "查找", "查一下", "找一下",
        "search", "find papers", "retrieve", "look up",
    )
    exploratory_markers = ("调研", "investigate", "literature review")
    pdf_markers = ("上传", "上传pdf", "上传 pdf", "这篇论文", "这篇文章", "本文", "该pdf", "该 pdf", "uploaded pdf", "this paper")
    inspect_markers = ("生成的内容", "生成文件", "草稿", "文档在哪", "文件在哪", "预览", "是不是空", "是否为空", "没生成", "没有生成")
    discovery_markers = ("科研进展", "最新进展", "研究趋势", "研究方向推荐", "推荐方向", "发现模块", "research trends")

    requests_writing = _contains_any(text, compact, writing_markers)
    uses_current_evidence = has_papers and _contains_any(text, compact, current_markers)
    # "根据当前检索结果写..." refers to an existing workspace, rather than asking
    # the system to search again. A writing request with an existing paper pool
    # also treats the ambiguous Chinese word "调研" as a deliverable by default.
    # Only explicit refresh/search wording, or the LLM's semantic decision, can
    # add external retrieval back into that route.
    requests_fresh_search = _contains_any(text, compact, explicit_refresh_markers) or (
        not uses_current_evidence
        and (
            _contains_any(text, compact, direct_search_markers)
            or (
                not (has_papers and requests_writing)
                and _contains_any(text, compact, exploratory_markers)
            )
        )
    )

    return RoutingSignals(
        uses_current_evidence=uses_current_evidence,
        requests_fresh_search=requests_fresh_search,
        mentions_uploaded_pdf=has_documents and _contains_any(text, compact, pdf_markers),
        inspects_generated_document=has_generated_document and _contains_any(text, compact, inspect_markers),
        requests_writing=requests_writing,
        requests_discovery=_contains_any(text, compact, discovery_markers),
        has_papers=has_papers,
        has_documents=has_documents,
        has_generated_document=has_generated_document,
    )


def compile_route(
    request: str,
    *,
    signals: RoutingSignals,
    llm_category: str = "",
    llm_subtask: str = "",
    llm_deliverable: str = "",
    llm_reason: str = "",
    llm_requests_fresh_search: bool = False,
) -> RouteDecision:
    """Compile one executable plan. Explicit workspace references win over LLM guesses."""
    deliverable = _normalized_deliverable(llm_deliverable, request)
    category = _normalized_category(llm_category)
    # The model can recognize less literal requests such as "拓展一下文献范围".
    # It never overrides an explicit current-evidence scope without a user-level
    # refresh phrase, which protects existing-paper writing from model drift.
    requests_fresh_search = signals.requests_fresh_search or (
        not signals.uses_current_evidence and llm_requests_fresh_search
    )
    scope = "mixed" if signals.uses_current_evidence and requests_fresh_search else (
        "current_evidence" if signals.uses_current_evidence else (
            "uploaded_pdf" if signals.mentions_uploaded_pdf and not requests_fresh_search else (
                "fresh_search" if requests_fresh_search else "none"
            )
        )
    )

    if signals.inspects_generated_document and not signals.requests_writing:
        return RouteDecision("document_inspection", "generated_file_check", "answer", "generated_document", ["document_inspect"], False, _reason(llm_reason, "用户在检查已生成文档。"))

    if signals.requests_discovery and not signals.requests_writing:
        return RouteDecision("discovery", "research_updates", "report", scope, ["free_chat"], False, _reason(llm_reason, "用户希望查看研究进展与方向。"))

    # This is the key invariant: no search is allowed unless the user also asks
    # for fresh/new literature. It covers Chinese and English section names.
    if signals.uses_current_evidence and signals.requests_writing and not requests_fresh_search:
        return RouteDecision("document_writing", "current_evidence_writing", deliverable, "current_evidence", ["write_document"], False, _reason(llm_reason, "明确要求复用当前证据池写作。"))

    if signals.mentions_uploaded_pdf and signals.requests_writing and not requests_fresh_search and not signals.has_papers:
        return RouteDecision("document_writing", "uploaded_pdf_writing", deliverable, "uploaded_pdf", ["pdf_read", "write_document"], False, _reason(llm_reason, "明确要求基于上传 PDF 写作。"))

    if signals.mentions_uploaded_pdf and not signals.requests_writing and not requests_fresh_search:
        return RouteDecision("paper_reading", "uploaded_pdf_qa", "answer", "uploaded_pdf", ["pdf_read"], False, _reason(llm_reason, "请求围绕上传 PDF。"))

    if signals.requests_writing:
        if requests_fresh_search or not signals.has_papers:
            tools = ["paper_search", "write_document"]
            if signals.mentions_uploaded_pdf:
                tools.insert(0, "pdf_read")
            return RouteDecision("document_writing", "search_then_write", deliverable, scope if scope != "none" else "fresh_search", tools, True, _reason(llm_reason, "写作前需要补充新的文献证据。"))
        return RouteDecision("document_writing", "current_evidence_writing", deliverable, "current_evidence", ["write_document"], False, _reason(llm_reason, "当前论文池可直接用于写作。"))

    if requests_fresh_search:
        return RouteDecision("literature_search", "fresh_search", "none", "fresh_search", ["paper_search"], True, _reason(llm_reason, "用户明确要求检索新文献。"))

    if signals.has_papers:
        return RouteDecision("evidence_qa", "current_evidence_question", "answer", "current_evidence", ["evidence_answer"], False, _reason(llm_reason, "当前证据池足以回答该追问。"))
    if category == "paper_reading" and signals.has_documents:
        return RouteDecision("paper_reading", llm_subtask or "uploaded_pdf_qa", "answer", "uploaded_pdf", ["pdf_read"], False, _reason(llm_reason, "请求需要阅读上传 PDF。"))
    return RouteDecision("chat", llm_subtask or "general_chat", "none", "none", ["free_chat"], False, _reason(llm_reason, "未识别到需要检索或证据工具的请求。"))


def _normalized_category(value: str) -> str:
    cleaned = normalize_text(value).lower()
    return cleaned if cleaned in CATEGORIES else ""


def _normalized_deliverable(value: str, request: str) -> str:
    text = request.lower()
    if any(marker in text for marker in ("related work", "相关工作")):
        return "related_work"
    if any(marker in text for marker in ("survey", "综述", "review")):
        return "survey"
    if any(marker in text for marker in ("introduction", "引言")):
        return "introduction"
    if any(marker in text for marker in ("method", "方法")):
        return "method_section"
    if any(marker in text for marker in ("experiment", "实验", "evaluation", "评测")):
        return "experiment_section"
    if any(marker in text for marker in ("outline", "大纲")):
        return "outline"
    if any(marker in text for marker in ("bibtex", ".bib")):
        return "bibtex"
    if any(marker in text for marker in ("summary", "总结")):
        return "summary"
    cleaned = normalize_text(value).lower()
    if cleaned in DELIVERABLES:
        return cleaned
    return "report"


def _contains_any(text: str, compact: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text or marker.replace(" ", "") in compact for marker in markers)


def _reason(llm_reason: str, fallback: str) -> str:
    return normalize_text(llm_reason) or fallback
