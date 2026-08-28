from __future__ import annotations

"""Schema validation for LLM-owned intent routing.

The language model is the only component that interprets the user's wording.
This module deliberately receives no request text: it validates a structured
route against workspace capabilities and never performs keyword or regex
classification.
"""

from dataclasses import dataclass


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

EVIDENCE_SCOPES = {"none", "current_evidence", "uploaded_pdf", "mixed", "fresh_search", "generated_document"}
ROUTABLE_TOOLS = {
    "free_chat",
    "paper_search",
    "pdf_read",
    "paper_fulltext_read",
    "evidence_answer",
    "write_document",
    "document_inspect",
}


@dataclass(frozen=True)
class RoutingContext:
    has_papers: bool
    has_documents: bool
    has_generated_document: bool


@dataclass(frozen=True)
class RouteDecision:
    category: str
    subtask: str
    deliverable: str
    evidence_scope: str
    tools: list[str]
    search_required: bool
    reason: str


def compile_route(
    *,
    context: RoutingContext,
    llm_category: str = "",
    llm_subtask: str = "",
    llm_deliverable: str = "",
    llm_evidence_scope: str = "",
    llm_tool_plan: object = None,
    llm_reason: str = "",
) -> RouteDecision:
    """Validate an LLM route without reinterpreting the user's message."""
    category = _enum(llm_category, CATEGORIES, "chat")
    deliverable = _enum(llm_deliverable, DELIVERABLES, "none")
    scope = _enum(llm_evidence_scope, EVIDENCE_SCOPES, "none")
    tools = _tool_plan(llm_tool_plan)
    if not tools:
        tools = _default_tools(category, scope, context)
    tools = _capability_safe_tools(tools, context)
    if not tools:
        tools = ["free_chat"]
        category, deliverable, scope = "chat", "none", "none"

    return RouteDecision(
        category=category,
        subtask=str(llm_subtask or "").strip() or _default_subtask(category),
        deliverable=deliverable,
        evidence_scope=scope,
        tools=tools,
        search_required="paper_search" in tools,
        reason=str(llm_reason or "").strip() or "由 Kimi 根据当前会话状态选择工具链。",
    )


def _enum(value: object, allowed: set[str], default: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in allowed else default


def _tool_plan(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    tools: list[str] = []
    for item in value:
        name = str(item or "").strip().lower()
        if name in ROUTABLE_TOOLS and name not in tools:
            tools.append(name)
    return tools


def _default_tools(category: str, scope: str, context: RoutingContext) -> list[str]:
    """Fallback only for an incomplete model response, based on its category."""
    if category == "literature_search":
        return ["paper_search"]
    if category == "document_writing":
        if scope == "fresh_search" or (not context.has_papers and not context.has_documents):
            return ["paper_search", "write_document"]
        if scope == "uploaded_pdf" and context.has_documents:
            return ["pdf_read", "write_document"]
        if scope == "mixed" and context.has_documents:
            return ["pdf_read", "paper_search", "write_document"]
        return ["write_document"]
    if category == "paper_reading":
        if context.has_documents:
            return ["pdf_read"]
        if context.has_papers:
            return ["paper_fulltext_read"]
        return ["free_chat"]
    if category == "evidence_qa":
        return ["evidence_answer"] if (context.has_papers or context.has_documents) else ["free_chat"]
    if category == "document_inspection":
        return ["document_inspect"] if context.has_generated_document else ["free_chat"]
    return ["free_chat"]


def _capability_safe_tools(tools: list[str], context: RoutingContext) -> list[str]:
    """Reject impossible calls; this is capability validation, not intent inference."""
    safe: list[str] = []
    for tool in tools:
        if tool == "pdf_read" and not context.has_documents:
            continue
        if tool == "paper_fulltext_read" and not context.has_papers:
            continue
        if tool == "evidence_answer" and not (context.has_papers or context.has_documents):
            continue
        if tool == "document_inspect" and not context.has_generated_document:
            continue
        if tool == "write_document" and not (context.has_papers or context.has_documents or "paper_search" in tools):
            continue
        safe.append(tool)
    return safe


def _default_subtask(category: str) -> str:
    return {
        "chat": "general_chat",
        "literature_search": "fresh_search",
        "evidence_qa": "current_evidence_question",
        "paper_reading": "paper_or_pdf_reading",
        "document_writing": "academic_writing",
        "document_inspection": "generated_file_check",
        "discovery": "research_updates",
    }.get(category, "general_chat")
