from __future__ import annotations

"""A small, guarded plan-act-observe loop for the research assistant.

Kimi is used as the planner, while Python remains the only component that can
actually execute a registered tool.  This keeps tool choice flexible without
letting model text masquerade as a search, a PDF read, or a generated file.
"""

from dataclasses import dataclass, field
import json
from typing import Any, Callable

from ..tools.llm import KimiClient


@dataclass
class ToolResult:
    name: str
    summary: str
    events: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    ok: bool = True
    terminal: bool = False

    def observation(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "ok": self.ok,
            "summary": self.summary[:5000],
            "answer": self.answer[:5000],
        }


@dataclass
class AgentTool:
    name: str
    description: str
    parameters: dict[str, str]
    execute: Callable[[dict[str, Any]], ToolResult]
    requires: str = "none"

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires": self.requires,
        }


@dataclass
class AgentDecision:
    kind: str
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    reason: str = ""


class IterativeAgentRuntime:
    """Ask the LLM for one next action at a time, with bounded iterations."""

    def __init__(self, llm: KimiClient, *, max_steps: int = 5) -> None:
        self.llm = llm
        self.max_steps = max_steps

    @property
    def available(self) -> bool:
        return self.llm.available

    def decide(
        self,
        *,
        user_message: str,
        mode: str,
        workspace_context: str,
        tools: list[AgentTool],
        observations: list[dict[str, Any]],
        step: int,
    ) -> AgentDecision:
        tool_catalog = [tool.prompt_payload() for tool in tools]
        system = (
            "You are the planner of a computer-science research assistant. "
            "Operate as a bounded plan-act-observe loop: select exactly one registered tool, "
            "then inspect its observation before selecting another tool or finishing. "
            "Never claim that a tool ran unless it appears in Observations. Never invent papers, "
            "citations, files, PDF pages, or tool results. Return only one valid JSON object. "
            "Prefer existing session evidence for follow-up questions; do not search again unless the user "
            "explicitly asks for new, changed, refreshed, broader, or external literature. "
            "If the user asks to search, retrieve, find, investigate, or check recent papers/literature, "
            "you must call paper_search; free_chat is only for ordinary conversation with no paper-search intent. "
            "Do not treat 'write a survey', 'survey section', or '综述章节' as search intent; survey is a deliverable there. "
            "Only use paper_search for survey when the user explicitly asks for survey papers or a fresh literature investigation. "
            "For a request that both searches and writes, call paper_search first and write_document after a "
            "successful search. For an uploaded-PDF request, use pdf_read; combine it with write_document only "
            "when the user also explicitly asks to create a document. "
            "For detailed questions about a specific retrieved paper's method, experiments, limitations, or claims, "
            "prefer paper_fulltext_read when available instead of answering from metadata only. "
            "Output formats such as BibTeX, markdown, and .bib are deliverables, not research topics. "
            "When finishing after a text-producing tool, faithfully use its answer and keep its citations intact."
        )
        user = {
            "user_message": user_message,
            "mode": mode,
            "workspace_context": workspace_context or "None",
            "step": step,
            "max_steps": self.max_steps,
            "registered_tools": tool_catalog,
            "observations": observations[-4:],
            "response_schema": {
                "kind": "tool or final",
                "tool": "registered tool name when kind is tool",
                "arguments": "object, empty when not needed",
                "reason": "short Chinese reason",
                "answer": "final response only when kind is final",
            },
        }
        data = self.llm.chat_json(
            system=system,
            user=json.dumps(user, ensure_ascii=False),
            temperature=0.1,
            max_tokens=1300,
            timeout=120,
            stream=False,
            label="agent_next_action",
        )
        kind = str(data.get("kind") or "").strip().lower()
        tool = str(data.get("tool") or "").strip().lower()
        arguments = data.get("arguments") if isinstance(data.get("arguments"), dict) else {}
        if kind == "tool" and tool in {item.name for item in tools}:
            return AgentDecision(
                kind="tool",
                tool=tool,
                arguments=arguments,
                reason=str(data.get("reason") or "").strip(),
            )
        if kind == "final":
            return AgentDecision(
                kind="final",
                answer=str(data.get("answer") or "").strip(),
                reason=str(data.get("reason") or "").strip(),
            )
        return AgentDecision(
            kind="invalid",
            reason="Kimi 未返回可执行的 Agent 决策。",
        )
