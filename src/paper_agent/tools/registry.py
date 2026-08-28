from __future__ import annotations

"""Declarative registry for executable paper-agent capabilities.

The registry deliberately contains contracts and availability checks only.  The
engine still owns session state and executes its already-tested operations.
This makes every capability visible to routing, future MCP adapters, and UI
status reporting without letting the model execute arbitrary Python.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ToolContext:
    has_papers: bool
    has_documents: bool
    has_generated_document: bool


@dataclass(frozen=True)
class PaperToolSpec:
    name: str
    title: str
    description: str
    parameters: dict[str, str]
    requires: str = "none"
    status: str = "正在执行任务。"
    skill: str = ""
    available_when: Callable[[ToolContext, set[str]], bool] = lambda _ctx, _plan: True

    def payload(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "parameters": self.parameters,
            "requires": self.requires,
            "skill": self.skill,
        }


class PaperToolRegistry:
    """A single source of truth for tool contracts and contextual availability."""

    def __init__(self) -> None:
        self._tools = {
            "paper_search": PaperToolSpec(
                "paper_search", "论文检索",
                "检索本地缓存与 OpenAlex、DBLP、arXiv、Semantic Scholar、Google Scholar。",
                {
                    "normalized_topic": "required concise English research noun phrase; never the full user request",
                    "display_topic": "required concise Chinese UI label for the same research topic",
                    "queries": "1-4 English query variants",
                    "source_queries": "optional source-specific query variants",
                    "target_venues": "explicit venue names only",
                    "target_venue_ranks": "CCF-A, CCF-B, SCI-Q1-Q3, arXiv",
                    "cs_area": "NLP, AI, ML, CV, DB, SE, Security, Systems, Networks, HCI, Graphics, Theory, Robotics, or Interdisciplinary CS",
                    "recent_years": "integer or null",
                    "from_year": "integer or null",
                    "to_year": "integer or null",
                },
                "明确的研究主题", "正在检索本地缓存与外部学术来源。", "literature-research",
            ),
            "pdf_read": PaperToolSpec(
                "pdf_read", "PDF 精读", "读取已上传 PDF，并从相关页面提取可追溯证据。",
                {"question": "user question"}, "当前会话已上传 PDF", "正在读取上传 PDF 的相关页面。", "pdf-reading",
                lambda ctx, _plan: ctx.has_documents,
            ),
            "paper_fulltext_read": PaperToolSpec(
                "paper_fulltext_read", "论文全文阅读", "读取个人文献库中已缓存或可下载的开放全文。",
                {"paper_id": "optional paper id", "title": "optional title", "question": "detailed question"},
                "当前证据池中有论文", "正在读取论文全文并提取证据。", "pdf-reading",
                lambda ctx, _plan: ctx.has_papers,
            ),
            "evidence_answer": PaperToolSpec(
                "evidence_answer", "证据问答", "仅根据当前论文池和已上传 PDF 回答，不发起新检索。",
                {"question": "user question"}, "当前论文池或上传 PDF", "正在依据当前证据池组织答案。", "evidence-grounding",
                lambda ctx, _plan: ctx.has_papers or ctx.has_documents,
            ),
            "write_document": PaperToolSpec(
                "write_document", "学术写作", "仅在用户明确要求写作交付物时生成章节、综述、Related Work、报告或 BibTeX；不能用于纯检索或调研请求。",
                {
                    "deliverable": "report, survey, related_work, introduction, method_section, experiment_section, summary, outline, bibliography, or general",
                }, "当前证据或同一计划中的论文检索", "正在基于当前证据生成写作草稿。", "academic-writing",
                lambda ctx, plan: ctx.has_papers or ctx.has_documents or "paper_search" in plan,
            ),
            "document_inspect": PaperToolSpec(
                "document_inspect", "文档核验", "读取最近生成文件的实际内容和完整性。",
                {"question": "user question"}, "当前会话已有生成文档", "正在核对最近生成文档。", "academic-writing",
                lambda ctx, _plan: ctx.has_generated_document,
            ),
            "free_chat": PaperToolSpec(
                "free_chat", "自由对话", "处理不依赖论文证据的普通交流。",
                {"message": "chat message"}, "none", "正在进行自由对话。", "",
            ),
        }

    def get(self, name: str) -> PaperToolSpec | None:
        return self._tools.get(name)

    def payloads(self) -> list[dict]:
        return [tool.payload() for tool in self._tools.values()]

    def validate_plan(self, plan: list[str], context: ToolContext) -> list[str]:
        """Remove unregistered or unavailable steps while preserving order."""
        unique_plan: list[str] = []
        plan_names = {str(name).strip().lower() for name in plan}
        for name in plan:
            normalized = str(name).strip().lower()
            tool = self.get(normalized)
            if not tool or normalized in unique_plan:
                continue
            if tool.available_when(context, plan_names):
                unique_plan.append(normalized)
        return unique_plan

    def statuses_for(self, plan: list[str]) -> list[str]:
        return [tool.status for name in plan if (tool := self.get(name))]

    def skills_for(self, plan: list[str]) -> list[str]:
        return [tool.skill for name in plan if (tool := self.get(name)) and tool.skill]
