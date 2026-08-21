from __future__ import annotations

"""Application-level orchestration between intent routing, skills, and tools."""

from dataclasses import dataclass

from .intent import IntentAnalyzer, RequestAnalysis
from ..skills.catalog import PaperSkill, PaperSkillCatalog
from ..tools.registry import PaperToolRegistry, ToolContext


@dataclass(frozen=True)
class AgentRequestContext:
    message: str
    mode: str
    has_papers: bool
    has_documents: bool
    has_generated_document: bool
    conversation_context: str = ""


@dataclass(frozen=True)
class PreparedTask:
    analysis: RequestAnalysis
    tools: list[str]
    skills: list[PaperSkill]

    def payload(self) -> dict:
        return {
            "category": self.analysis.category,
            "subtask": self.analysis.subtask,
            "deliverable": self.analysis.deliverable,
            "evidence_scope": self.analysis.evidence_scope,
            "tools": self.tools,
            "skills": [skill.payload() for skill in self.skills],
        }


class PaperAgentOrchestrator:
    """Prepare an executable, context-safe task without executing any tool."""

    def __init__(
        self,
        *,
        skill_catalog: PaperSkillCatalog | None = None,
        tool_registry: PaperToolRegistry | None = None,
    ) -> None:
        self.skill_catalog = skill_catalog or PaperSkillCatalog()
        self.tool_registry = tool_registry or PaperToolRegistry()

    def prepare(self, analyzer: IntentAnalyzer, context: AgentRequestContext) -> PreparedTask:
        try:
            analysis = analyzer.analyze_request(
                context.message,
                mode=context.mode,
                has_papers=context.has_papers,
                has_documents=context.has_documents,
                has_generated_document=context.has_generated_document,
                conversation_context=context.conversation_context,
            )
        except TypeError:
            # External analyzers from integrations may implement the older contract.
            analysis = analyzer.analyze_request(
                context.message,
                mode=context.mode,
                has_papers=context.has_papers,
                has_documents=context.has_documents,
            )

        tools = self.tool_registry.validate_plan(
            analysis.tools,
            ToolContext(
                has_papers=context.has_papers,
                has_documents=context.has_documents,
                has_generated_document=context.has_generated_document,
            ),
        )
        skills = self.skill_catalog.for_plan(tools, category=analysis.category)
        return PreparedTask(analysis=analysis, tools=tools, skills=skills)
