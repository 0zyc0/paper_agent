from paper_agent.core.intent import IntentAnalyzer
from paper_agent.core.orchestration import AgentRequestContext, PaperAgentOrchestrator
from paper_agent.tools.registry import PaperToolRegistry, ToolContext


class _WritingRouteLlm:
    available = True

    def chat_json(self, **_kwargs):
        return {
            "category": "document_writing",
            "subtask": "current_evidence_writing",
            "deliverable": "introduction",
            "evidence_scope": "current_evidence",
            "tool_plan": ["write_document"],
            "reason": "基于当前证据生成引言。",
        }


def test_orchestrator_routes_writing_deliverable_inside_the_unified_skill():
    registry = PaperToolRegistry()
    orchestrator = PaperAgentOrchestrator(tool_registry=registry)

    task = orchestrator.prepare(
        IntentAnalyzer(llm=_WritingRouteLlm()),
        AgentRequestContext(
            message="基于当前证据池生成 introduction",
            mode="auto",
            has_papers=True,
            has_documents=False,
            has_generated_document=False,
        ),
    )

    assert task.tools == ["write_document"]
    assert [skill.name for skill in task.skills] == ["academic-writing"]
    assert task.skills[0].variant == "introduction"
    assert "# 引言" in task.skills[0].prompt()
    assert task.skills[0].payload()["instruction_file"] == "academic-writing/resources/introduction.md"
    assert "# 写作场景路由" in task.skills[0].prompt()
    assert task.analysis.search_required is False


def test_tool_registry_keeps_write_step_after_planned_search():
    registry = PaperToolRegistry()
    plan = registry.validate_plan(
        ["paper_search", "write_document"],
        ToolContext(has_papers=False, has_documents=False, has_generated_document=False),
    )

    assert plan == ["paper_search", "write_document"]


def test_tool_registry_hides_pdf_and_document_inspection_without_assets():
    registry = PaperToolRegistry()
    plan = registry.validate_plan(
        ["pdf_read", "document_inspect", "free_chat"],
        ToolContext(has_papers=False, has_documents=False, has_generated_document=False),
    )

    assert plan == ["free_chat"]
