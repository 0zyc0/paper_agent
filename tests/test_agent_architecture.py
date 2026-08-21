from paper_agent.core.intent import IntentAnalyzer
from paper_agent.core.orchestration import AgentRequestContext, PaperAgentOrchestrator
from paper_agent.tools.registry import PaperToolRegistry, ToolContext


def test_orchestrator_binds_current_evidence_writing_to_academic_skill():
    registry = PaperToolRegistry()
    orchestrator = PaperAgentOrchestrator(tool_registry=registry)

    task = orchestrator.prepare(
        IntentAnalyzer(),
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
