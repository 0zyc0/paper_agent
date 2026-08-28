from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterator
from uuid import uuid4

from .agent_runtime import AgentTool, IterativeAgentRuntime, ToolResult
from .discovery import DiscoveryService
from .draft_exports import DraftExportError, export_draft as export_draft_artifact
from .fulltext import PaperFullTextManager
from .intent import ActionIntent, IntentAnalyzer, ResearchIntent
from .orchestration import AgentRequestContext, PaperAgentOrchestrator
from ..tools.kimi_writer import KimiRelatedWorkWriter
from ..tools.llm import KimiClient
from ..tools.http_client import HttpError
from .models import Paper, normalize_text, utc_now_iso
from .pdf_reader import PdfChunk, PdfExtractionError, extract_pdf_text, relevant_chunks
from .rank import rank_papers
from .related_work import RelatedWorkGenerator, writing_plan_for
from .systematic_review import review_snapshot, write_review_export
from ..tools.search import PaperSearchService, dedupe_papers
from ..tools.registry import PaperToolRegistry, ToolContext
from ..storage.store import SQLitePaperStore
from ..tools.venues import VenuePolicy


# A search source is best-effort. The UI should return usable evidence from
# healthy sources instead of waiting indefinitely for one public endpoint.
SEARCH_SOURCE_BATCH_TIMEOUT = 8


@dataclass
class AssistantState:
    papers: list[Paper] = field(default_factory=list)
    last_intent: ResearchIntent | None = None
    last_answer: str = ""
    generated_files: list[dict] = field(default_factory=list)
    last_generated_document: dict | None = None
    last_debug: dict = field(default_factory=dict)
    last_tool_plan: list[str] = field(default_factory=list)
    # Search, PDF, and writing turns share this history so follow-ups keep their referent.
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    evidence_paper_ids: list[str] = field(default_factory=list)
    uploaded_documents: list["UploadedDocument"] = field(default_factory=list)
    research_topics: list[str] = field(default_factory=list)


@dataclass
class UploadedDocument:
    id: str
    name: str
    path: str
    page_count: int
    char_count: int
    chunks: list[PdfChunk]
    uploaded_at: str
    full_reading_notes: list[str] = field(default_factory=list)
    full_reading_analysis: str = ""

    def to_payload(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "local_pdf_path": self.path,
            "local_pdf_display_path": str(Path(self.path).resolve()),
            "page_count": self.page_count,
            "char_count": self.char_count,
            "uploaded_at": self.uploaded_at,
        }


class ResearchAssistantEngine:
    def __init__(
        self,
        *,
        store_path: str = "data/papers.sqlite",
        venue_config: str = "config/venues.json",
        output_dir: str = "outputs",
        upload_dir: str | Path | None = None,
        session_store_path: str | Path | None = None,
    ) -> None:
        self.store = SQLitePaperStore(store_path)
        self.venue_policy = VenuePolicy(venue_config)
        self.search_service = PaperSearchService()
        self.discovery_service = DiscoveryService()
        self.fulltext_manager = PaperFullTextManager(Path(store_path).parent)
        # Agent planning and the legacy structured fallback must observe the
        # same Kimi configuration.  Separate clients could otherwise make one
        # route appear unavailable while the other can still call the model.
        self.llm = KimiClient()
        self.intent_analyzer = IntentAnalyzer(self.llm)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = Path(upload_dir) if upload_dir else self.fulltext_manager.pdf_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.session_store_path = Path(session_store_path) if session_store_path else Path(store_path).parent / "sessions.json"
        self.session_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = AssistantState()
        self.sessions: dict[str, AssistantState] = {"default": self.state}
        self.agent_runtime = IterativeAgentRuntime(self.llm)
        self.tool_registry = PaperToolRegistry()
        self.orchestrator = PaperAgentOrchestrator(tool_registry=self.tool_registry)
        self._last_search_debug: dict = {}
        self._load_sessions()

    def discovery_feed(self, *, session_id: str = "default", topic: str = "") -> dict:
        state = self._session_state(session_id)
        self.state = state
        profile = self._discovery_profile(state, topic=topic)
        payload = self.discovery_service.discover_profile(profile, recent_years=1, limit=24)
        # _discovery_profile may repair a legacy persisted request-as-topic.
        self._persist_sessions()
        discovered_papers = [
            Paper(
                title=item.get("title") or "",
                authors=item.get("authors") or [],
                abstract=item.get("abstract") or None,
                year=item.get("year"),
                published_at=item.get("published_at"),
                venue=item.get("venue"),
                source=item.get("source") or "discovery",
                source_url=item.get("url") or None,
                citation_count=item.get("citation_count"),
            )
            for item in payload.get("papers", [])
            if isinstance(item, dict) and item.get("title")
        ]
        if discovered_papers:
            self.store.save_papers(discovered_papers, query=profile["primary_topic"], topic=profile["primary_topic"])
        return payload

    def upload_pdf(
        self,
        data: bytes,
        *,
        filename: str,
        session_id: str = "default",
        paper_id: str | None = None,
    ) -> dict:
        """Store one PDF in the current session after extracting its readable text."""
        self.state = self._session_state(session_id)
        safe_name = _safe_pdf_filename(filename)
        if not safe_name.lower().endswith(".pdf"):
            raise PdfExtractionError("只能上传 .pdf 文件。")
        extraction = extract_pdf_text(data)
        document_id = uuid4().hex
        linked_paper = self._paper_by_id(paper_id or "") if paper_id else None
        match = {"status": "manual" if paper_id else "unmatched", "confidence": 0.0, "reason": ""}
        attachment_summary = ""
        if paper_id and not linked_paper:
            raise PdfExtractionError("未找到要补全的论文，请刷新文献库后重试。")
        if not linked_paper:
            linked_paper, match = self._match_uploaded_pdf(extraction, filename)
        if linked_paper:
            result = self.fulltext_manager.attach_uploaded_pdf(linked_paper, data, extraction=extraction)
            self.store.save_papers([linked_paper])
            self._upsert_session_paper(linked_paper)
            path = Path(linked_paper.local_pdf_path or "")
            attachment_summary = result.summary
        else:
            path = self.upload_dir / f"uploaded_{document_id}_{safe_name}"
            path.write_bytes(data)
        document = UploadedDocument(
            id=document_id,
            name=safe_name,
            path=str(path),
            page_count=extraction.page_count,
            char_count=extraction.char_count,
            chunks=extraction.chunks,
            uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        )
        self.state.uploaded_documents.append(document)
        self._persist_sessions()
        payload = document.to_payload()
        payload["match"] = match
        if linked_paper:
            payload["linked_paper_id"] = linked_paper.id
            payload["linked_paper_title"] = linked_paper.title
            payload["attachment_summary"] = attachment_summary
            payload["linked_paper"] = self._paper_payloads([linked_paper])[0]
        return payload

    def match_local_library_pdfs(self, *, session_id: str = "default") -> dict:
        """Attach already imported local PDFs to known papers when the match is clear.

        This only scans the project's managed PDF directory. Ambiguous files are
        deliberately left untouched so an unrelated paper is never bound by a
        filename guess.
        """
        self.state = self._session_state(session_id)
        managed_dir = self.fulltext_manager.pdf_dir
        known_paths = {
            str(Path(paper.local_pdf_path).resolve())
            for paper in self.store.load_papers()
            if paper.local_pdf_path and Path(paper.local_pdf_path).exists()
        }
        match_candidates = [*self.state.papers, *self.store.load_papers()]
        summary = {"scanned": 0, "matched": 0, "unmatched": 0, "ambiguous": 0, "failed": 0, "papers": [], "details": []}
        for path in sorted(managed_dir.glob("*.pdf")):
            resolved = str(path.resolve())
            if resolved in known_paths:
                continue
            summary["scanned"] += 1
            try:
                extraction = extract_pdf_text(path.read_bytes())
            except Exception as exc:
                summary["failed"] += 1
                summary["details"].append({"name": path.name, "status": "failed", "reason": str(exc)[:160]})
                continue
            paper, match = self._match_uploaded_pdf(extraction, path.name, candidates=match_candidates)
            status = str(match.get("status") or "unmatched")
            if not paper:
                summary[status if status in {"unmatched", "ambiguous"} else "unmatched"] += 1
                summary["details"].append({"name": path.name, **match})
                continue
            result = self.fulltext_manager.attach_uploaded_pdf(paper, path.read_bytes(), extraction=extraction)
            self.store.save_papers([paper])
            self._upsert_session_paper(paper)
            summary["matched"] += 1
            summary["papers"].append(self._paper_payloads([paper])[0])
            summary["details"].append({"name": path.name, "status": "matched", "paper_id": paper.id, "title": paper.title, "reason": result.summary, **match})
        if summary["matched"]:
            self._persist_sessions()
        return summary

    def handle_stream(
        self,
        message: str,
        *,
        mode: str = "auto",
        session_id: str = "default",
        evidence_paper_ids: list[str] | None = None,
    ) -> Iterator[dict]:
        """Run one request through the model-owned intent route.

        Python validates tool availability but never reclassifies the user's
        wording after Kimi has produced the structured route.
        """
        try:
            # The standard path is the plan-act-observe Agent. The older
            # structured router remains a safe fallback when Kimi is unavailable.
            if self.agent_runtime.available:
                yield from self._handle_agent_stream(
                    message,
                    mode=mode,
                    session_id=session_id,
                    evidence_paper_ids=evidence_paper_ids,
                )
                return
            yield from self._handle_legacy_stream(
                message,
                mode=mode,
                session_id=session_id,
                evidence_paper_ids=evidence_paper_ids,
            )
        finally:
            self._persist_sessions()

    def _handle_legacy_stream(
        self,
        message: str,
        *,
        mode: str = "auto",
        session_id: str = "default",
        evidence_paper_ids: list[str] | None = None,
    ) -> Iterator[dict]:
        self.state = self._session_state(session_id)
        kimi_log_start = KimiClient.log_size()
        if evidence_paper_ids is not None:
            self.state.evidence_paper_ids = evidence_paper_ids
            if evidence_paper_ids and not self.state.papers:
                wanted = set(evidence_paper_ids)
                self.state.papers = [paper for paper in self.store.load_papers() if paper.id in wanted]
        message = message.strip()
        if not message:
            yield {"type": "error", "message": "请输入要调研或提问的内容。"}
            return

        prefetched_intent = None
        request_analysis = None
        prepared_task = None
        tool_plan: list[str] = []
        if hasattr(self.intent_analyzer, "analyze_request"):
            prepared_task = self.orchestrator.prepare(
                self.intent_analyzer,
                AgentRequestContext(
                    message=message,
                    mode=mode,
                    has_papers=bool(self.state.papers),
                    has_documents=bool(self.state.uploaded_documents),
                    has_generated_document=bool(self.state.last_generated_document),
                    conversation_context=self._conversation_context(),
                ),
            )
            request_analysis = prepared_task.analysis
            action_intent = request_analysis.action
            prefetched_intent = request_analysis.research
            tool_plan = prepared_task.tools
        else:  # Keep lightweight test doubles and external integrations compatible.
            action_intent = self.intent_analyzer.analyze_action(
                message,
                mode=mode,
                has_papers=bool(self.state.papers),
            )
        if not tool_plan:
            tool_plan = _fallback_tool_plan(
                action_intent.action,
                has_documents=bool(self.state.uploaded_documents),
                has_evidence=bool(self._evidence_papers()),
            )
            tool_plan = self.tool_registry.validate_plan(
                tool_plan,
                ToolContext(
                    has_papers=bool(self.state.papers),
                    has_documents=bool(self.state.uploaded_documents),
                    has_generated_document=bool(self.state.last_generated_document),
                ),
            )
        self.state.last_tool_plan = list(tool_plan)
        self._remember_conversation("user", message)
        action = action_intent.action
        yield {
            "type": "action",
            "action": {
                "action": action,
                "reason": action_intent.reason,
                "confidence": action_intent.confidence,
                "source": action_intent.source,
                "tools": tool_plan,
                "category": getattr(request_analysis, "category", ""),
                "subtask": getattr(request_analysis, "subtask", ""),
                "evidence_scope": getattr(request_analysis, "evidence_scope", "none"),
                "skills": [skill.payload() for skill in (prepared_task.skills if prepared_task else [])],
            },
        }
        if prepared_task and prepared_task.skills:
            yield {
                "type": "skills",
                "skills": [skill.payload() for skill in prepared_task.skills],
            }
        yield {
            "type": "status",
            "message": f"收到请求，判断动作为：{action}（{action_intent.source}），计划调用：{' -> '.join(tool_plan)}。",
        }

        if action == "chat":
            yield {"type": "status", "message": "正在调用 Kimi 进行自由聊天。"}
            answer = self._free_chat(message)
            self.state.last_answer = answer
            self._remember_conversation("assistant", answer)
            yield {"type": "answer", "content": answer}
            yield {"type": "debug", "debug": self._finalize_debug(action=action, kimi_log_start=kimi_log_start, search_used=False)}
            return

        run_pdf_read = "pdf_read" in tool_plan and bool(self.state.uploaded_documents)
        run_document = "write_document" in tool_plan
        should_search = "paper_search" in tool_plan
        run_document_inspect = "document_inspect" in tool_plan and bool(self.state.last_generated_document)

        if run_document_inspect:
            yield {"type": "status", "message": "正在读取上一份生成文档并核对实际内容。"}
            answer = self._inspect_generated_document(message)
            self.state.last_answer = answer
            self._remember_conversation("assistant", answer)
            yield {"type": "answer", "content": answer}
            if not run_document and not should_search and not run_pdf_read:
                yield {"type": "debug", "debug": self._finalize_debug(action=action, kimi_log_start=kimi_log_start, search_used=False)}
                return

        if run_pdf_read:
            status = (
                "正在分段覆盖全文并综合论文解读，篇幅较长时需要多次 Kimi 调用。"
                if _is_full_pdf_analysis_request(message)
                else "正在读取上传 PDF，并提取与请求相关的页面。"
            )
            yield {"type": "status", "message": status}
            pdf_answer = self._answer_from_uploaded_pdfs(message)
            self.state.last_answer = pdf_answer
            self._remember_conversation("assistant", pdf_answer)
            yield {"type": "answer", "content": pdf_answer}

        if should_search:
            yield {"type": "status", "message": "正在识别研究方向、年份范围和目标会议/期刊。"}
            intent = prefetched_intent or self.intent_analyzer.analyze(message)
            self.state.last_intent = intent
            self._remember_research_topic(intent.normalized_topic, intent.display_topic)
            yield {"type": "intent", "intent": self._intent_payload(intent)}

            yield {"type": "status", "message": "正在先查本地缓存，再检索 DBLP、arXiv、Semantic Scholar 和 Google Scholar。"}
            papers = self._search_for_intent(intent)
            self.state.papers = papers
            self.state.evidence_paper_ids = [paper.id for paper in papers if paper.id]
            self.store.save_papers(papers)
            yield {"type": "papers", "papers": self._paper_payloads(papers)}

            if not papers:
                yield {
                    "type": "answer",
                    "content": "没有在当前目标源中找到可用论文。可以放宽年份、取消指定会议/期刊，或换一个关键词再试。",
                }
                self._remember_conversation("assistant", "没有在当前目标源中找到可用论文。")
                yield {"type": "debug", "debug": self._finalize_debug(action=action, kimi_log_start=kimi_log_start, search_used=True)}
                return

        if run_document:
            evidence_count = len(self._evidence_papers())
            if run_pdf_read and not evidence_count:
                yield {"type": "status", "message": "正在根据上传 PDF 的证据生成写作草稿。"}
                document = self._generate_pdf_document(message)
            else:
                yield {"type": "status", "message": f"正在基于当前证据池生成文档和 BibTeX（{evidence_count} 篇论文）。"}
                document = self._generate_document(message)
            yield {"type": "document", **document}
            document_answer = self._answer_about_generated_outputs("请展示生成内容预览")
            self.state.last_answer = document_answer
            self._remember_conversation("assistant", document_answer)
            yield {"type": "answer", "content": document_answer}
            yield {"type": "debug", "debug": self._finalize_debug(action=action, kimi_log_start=kimi_log_start, search_used=should_search)}
            return

        if run_pdf_read and not should_search:
            yield {"type": "debug", "debug": self._finalize_debug(action=action, kimi_log_start=kimi_log_start, search_used=False)}
            return

        if self.state.uploaded_documents and (not self.state.papers or _looks_like_pdf_question(message)):
            yield {"type": "status", "message": "正在从上传 PDF 的相关页面中整理答案。"}
        else:
            yield {"type": "status", "message": "正在依据当前检索结果回答问题。"}
        answer = self._answer_from_papers(message)
        self.state.last_answer = answer
        self._remember_conversation("assistant", answer)
        yield {"type": "answer", "content": answer}
        yield {"type": "debug", "debug": self._finalize_debug(action=action, kimi_log_start=kimi_log_start, search_used=should_search)}

    def _handle_agent_stream(
        self,
        message: str,
        *,
        mode: str,
        session_id: str,
        evidence_paper_ids: list[str] | None,
    ) -> Iterator[dict]:
        """Execute one user request through a guarded plan-act-observe loop."""
        self.state = self._session_state(session_id)
        kimi_log_start = KimiClient.log_size()
        if evidence_paper_ids is not None:
            self.state.evidence_paper_ids = evidence_paper_ids
            if evidence_paper_ids and not self.state.papers:
                wanted = set(evidence_paper_ids)
                self.state.papers = [paper for paper in self.store.load_papers() if paper.id in wanted]
        message = message.strip()
        if not message:
            yield {"type": "error", "message": "请输入要调研或提问的内容。"}
            return

        self._remember_conversation("user", message)
        self.state.last_tool_plan = []
        observations: list[dict] = []
        last_result: ToolResult | None = None
        action = "answer"
        search_used = False
        tools = self._agent_tools(message)
        seen_tool_calls: set[str] = set()

        self.intent_analyzer.last_action_trace = {
            "requested": "agent",
            "used": "agent",
            "fallback": False,
            "error": "",
        }
        for step in range(1, self.agent_runtime.max_steps + 1):
            try:
                decision = self.agent_runtime.decide(
                    user_message=message,
                    mode=mode,
                    workspace_context=self._conversation_context(),
                    tools=tools,
                    observations=observations,
                    step=step,
                )
            except Exception as exc:
                if not last_result and not observations:
                    yield {"type": "status", "message": "Kimi 的 Agent 规划没有完成，正在切换到备用意图识别流程。"}
                    yield from self._fallback_to_legacy_stream_after_agent_failure(
                        message,
                        mode=mode,
                        session_id=session_id,
                        evidence_paper_ids=evidence_paper_ids,
                    )
                    return
                yield {"type": "status", "message": "Kimi 的 Agent 规划中断，正在保留已完成工具结果。"}
                answer = last_result.answer if last_result and last_result.answer else f"Agent 规划失败：{exc}"
                self.state.last_answer = answer
                self._remember_conversation("assistant", answer)
                yield {"type": "answer", "content": answer}
                yield {"type": "debug", "debug": self._finalize_debug(action=action, kimi_log_start=kimi_log_start, search_used=search_used)}
                return

            if decision.kind == "final":
                answer = decision.answer or (last_result.answer if last_result else "")
                if not answer:
                    answer = "已完成当前请求，但没有得到可展示的文本结果。"
                self.state.last_answer = answer
                self._remember_conversation("assistant", answer)
                yield {"type": "answer", "content": answer}
                yield {"type": "debug", "debug": self._finalize_debug(action=action, kimi_log_start=kimi_log_start, search_used=search_used)}
                return

            if decision.kind != "tool":
                observations.append({"tool": "planner", "ok": False, "summary": decision.reason, "answer": ""})
                yield {"type": "status", "message": "Kimi 的 Agent 决策不可执行，正在重试规划。"}
                continue

            tool = next((item for item in tools if item.name == decision.tool), None)
            if not tool:
                observations.append({"tool": decision.tool, "ok": False, "summary": "未注册工具。", "answer": ""})
                continue
            signature = _tool_call_signature(tool.name, decision.arguments)
            if signature in seen_tool_calls:
                yield {"type": "status", "message": f"检测到 Agent 重复调用 {tool.name}，已停止重复步骤并返回已有结果。"}
                answer = (last_result.answer if last_result else "") or (last_result.summary if last_result else "")
                if not answer:
                    answer = "已停止重复工具调用，请基于当前结果继续提问。"
                self.state.last_answer = answer
                self._remember_conversation("assistant", answer)
                yield {"type": "answer", "content": answer}
                yield {"type": "debug", "debug": self._finalize_debug(action=action, kimi_log_start=kimi_log_start, search_used=search_used)}
                return
            seen_tool_calls.add(signature)

            action = _agent_action_for_tool(tool.name)
            self.state.last_tool_plan.append(tool.name)
            yield {
                "type": "action",
                "action": {
                    "action": action,
                    "reason": decision.reason or f"Agent 选择调用 {tool.name}。",
                    "confidence": 1.0,
                    "source": "agent",
                    "tools": list(self.state.last_tool_plan),
                },
            }
            yield {"type": "status", "message": _agent_tool_status(tool.name)}
            if tool.name == "free_chat":
                chat_message = str(decision.arguments.get("message") or message).strip()
                yield {"type": "answer_start"}
                parts: list[str] = []
                try:
                    for delta in self._free_chat_stream(chat_message):
                        if delta:
                            text = str(delta)
                            parts.append(text)
                            yield {"type": "answer_delta", "delta": text}
                except HttpError as exc:
                    parts = [
                        "Kimi 这次没有及时返回，可能是网络或模型响应较慢。\n\n"
                        f"错误摘要：{exc}\n\n可以稍后重试，或缩短输入内容再试。"
                    ]
                    yield {"type": "answer_delta", "delta": parts[0]}
                answer = "".join(parts).strip() or "我在。刚刚模型没有返回可展示内容，你可以继续问我检索、阅读或写作相关的问题。"
                result = ToolResult("free_chat", "已完成自由聊天。", answer=answer, terminal=True)
            else:
                result = tool.execute(decision.arguments)
            last_result = result
            search_used = search_used or tool.name == "paper_search"
            observations.append(result.observation())
            for event in result.events:
                yield event

            if result.terminal:
                answer = result.answer or result.summary
                self.state.last_answer = answer
                self._remember_conversation("assistant", answer)
                yield {"type": "answer", "content": answer}
                yield {"type": "debug", "debug": self._finalize_debug(action=action, kimi_log_start=kimi_log_start, search_used=search_used)}
                return

        if not last_result:
            yield {"type": "status", "message": "Kimi 的 Agent 规划没有选择可执行工具，正在切换到备用意图识别流程。"}
            yield from self._fallback_to_legacy_stream_after_agent_failure(
                message,
                mode=mode,
                session_id=session_id,
                evidence_paper_ids=evidence_paper_ids,
            )
            return
        answer = last_result.answer if last_result and last_result.answer else "已完成可执行步骤，请根据当前证据继续提问。"
        self.state.last_answer = answer
        self._remember_conversation("assistant", answer)
        yield {"type": "answer", "content": answer}
        yield {"type": "debug", "debug": self._finalize_debug(action=action, kimi_log_start=kimi_log_start, search_used=search_used)}

    def _fallback_to_legacy_stream_after_agent_failure(
        self,
        message: str,
        *,
        mode: str,
        session_id: str,
        evidence_paper_ids: list[str] | None,
    ) -> Iterator[dict]:
        if self.state.conversation_history and self.state.conversation_history[-1].get("role") == "user":
            self.state.conversation_history.pop()
        self.state.last_tool_plan = []
        yield from self._handle_legacy_stream(
            message,
            mode=mode,
            session_id=session_id,
            evidence_paper_ids=evidence_paper_ids,
        )

    def _agent_tools(self, message: str) -> list[AgentTool]:
        """Expose existing engine capabilities as executable, validated Agent tools."""

        def paper_search(arguments: dict) -> ToolResult:
            try:
                intent = self.intent_analyzer.research_from_plan(message, arguments, source="agent")
            except ValueError as exc:
                return ToolResult(
                    name="paper_search",
                    ok=False,
                    summary=str(exc),
                    answer="请根据工具契约重新给出规范研究主题，而不是复制完整用户请求。",
                    terminal=False,
                )
            self.state.last_intent = intent
            self._remember_research_topic(intent.normalized_topic, intent.display_topic)
            papers = self._search_for_intent(intent)
            self.state.papers = papers
            self.state.evidence_paper_ids = [paper.id for paper in papers if paper.id]
            self.store.save_papers(papers)
            titles = "; ".join(paper.title[:130] for paper in papers[:8])
            source_summary = _search_source_summary(self._last_search_debug.get("sources", {}))
            summary = (
                f"检索主题：{intent.normalized_topic}。找到 {len(papers)} 篇论文。"
                + (f" 代表结果：{titles}" if titles else "")
            )
            answer = _search_answer_summary(intent, papers, source_summary)
            events = [
                {"type": "intent", "intent": self._intent_payload(intent)},
                {"type": "papers", "papers": self._paper_payloads(papers)},
            ]
            if source_summary:
                events.append({"type": "status", "message": f"来源执行结果：{source_summary}"})
            if not papers:
                return ToolResult(
                    name="paper_search",
                    ok=False,
                    summary=summary,
                    events=events,
                    answer="没有在当前约束下找到可用论文。可以放宽年份、会议/期刊限制，或换一个更短的英文关键词重试。",
                    terminal=True,
                )
            return ToolResult(
                name="paper_search",
                summary=summary,
                events=events,
                answer=answer,
                # The model decides whether to write after observing the search.
                terminal=False,
            )

        def pdf_read(arguments: dict) -> ToolResult:
            if not self.state.uploaded_documents:
                return ToolResult("pdf_read", "当前会话没有上传 PDF。", ok=False)
            question = str(arguments.get("question") or message).strip()
            answer = self._answer_from_uploaded_pdfs(question)
            return ToolResult(
                name="pdf_read",
                summary="已从上传 PDF 提取并分析与请求相关的证据。",
                answer=answer,
                # Keep the observation available for a model-planned writing step.
                terminal=False,
            )

        def evidence_answer(arguments: dict) -> ToolResult:
            if not self._evidence_papers() and not self.state.uploaded_documents:
                return ToolResult("evidence_answer", "当前没有论文或 PDF 证据可供问答。", ok=False)
            question = str(arguments.get("question") or message).strip()
            answer = self._answer_from_papers(question)
            return ToolResult(
                name="evidence_answer",
                summary="已基于当前证据池生成答案。",
                answer=answer,
                terminal=True,
            )

        def paper_fulltext_read(arguments: dict) -> ToolResult:
            paper = self._resolve_fulltext_paper(arguments)
            if not paper:
                return ToolResult("paper_fulltext_read", "没有找到用户指定的当前论文。", ok=False)
            question = str(arguments.get("question") or message).strip()
            answer = self._answer_from_paper_fulltext(paper, question)
            self.store.save_papers([paper])
            events = [{"type": "papers", "papers": self._paper_payloads(self.state.papers)}]
            return ToolResult(
                name="paper_fulltext_read",
                summary=f"已尝试读取 {paper.title} 的本地全文缓存。",
                events=events,
                answer=answer,
                terminal=True,
            )

        def write_document(arguments: dict) -> ToolResult:
            # The Agent may select a writing *kind*, but it must not replace
            # the user's actual request with a fabricated prompt such as
            # "generate an outline".  The original message remains the sole
            # source of the writing goal.
            request = message
            writing_kind = str(arguments.get("deliverable") or "").strip().lower()
            admission = self._confirm_writing_request(request)
            if not admission["should_write"]:
                answer = self._search_only_completion_answer(admission["reason"])
                return ToolResult(
                    "write_document",
                    "写作准入未通过：用户仅要求检索或调研，不应擅自生成文稿。",
                    answer=answer,
                    terminal=True,
                )
            if self.state.uploaded_documents and not self._evidence_papers():
                document = self._generate_pdf_document(request)
            elif self._evidence_papers():
                document = (
                    self._generate_document(request, writing_kind=writing_kind)
                    if writing_kind
                    else self._generate_document(request)
                )
            else:
                return ToolResult("write_document", "没有可用于写作的论文或 PDF 证据。", ok=False)
            answer = self._answer_about_generated_outputs("请展示生成内容预览")
            return ToolResult(
                name="write_document",
                summary=f"已生成文档：{(document.get('markdown') or {}).get('name') or '草稿'}。",
                events=[{"type": "document", **document}],
                answer=answer,
                terminal=True,
            )

        def document_inspect(arguments: dict) -> ToolResult:
            if not self.state.last_generated_document:
                return ToolResult("document_inspect", "当前会话还没有生成文档。", ok=False)
            question = str(arguments.get("question") or message).strip()
            answer = self._inspect_generated_document(question)
            return ToolResult("document_inspect", "已检查上一份生成文档的实际内容。", answer=answer, terminal=True)

        def free_chat(arguments: dict) -> ToolResult:
            chat_message = str(arguments.get("message") or message).strip()
            answer = self._free_chat(chat_message)
            return ToolResult("free_chat", "已完成自由聊天。", answer=answer, terminal=True)

        handlers = {
            "paper_search": paper_search,
            "pdf_read": pdf_read,
            "evidence_answer": evidence_answer,
            "paper_fulltext_read": paper_fulltext_read,
            "write_document": write_document,
            "document_inspect": document_inspect,
            "free_chat": free_chat,
        }
        context = ToolContext(
            has_papers=bool(self._evidence_papers()),
            has_documents=bool(self.state.uploaded_documents),
            has_generated_document=bool(self.state.last_generated_document),
        )
        available = self.tool_registry.validate_plan(list(handlers), context)
        return [
            AgentTool(spec.name, spec.description, spec.parameters, handlers[spec.name], spec.requires)
            for name in available
            if (spec := self.tool_registry.get(name)) is not None
        ]

    def snapshot(self, *, session_id: str = "default") -> dict:
        # Snapshot is read-only. Do not overwrite the shared working state,
        # otherwise refreshing the UI can contend with a long-running stream.
        state = self._session_state(session_id)
        return {
            "project": self.store.get_project(_clean_session_id(session_id)),
            "papers": self._paper_payloads(state.papers),
            "intent": self._intent_payload(state.last_intent) if state.last_intent else None,
            "generated_files": state.generated_files,
            "last_generated_document": state.last_generated_document,
            "evidence_paper_ids": state.evidence_paper_ids,
            "conversation_turns": len(state.conversation_history),
            # Browser storage is only a cache. Return persisted history so a
            # refresh or service restart can rebuild the conversation.
            "messages": state.conversation_history,
            "research_topics": state.research_topics,
            "uploaded_documents": [document.to_payload() for document in state.uploaded_documents],
            "debug": state.last_debug,
            "store": self.store.stats(),
            "llm": {
                "kimi_available": self.intent_analyzer.llm.available,
                "kimi_model": self.intent_analyzer.llm.model,
                "intent_source": state.last_intent.source if state.last_intent else None,
            },
            "agent": {
                "tools": self.tool_registry.payloads(),
                "skills": [
                    skill.payload()
                    for skill in self.orchestrator.skill_catalog.for_plan(
                        state.last_tool_plan,
                        category=getattr(state.last_intent, "category", ""),
                        deliverable=(
                            getattr(state.last_intent, "deliverable", "")
                            or str((state.last_generated_document or {}).get("writing_kind") or "")
                        ),
                    )
                ],
            },
        }

    def list_projects(self) -> list[dict]:
        return self.store.list_projects()

    def create_project(self, title: str = "新研究项目") -> dict:
        project = self.store.create_project(title, state=_assistant_state_to_dict(AssistantState()))
        self.sessions[str(project["id"])] = AssistantState()
        return project

    def rename_project(self, project_id: str, title: str) -> dict | None:
        state = self._session_state(project_id)
        return self.store.update_project(_clean_session_id(project_id), title=title, state=_assistant_state_to_dict(state))

    def list_drafts(self, *, session_id: str = "default") -> list[dict]:
        return self.store.list_drafts(_clean_session_id(session_id))

    def get_draft(self, draft_id: str, *, session_id: str = "default") -> dict | None:
        return self.store.get_draft(_clean_session_id(session_id), draft_id)

    def update_draft(self, draft_id: str, payload: dict, *, session_id: str = "default", note: str = "手动编辑") -> dict | None:
        return self.store.update_draft(_clean_session_id(session_id), draft_id, payload, note=note)

    def draft_versions(self, draft_id: str, *, session_id: str = "default") -> list[dict]:
        return self.store.draft_versions(_clean_session_id(session_id), draft_id)

    def revise_draft(
        self,
        draft_id: str,
        *,
        selected_text: str,
        instruction: str,
        session_id: str = "default",
    ) -> dict:
        """Rewrite one selected passage while retaining the rest of the draft verbatim."""
        draft = self.get_draft(draft_id, session_id=session_id)
        selected = str(selected_text or "")
        request = normalize_text(instruction or "")
        if not draft:
            return {"ok": False, "error": "未找到草稿。"}
        if not selected or selected not in str(draft.get("content_markdown") or ""):
            return {"ok": False, "error": "请先在编辑器中选中需要修改的原文。"}
        if not request:
            return {"ok": False, "error": "请填写对选中段落的修改要求。"}
        if not self.llm.available:
            return {"ok": False, "error": "局部改写需要配置 Kimi API Key。"}
        citation_keys = ", ".join(_bibtex_keys(str(draft.get("bibtex") or ""))) or "无"
        system = (
            "你是严谨的学术论文编辑。只改写用户提供的选中段落，不输出标题、解释、Markdown 代码块或任何额外文字。"
            "不得捏造事实、论文、实验数值或引用。可保留原有 \\cite{key}；只能使用允许的引用键。"
            "输出必须是一段可以直接替换原文的完整中文或英文论文段落。"
        )
        user = (
            f"修改要求：{request}\n\n"
            f"允许引用键：{citation_keys}\n\n"
            f"选中原文：\n{selected}"
        )
        try:
            revised = self.llm.chat_text(
                system=system,
                user=user,
                max_tokens=max(600, min(2400, len(selected) * 3)),
                label="draft_local_revision",
            ).strip()
        except Exception as exc:
            return {"ok": False, "error": f"局部改写失败：{exc}"}
        if not revised:
            return {"ok": False, "error": "模型没有返回可用的局部改写内容。"}
        content = str(draft.get("content_markdown") or "").replace(selected, revised, 1)
        updated = self.update_draft(
            draft_id,
            {"content_markdown": content},
            session_id=session_id,
            note=f"局部改写：{request[:80]}",
        )
        return {"ok": bool(updated), "draft": updated, "revised_text": revised}

    def restore_draft_version(self, draft_id: str, *, version: int, session_id: str = "default") -> dict:
        """Restore an historical draft as a new version, preserving full history."""
        draft = self.get_draft(draft_id, session_id=session_id)
        if not draft:
            return {"ok": False, "error": "未找到草稿。"}
        historical = next(
            (item for item in self.draft_versions(draft_id, session_id=session_id) if int(item.get("version") or 0) == int(version)),
            None,
        )
        if not historical:
            return {"ok": False, "error": f"未找到版本 {version}。"}
        updated = self.update_draft(
            draft_id,
            {
                "content_markdown": historical.get("content_markdown") or "",
                "bibtex": historical.get("bibtex") or "",
            },
            session_id=session_id,
            note=f"恢复版本 {version}",
        )
        # Restoring the already-current version should still be a successful no-op.
        return {"ok": bool(updated), "draft": updated}

    def systematic_review(self, *, session_id: str = "default") -> dict:
        project_id = _clean_session_id(session_id)
        state = self._session_state(project_id)
        project_paper_ids, _ = self.store.project_paper_ids(project_id)
        wanted = set(project_paper_ids or [paper.id for paper in state.papers if paper.id])
        papers = [paper for paper in self.store.load_papers() if paper.id in wanted]
        if not papers:
            papers = list(state.papers)
        snapshot = review_snapshot(
            papers=papers,
            protocol=self.store.get_review_protocol(project_id),
            screenings=self.store.list_review_screenings(project_id),
        )
        return {"ok": True, **snapshot}

    def update_systematic_review_protocol(self, payload: dict, *, session_id: str = "default") -> dict:
        project_id = _clean_session_id(session_id)
        self._session_state(project_id)
        protocol = self.store.upsert_review_protocol(project_id, payload)
        return {"ok": True, "protocol": protocol}

    def screen_review_paper(
        self,
        paper_id: str,
        *,
        stage: str,
        decision: str,
        reason: str = "",
        session_id: str = "default",
    ) -> dict:
        project_id = _clean_session_id(session_id)
        state = self._session_state(project_id)
        known_ids = {paper.id for paper in state.papers if paper.id}
        known_ids.update(self.store.project_paper_ids(project_id)[0])
        if paper_id not in known_ids:
            return {"ok": False, "error": "该论文不在当前项目的文献池中。"}
        try:
            screening = self.store.set_review_screening(
                project_id, paper_id, stage=stage, decision=decision, reason=reason
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "screening": screening}

    def export_systematic_review(self, *, session_id: str = "default", format: str = "evidence_csv") -> dict:
        snapshot = self.systematic_review(session_id=session_id)
        normalized = str(format or "evidence_csv").strip().lower().replace("-", "_")
        suffixes = {"evidence_csv": ".csv", "csv": ".csv", "prisma": ".md", "prisma_markdown": ".md", "markdown": ".md"}
        if normalized not in suffixes:
            return {"ok": False, "error": f"不支持的系统综述导出格式：{format}"}
        name = f"systematic_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffixes[normalized]}"
        try:
            artifact = write_review_export(snapshot=snapshot, format=normalized, path=self.output_dir / name)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "file": {"name": artifact["name"], "url": f"/outputs/{artifact['name']}", "kind": artifact["kind"]}}

    def export_draft(self, draft_id: str, *, session_id: str = "default", format: str = "markdown") -> dict:
        draft = self.get_draft(draft_id, session_id=session_id)
        if not draft:
            return {"ok": False, "error": "未找到草稿。"}
        normalized_format = str(format or "markdown").strip().lower().replace("-", "_")
        formats = {
            "markdown": (".md", "markdown"),
            "bibtex": (".bib", "bibtex"),
            "latex": (".tex", "latex"),
            "tex": (".tex", "latex"),
            "docx": (".docx", "docx"),
            "ris": (".ris", "ris"),
            "csl": (".json", "csl_json"),
            "csl_json": (".json", "csl_json"),
            "csljson": (".json", "csl_json"),
        }
        if normalized_format not in formats:
            return {"ok": False, "error": f"不支持的导出格式：{format}"}
        safe_title = re.sub(r"[^A-Za-z0-9._-]+", "_", draft["title"]).strip("_") or "draft"
        suffix, kind = formats[normalized_format]
        path = self.output_dir / f"{safe_title}_v{draft['version']}{suffix}"
        paper_ids = {str(paper_id) for paper_id in draft.get("paper_ids") or []}
        papers = [paper for paper in self.store.load_papers() if paper.id in paper_ids]
        try:
            export_draft_artifact(draft=draft, papers=papers, format=normalized_format, path=path)
        except DraftExportError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "file": {"name": path.name, "url": f"/outputs/{path.name}", "kind": kind}}

    def update_paper_asset_state(self, paper_id: str, updates: dict, *, session_id: str = "default") -> dict:
        self.state = self._session_state(session_id)
        updated = self.store.update_paper_asset_state(
            paper_id,
            reading_status=updates.get("reading_status"),
            importance=updates.get("importance"),
            user_tags=updates.get("user_tags") if isinstance(updates.get("user_tags"), list) else None,
            excluded=updates.get("excluded") if isinstance(updates.get("excluded"), bool) else None,
            exclusion_reason=updates.get("exclusion_reason"),
            user_notes=updates.get("user_notes"),
            used_in_sections=updates.get("used_in_sections") if isinstance(updates.get("used_in_sections"), list) else None,
            relevance_score=updates.get("relevance_score") if updates.get("relevance_score") is not None else None,
        )
        if not updated:
            return {"ok": False, "error": "未找到对应论文。"}
        for index, paper in enumerate(self.state.papers):
            if paper.id == updated.id:
                self.state.papers[index] = updated
        return {"ok": True, "paper": self._paper_payloads([updated])[0]}

    def download_paper_pdf(self, paper_id: str, *, session_id: str = "default") -> dict:
        """Download one paper's PDF into the local personal library."""
        self.state = self._session_state(session_id)
        paper = self._paper_by_id(paper_id)
        if not paper:
            return {"ok": False, "error": "未找到对应论文。"}
        result = self.fulltext_manager.cache_pdf(paper, extract_text=True)
        self.store.save_papers([paper])
        self._upsert_session_paper(paper)
        self._persist_sessions()
        ok = bool(paper.local_pdf_path and Path(paper.local_pdf_path).exists())
        message = result.summary
        if not ok:
            message = (
                f"{result.summary}\n\n"
                "这篇论文暂时无法自动下载。请在文献库中点击“上传原文”，选择你本地的 PDF；"
                "上传后系统会保存到个人文献库并在写作时优先使用全文。"
            )
        return {
            "ok": ok,
            "message": message,
            "paper": self._paper_payloads([paper])[0],
        }

    def find_open_pdf(self, paper_id: str, *, session_id: str = "default") -> dict:
        """Use Scholar's explicit PDF resource, if available, to enrich one library record."""
        self.state = self._session_state(session_id)
        paper = self._paper_by_id(paper_id)
        if not paper:
            return {"ok": False, "error": "未找到对应论文。"}
        if paper.local_pdf_path and Path(paper.local_pdf_path).exists():
            return {"ok": True, "message": "这篇论文已在个人文献库中。", "paper": self._paper_payloads([paper])[0]}
        if paper.pdf_url:
            return {"ok": True, "message": "该论文已经记录了开放 PDF 链接。", "paper": self._paper_payloads([paper])[0]}

        try:
            matched = self.search_service.google_scholar.lookup_by_title(paper.title, year=paper.year)
        except Exception as exc:
            return {"ok": False, "error": f"Google Scholar 查询失败：{exc}"}
        if not matched or not matched.pdf_url:
            return {
                "ok": False,
                "error": "Google Scholar 未返回可直接下载的公开 PDF 链接。可打开论文来源页确认，或手动上传你已获取的原文。",
                "paper": self._paper_payloads([paper])[0],
            }

        paper.pdf_url = matched.pdf_url
        self.store.save_papers([paper])
        self._upsert_session_paper(paper)
        self._persist_sessions()
        return {
            "ok": True,
            "message": "已从 Google Scholar 的公开 PDF 资源中找到下载链接。",
            "paper": self._paper_payloads([paper])[0],
        }

    def open_local_library_folder(self) -> dict:
        """Open the managed personal-library PDF folder in the local file manager."""
        folder = self.fulltext_manager.pdf_dir.resolve()
        if sys.platform == "darwin":
            command = ["open", str(folder)]
        elif sys.platform.startswith("win"):
            command = ["explorer", str(folder)]
        else:
            command = ["xdg-open", str(folder)]

        try:
            subprocess.run(command, check=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "ok": False,
                "folder": str(folder),
                "error": f"无法打开个人文献库文件夹：{exc}",
            }
        return {
            "ok": True,
            "folder": str(folder),
            "message": "已打开个人文献库文件夹。",
        }

    def cached_paper_pdf_path(self, paper_id: str, *, session_id: str = "default") -> Path | None:
        self.state = self._session_state(session_id)
        paper = self._paper_by_id(paper_id)
        if not paper or not paper.local_pdf_path:
            return None
        path = Path(paper.local_pdf_path)
        if not path.exists() or not path.is_file():
            return None
        if _is_path_inside(path, self.fulltext_manager.pdf_dir) or _is_path_inside(path, self.upload_dir):
            return path
        return None

    def reset_session(self, session_id: str = "default") -> None:
        clean_id = _clean_session_id(session_id)
        previous = self.sessions.get(clean_id)
        if previous:
            for document in previous.uploaded_documents:
                try:
                    path = Path(document.path)
                    if _is_path_inside(path, self.fulltext_manager.pdf_dir):
                        continue
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        self.sessions[clean_id] = AssistantState()
        self.store.clear_project_workspace(clean_id, state=_assistant_state_to_dict(self.sessions[clean_id]))
        if clean_id == "default":
            self.state = self.sessions[clean_id]
        self._persist_sessions()

    def _session_state(self, session_id: str) -> AssistantState:
        clean_id = _clean_session_id(session_id)
        if clean_id not in self.sessions:
            self.sessions[clean_id] = AssistantState()
        if not self.store.get_project(clean_id):
            self.store.create_project(
                "新研究项目" if clean_id == "default" else "研究项目",
                project_id=clean_id,
                state=_assistant_state_to_dict(self.sessions[clean_id]),
            )
        return self.sessions[clean_id]

    def _load_sessions(self) -> None:
        # SQLite is the primary source of truth. Import sessions.json only for
        # existing installations that predate durable project storage.
        projects = self.store.list_projects()
        if projects:
            restored: dict[str, AssistantState] = {}
            for project in projects:
                project_id = str(project["id"])
                state = _assistant_state_from_dict(project.get("state") or {})
                paper_ids, selected_ids = self.store.project_paper_ids(project_id)
                if paper_ids:
                    wanted = set(paper_ids)
                    state.papers = [paper for paper in self.store.load_papers() if paper.id in wanted]
                    state.evidence_paper_ids = selected_ids
                documents = self.store.project_documents(project_id)
                if documents:
                    state.uploaded_documents = [_uploaded_document_from_dict(item) for item in documents]
                messages = self.store.project_messages(project_id)
                if messages:
                    state.conversation_history = [
                        {"role": item["role"], "content": item["content"]} for item in messages
                    ][-28:]
                restored[project_id] = state
            self.sessions = restored
            self.state = self.sessions.get("default") or next(iter(self.sessions.values()))
            return
        if not self.session_store_path.exists():
            self.store.create_project("新研究项目", project_id="default", state=_assistant_state_to_dict(self.state))
            return
        try:
            data = json.loads(self.session_store_path.read_text(encoding="utf-8"))
            sessions = data.get("sessions") if isinstance(data, dict) else {}
            if not isinstance(sessions, dict):
                return
            restored = {
                _clean_session_id(session_id): _assistant_state_from_dict(payload)
                for session_id, payload in sessions.items()
                if isinstance(payload, dict)
            }
            if restored:
                self.sessions = restored
                self.state = self.sessions.get("default") or next(iter(self.sessions.values()))
                self._persist_sessions()
        except Exception:
            self.sessions = {"default": self.state}

    def _persist_sessions(self) -> None:
        for session_id, state in self.sessions.items():
            project = self.store.get_project(session_id)
            title = project.get("title") if project else ("新研究项目" if session_id == "default" else "研究项目")
            if project:
                self.store.update_project(session_id, title=title, state=_assistant_state_to_dict(state))
            else:
                self.store.create_project(title, project_id=session_id, state=_assistant_state_to_dict(state))
            self.store.replace_project_papers(
                session_id,
                [paper.id for paper in state.papers if paper.id],
                selected_ids=state.evidence_paper_ids,
            )
            self.store.replace_project_documents(
                session_id,
                [_uploaded_document_to_dict(document) for document in state.uploaded_documents],
            )
            # Migrate the old JSON history exactly once. New turns are inserted
            # incrementally in _remember_conversation.
            if not self.store.project_messages(session_id, limit=1):
                for message in state.conversation_history:
                    self.store.append_project_message(
                        session_id, str(message.get("role") or "assistant"), str(message.get("content") or "")
                    )
        payload = {
            "version": 1,
            "sessions": {
                session_id: _assistant_state_to_dict(state)
                for session_id, state in self.sessions.items()
            },
        }
        tmp_path = self.session_store_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.session_store_path)

    def _search_for_intent(self, intent: ResearchIntent, *, limit: int = 24) -> list[Paper]:
        recent_years = intent.recent_years
        from_year = intent.from_year
        to_year = intent.to_year
        if recent_years is None and from_year is None:
            recent_years = 3

        cached_papers = self.store.search_cached(
            intent.normalized_topic,
            from_year=from_year,
            to_year=to_year,
            limit=80,
        )
        all_papers: list[Paper] = list(cached_papers)
        # Start with one canonical query per source. Only expand once if the
        # first pass cannot form a usable evidence pool.
        source_plan = _source_query_plan(intent, max_queries=1)
        candidate_limit = max(limit * 3, 90) if intent.target_venue_ranks else max(limit * 2, 50)
        fetched_counts: dict[str, int] = {source: 0 for source in source_plan}
        source_reports: dict[str, dict] = {
            source: {"queries": 0, "successful_queries": 0, "empty_queries": 0, "failures": []}
            for source in source_plan
        }
        failed_sources: set[str] = set()
        rank_search_venues = []
        if intent.target_venue_ranks and not intent.target_venues:
            rank_search_venues = self.venue_policy.venues_for_ranks(intent.target_venue_ranks, intent.cs_area)
        executed_jobs: set[tuple[str, str, tuple[str, ...]]] = set()

        def fetch_one(source: str, query: str, target_venues: list[str] | None):
            return self.search_service.search(
                query,
                limit=candidate_limit,
                sources=[source],
                recent_years=recent_years,
                from_year=from_year,
                to_year=to_year,
                target_venues=target_venues,
            )

        def build_jobs(plan: dict[str, list[str]], *, relax_rank_venues: bool = False) -> list[tuple[str, str, list[str] | None]]:
            jobs: list[tuple[str, str, list[str] | None]] = []
            for source, queries in plan.items():
                if source in failed_sources:
                    continue
                for query in queries:
                    query = normalize_text(query)
                    if not query:
                        continue
                    search_target_venues = intent.target_venues
                    if source in {"dblp", "google_scholar"} and rank_search_venues and not relax_rank_venues:
                        search_target_venues = rank_search_venues
                    job_key = (source, query.lower(), tuple(sorted(search_target_venues or [])))
                    if job_key in executed_jobs:
                        continue
                    executed_jobs.add(job_key)
                    jobs.append((source, query, search_target_venues))
            return jobs

        def run_jobs(jobs: list[tuple[str, str, list[str] | None]]) -> None:
            if not jobs:
                return
            grouped_jobs: dict[str, list[tuple[str, str, list[str] | None]]] = {}
            for job in jobs:
                grouped_jobs.setdefault(job[0], []).append(job)

            def run_source_jobs(source_jobs: list[tuple[str, str, list[str] | None]]) -> dict:
                source = source_jobs[0][0]
                report = {"queries": 0, "successful_queries": 0, "empty_queries": 0, "failures": []}
                papers: list[Paper] = []
                fetched_count = 0
                disable_source = False
                for source, query, target_venues in source_jobs:
                    report["queries"] += 1
                    try:
                        result = fetch_one(source, query, target_venues)
                        papers.extend(result.papers)
                        fetched_count += len(result.papers)
                        if result.papers:
                            report["successful_queries"] += 1
                        else:
                            report["empty_queries"] += 1
                        for status in result.source_status.values():
                            if status.get("status") in {"error", "unavailable"}:
                                error = str(status.get("error") or "来源请求失败")
                                if error not in report["failures"]:
                                    report["failures"].append(error[:500])
                    except Exception as exc:
                        error = str(exc)[:500]
                        report["failures"].append(error)
                        if _is_source_rate_or_permission_error(error):
                            disable_source = True
                            break
                return {
                    "source": source,
                    "papers": papers,
                    "fetched_count": fetched_count,
                    "report": report,
                    "disable_source": disable_source,
                }

            executor = ThreadPoolExecutor(max_workers=min(3, max(1, len(grouped_jobs))))
            try:
                future_map = {
                    executor.submit(run_source_jobs, source_jobs): source
                    for source, source_jobs in grouped_jobs.items()
                }
                done, pending = wait(future_map, timeout=SEARCH_SOURCE_BATCH_TIMEOUT)
                for future in done:
                    source = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        source_reports[source]["failures"].append(str(exc)[:500])
                        continue
                    all_papers.extend(result["papers"])
                    fetched_counts[source] = fetched_counts.get(source, 0) + int(result["fetched_count"])
                    report = source_reports[source]
                    source_report = result["report"]
                    for key in ("queries", "successful_queries", "empty_queries"):
                        report[key] += int(source_report[key])
                    for error in source_report["failures"]:
                        if error not in report["failures"]:
                            report["failures"].append(error)
                    if result["disable_source"]:
                        failed_sources.add(source)
                for future in pending:
                    source = future_map[future]
                    future.cancel()
                    failed_sources.add(source)
                    source_reports[source]["failures"].append(
                        f"来源响应超过 {SEARCH_SOURCE_BATCH_TIMEOUT} 秒，已跳过本轮查询。"
                    )
            finally:
                # Do not wait for an unresponsive third-party request. It has
                # no shared state to commit, so late completion is harmless.
                executor.shutdown(wait=False, cancel_futures=True)

        def filtered_candidates() -> list[Paper]:
            return self.venue_policy.filter(
                dedupe_papers(all_papers),
                target_venues=intent.target_venues,
                target_venue_ranks=intent.target_venue_ranks,
            )

        run_jobs(build_jobs(source_plan))
        target_papers = filtered_candidates()
        supplemental_plan = {}
        if len(target_papers) < _low_recall_threshold(limit):
            supplemental_plan = _source_query_plan(intent, max_queries=2)
            run_jobs(build_jobs(supplemental_plan, relax_rank_venues=bool(intent.target_venue_ranks)))

        deduped = dedupe_papers(all_papers)
        self.store.save_papers(deduped, topic=intent.normalized_topic)
        target_papers = self.venue_policy.filter(
            deduped,
            target_venues=intent.target_venues,
            target_venue_ranks=intent.target_venue_ranks,
        )
        # Metadata enrichment improves display quality but must remain a small
        # best-effort step after the primary evidence pool is ready.
        enrichment_debug = _enrich_missing_metadata(self.search_service, target_papers, max_items=1)
        self.store.save_papers(target_papers, topic=intent.normalized_topic)
        topic_constraints = _topic_constraint_groups(intent)
        constrained_papers = [
            paper for paper in target_papers if _passes_topic_constraints(paper, topic_constraints)
        ]
        ranking_candidates = constrained_papers if topic_constraints else target_papers
        ranked_pool = rank_papers(ranking_candidates, intent.normalized_topic, limit=max(limit * 2, 48))
        ranked = self._filter_papers_with_llm(intent, ranked_pool, limit=limit)
        self._last_search_debug = {
            "cached_hits": len(cached_papers),
            "external_hits": fetched_counts,
            "sources": source_reports,
            "deduped_count": len(deduped),
            "filtered_count": len(target_papers),
            "ranked_count": len(ranked),
            "used_rank_filter": bool(intent.target_venue_ranks),
            "query_plan": source_plan,
            "supplemental_search": bool(supplemental_plan),
            "metadata_enrichment": enrichment_debug,
            "topic_constraints": {
                "groups": topic_constraints,
                "input_count": len(target_papers),
                "kept_count": len(constrained_papers) if topic_constraints else len(target_papers),
                "used": bool(topic_constraints),
            },
            "llm_relevance": self._last_search_debug.get("llm_relevance", {}),
        }
        return ranked

    def _filter_papers_with_llm(self, intent: ResearchIntent, papers: list[Paper], *, limit: int) -> list[Paper]:
        """Ask Kimi to remove obvious topic mismatches after high-recall search."""
        self._last_search_debug["llm_relevance"] = {
            "requested": bool(self.llm.available),
            "used": False,
            "input_count": len(papers),
            "kept_count": min(len(papers), limit),
            "rejected_count": 0,
            "error": "",
        }
        if not papers or not self.llm.available:
            return papers[:limit]
        candidates = papers[: min(len(papers), 48)]
        payload = [
            {
                "id": paper.id,
                "title": paper.title,
                "abstract": (paper.abstract or "")[:900],
                "venue": paper.venue,
                "year": paper.year,
                "source": paper.source,
            }
            for paper in candidates
        ]
        system = (
            "You are a strict but recall-friendly paper relevance reviewer for a CS literature search. "
            "Return only valid JSON. Do not invent paper IDs. "
            "Judge whether each candidate matches the requested research topic. "
            "For compound topics, every essential modifier must be satisfied. "
            "For example, 'debiasing dynamic recommender systems' requires debiasing/fairness/unbiased/bias-mitigation, "
            "dynamic/temporal/sequential/time-aware behavior, and recommender/recommendation context. "
            "Reject papers that only match a parent field such as dynamic recommender systems but miss debiasing/fairness. "
            "Keep metadata-limited papers only when the title itself satisfies the essential modifiers."
        )
        user = json.dumps(
            {
                "research_topic": intent.normalized_topic,
                "keywords": intent.keywords,
                "original_request": intent.original_request,
                "candidates": payload,
                "response_schema": {
                    "relevant_ids": ["paper ids that clearly match"],
                    "borderline_ids": ["paper ids that may match; keep if metadata is limited"],
                    "rejected_ids": ["paper ids that are off-topic"],
                    "reason": "short Chinese summary",
                },
            },
            ensure_ascii=False,
        )
        try:
            data = self.llm.chat_json(
                system=system,
                user=user,
                temperature=0.1,
                max_tokens=2200,
                timeout=12,
                stream=False,
                label="search_relevance_check",
            )
            relevant_ids = _clean_id_list(data.get("relevant_ids"))
            borderline_ids = _clean_id_list(data.get("borderline_ids"))
            keep_ids = set([*relevant_ids, *borderline_ids])
            if not keep_ids:
                raise ValueError("Kimi relevance check returned no keepable paper IDs.")
            kept = [paper for paper in papers if paper.id in keep_ids]
            rejected_count = len([paper for paper in candidates if paper.id not in keep_ids])
            self._last_search_debug["llm_relevance"] = {
                "requested": True,
                "used": True,
                "input_count": len(candidates),
                "kept_count": len(kept),
                "rejected_count": rejected_count,
                "reason": normalize_text(str(data.get("reason") or "")),
                "error": "",
            }
            return (kept or papers)[:limit]
        except Exception as exc:
            self._last_search_debug["llm_relevance"] = {
                "requested": True,
                "used": False,
                "input_count": len(candidates),
                "kept_count": min(len(papers), limit),
                "rejected_count": 0,
                "error": str(exc)[:500],
            }
            return papers[:limit]

    def _answer_from_papers(self, question: str) -> str:
        if _looks_like_generated_output_question(question):
            return self._answer_about_generated_outputs(question)

        if self.state.uploaded_documents and (not self.state.papers or _looks_like_pdf_question(question)):
            return self._answer_from_uploaded_pdfs(question)

        if not self.state.papers:
            return "当前还没有检索结果。请先告诉我你想调研的方向，例如“近五年 ACL 上 RAG 相关论文”。"

        if self.llm.available:
            try:
                return self._answer_with_kimi(question)
            except Exception:
                pass

        papers = self.state.papers[:8]
        lines = [
            "基于当前检索结果，可以先这样回答：",
            "",
            f"- 当前论文池共有 {len(self.state.papers)} 篇，主要来源包括：{self._source_summary(self.state.papers)}。",
            "- 下面这些论文最值得优先看：",
        ]
        for index, paper in enumerate(papers, start=1):
            lines.append(
                f"{index}. {paper.title}（{paper.year or 'n.d.'}，{paper.venue or paper.source}）："
                f"{self._paper_hint(paper)}"
            )
        lines.extend(
            [
                "",
                "注意：这只是基于标题、摘要和元数据的回答。涉及实验结论、优劣比较和具体贡献时，需要进一步拉取全文核查。",
            ]
        )
        return "\n".join(lines)

    def _answer_from_paper_fulltext(self, paper: Paper, question: str) -> str:
        result = self.fulltext_manager.ensure_text(paper)
        if not result.chunks:
            return (
                f"还不能读取《{paper.title}》的全文。{result.summary}\n\n"
                "当前仍可基于标题、摘要和元数据回答；如果需要全文级细节，请选择有开放 PDF 链接的论文，或手动上传该论文 PDF。"
            )
        selected = relevant_chunks(question, result.chunks, limit=8, max_chars=18_000)
        if not selected:
            selected = result.chunks[:6]
        excerpts = "\n\n".join(f"[{paper.title}，第 {chunk.page} 页]\n{chunk.text}" for chunk in selected)
        if self.llm.available:
            system = (
                "You answer detailed questions about one computer-science paper using only the provided local full-text excerpts. "
                "Do not invent claims, metrics, datasets, or results. Cite page evidence in Chinese as 【论文标题，第 N 页】."
            )
            user = f"用户问题：{question}\n\n本地全文缓存状态：{result.summary}\n\n论文片段：\n{excerpts}"
            try:
                answer = self.llm.chat_text(
                    system=system,
                    user=user,
                    temperature=0.2,
                    max_tokens=1800,
                    timeout=120,
                    stream=False,
                    label="paper_fulltext_read",
                ).strip()
                if answer:
                    return answer
            except Exception:
                pass
        lines = [f"已读取《{paper.title}》的本地全文片段。{result.summary}", ""]
        for chunk in selected[:4]:
            lines.append(f"- 第 {chunk.page} 页：{chunk.text[:360]}")
        return "\n".join(lines)

    def _answer_from_uploaded_pdfs(self, question: str) -> str:
        if _is_full_pdf_analysis_request(question):
            return self._analyze_uploaded_pdf_in_full(question)

        selected = self._relevant_pdf_chunks(question)
        if not selected:
            return "当前上传的 PDF 没有可供回答的文本内容。请重新上传带文字层的 PDF。"

        excerpts = "\n\n".join(
            f"[文档：{document.name}，第 {chunk.page} 页]\n{chunk.text}"
            for document, chunk in selected
        )
        if not self.llm.available:
            references = "、".join(f"{document.name} 第 {chunk.page} 页" for document, chunk in selected[:4])
            preview = "\n\n".join(chunk.text[:500] for _, chunk in selected[:3])
            return (
                "当前没有配置 Kimi API key，无法对 PDF 进行语义分析。\n\n"
                f"已定位到可能相关的页面：{references}。\n\n"
                f"文本摘录：\n{preview}"
            )

        history = self._conversation_history_text(limit=8, per_message_limit=700)
        system = (
            "You are a careful paper reading assistant. Answer only from the supplied PDF excerpts. "
            "Do not use outside knowledge or invent claims. Cite every substantive statement with the exact format "
            "【filename，第 N 页】. Never use Markdown footnotes, square-bracket citations, caret symbols, links, or a bibliography. "
            "For a broad request such as an overview or interpretation, provide a compact but complete synthesis: topic, approach, key findings, and limitations. "
            "Do not start a numbered section unless you finish it. If the excerpts do not establish an answer, say that clearly and ask for a narrower question. "
            "Answer in the user's language and end with the exact marker 【回答完毕】."
        )
        user = (
            f"Previous PDF Q&A:\n{history or 'None'}\n\n"
            f"User question:\n{question}\n\n"
            f"Relevant excerpts:\n{excerpts}"
        )
        try:
            answer = self.llm.chat_text(
                system=system,
                user=user,
                temperature=0.2,
                max_tokens=2600,
                timeout=180,
                stream=False,
                label="pdf_qa",
            )
        except HttpError as exc:
            return f"Kimi 未能完成这次 PDF 问答：{exc}\n\n请稍后重试，或把问题缩小到一个具体章节或术语。"
        answer = _clean_pdf_answer(answer)
        return answer

    def _analyze_uploaded_pdf_in_full(self, question: str) -> str:
        """Read every extracted chunk through a map-reduce pass for broad paper interpretation."""
        document = self.state.uploaded_documents[-1] if self.state.uploaded_documents else None
        if not document:
            return "当前没有可供解析的 PDF。"
        if document.full_reading_analysis:
            return document.full_reading_analysis
        if not self.llm.available:
            return (
                "已提取整份 PDF，但当前没有配置 Kimi API key，无法完成全文语义解析。"
                "配置 Kimi 后，系统会逐段覆盖全文并生成综合解读。"
            )

        batches = _pdf_reading_batches(document.chunks)
        if not batches:
            return "当前上传的 PDF 没有可供全文解析的文本内容。"

        map_system = (
            "You are reading one consecutive part of a CS research paper. Extract only evidence present in this part: "
            "section/topic, problem, method, data or experiment, findings, limitations, and links to other parts. "
            "Keep it concise but cover every substantial point. Cite claims with 【filename，第 N 页】. "
            "Do not invent missing context, use Markdown footnotes, or write a bibliography. Answer in Chinese."
        )

        def read_batch(batch_index: int, batch: list[PdfChunk]) -> tuple[int, str]:
            pages = f"{batch[0].page}-{batch[-1].page}"
            excerpts = "\n\n".join(f"[第 {chunk.page} 页]\n{chunk.text}" for chunk in batch)
            response = self.llm.chat_text(
                system=map_system,
                user=(
                    f"论文文件：{document.name}\n"
                    f"这是全文阅读的第 {batch_index + 1}/{len(batches)} 段，覆盖第 {pages} 页。\n\n{excerpts}"
                ),
                temperature=0.2,
                max_tokens=1100,
                timeout=180,
                stream=False,
                label="pdf_full_map",
            )
            return batch_index, _clean_pdf_answer(response)

        notes: list[str] = [""] * len(batches)
        try:
            with ThreadPoolExecutor(max_workers=min(3, len(batches))) as executor:
                futures = [executor.submit(read_batch, index, batch) for index, batch in enumerate(batches)]
                for future in as_completed(futures):
                    index, note = future.result()
                    notes[index] = note
        except HttpError as exc:
            return f"Kimi 未能完成全文分段阅读：{exc}"
        except Exception as exc:
            return f"全文分段阅读失败：{exc}"

        notes = [note for note in notes if note]
        if not notes:
            return "全文分段阅读没有得到可用笔记，请稍后重试。"
        synthesis_system = (
            "You are a careful CS paper-reading assistant. Write a complete, well-structured interpretation of one paper "
            "only from the supplied full-document reading notes. Cover research question, core idea, technical approach, "
            "experimental setup/findings, contributions, limitations, and a concise takeaway. Keep citations already present "
            "in the notes in the exact format 【filename，第 N 页】. Do not invent claims, references, or uncited details. "
            "Answer in the user's language and do not leave unfinished headings."
        )
        try:
            analysis = self.llm.chat_text(
                system=synthesis_system,
                user=(
                    f"用户请求：{question}\n\n"
                    f"论文：{document.name}，共 {document.page_count} 页，已按 {len(notes)} 个连续片段覆盖阅读。\n\n"
                    f"全文阅读笔记：\n\n{'\n\n'.join(notes)}"
                ),
                temperature=0.2,
                max_tokens=2600,
                timeout=220,
                stream=False,
                label="pdf_full_synthesis",
            )
        except HttpError as exc:
            return f"全文笔记已经生成，但 Kimi 未能完成最终综合：{exc}"
        analysis = _clean_pdf_answer(analysis)
        if not analysis:
            return "全文笔记已经生成，但最终综合结果为空，请稍后重试。"
        document.full_reading_notes = notes
        document.full_reading_analysis = analysis
        return analysis

    def _relevant_pdf_chunks(self, question: str) -> list[tuple[UploadedDocument, PdfChunk]]:
        candidates: list[tuple[UploadedDocument, PdfChunk]] = []
        for document in self.state.uploaded_documents:
            candidates.extend((document, chunk) for chunk in relevant_chunks(question, document.chunks, limit=7))
        return candidates[:10]

    def _answer_with_kimi(self, question: str) -> str:
        records = [
            {
                "title": paper.title,
                "authors": paper.authors[:6],
                "year": paper.year,
                "venue": paper.venue,
                "rank": paper.venue_rank,
                "source": paper.source,
                "url": paper.source_url,
                "abstract": (paper.abstract or "")[:900],
            }
            for paper in self._evidence_papers()[:18]
        ]
        system = (
            "You are a careful CS research assistant. Answer only from the provided paper records. "
            "Do not invent papers, venues, authors, years, claims, or citations. "
            "If the records are insufficient, say what needs to be verified."
        )
        user = (
            f"最近对话：\n{self._conversation_history_text(limit=8, per_message_limit=650) or '无'}\n\n"
            f"用户问题：{question}\n\n当前检索论文记录：\n{json.dumps(records, ensure_ascii=False, indent=2)}"
        )
        try:
            return self.llm.chat_text(
                system=system,
                user=user,
                temperature=0.2,
                max_tokens=1600,
                timeout=120,
                label="answer_from_papers",
            )
        except HttpError:
            raise

    def _answer_about_generated_outputs(self, question: str) -> str:
        doc = self.state.last_generated_document
        if not doc:
            if self.state.generated_files:
                names = "、".join(file.get("name", "") for file in self.state.generated_files[:3] if file.get("name"))
                return f"当前会话里已经有生成文件：{names}。你可以直接在文件区打开它们。"
            return (
                "当前会话里还没有生成过 related work 或 BibTeX 文件。"
                "如果你的意思是要基于当前论文池写出来，可以直接说“基于当前结果生成 related work 和 bib”。"
            )

        markdown = doc.get("markdown") or {}
        bibtex = doc.get("bibtex") or {}
        claim_map = doc.get("claim_map") or {}
        preview = normalize_text(str(doc.get("preview") or ""))
        topic = normalize_text(str(doc.get("query") or "当前主题"))

        lines = [f"已经生成了基于“{topic}”的写作结果。"]
        if markdown.get("name"):
            lines.append(f"- Related work 草稿：[{markdown['name']}]({markdown.get('url') or '#'})")
        if bibtex.get("name"):
            lines.append(f"- BibTeX 引用：[{bibtex['name']}]({bibtex.get('url') or '#'})")
        if claim_map.get("name"):
            lines.append(f"- 证据映射：[{claim_map['name']}]({claim_map.get('url') or '#'})")

        asks_preview = any(token in question.lower() for token in ["在哪", "哪里", "没看到", "preview", "内容", "打开", "看下"])
        if preview and asks_preview:
            lines.extend(["", "开头预览：", preview[:520]])
        evidence_report = doc.get("evidence_report") if isinstance(doc.get("evidence_report"), dict) else None
        if evidence_report:
            lines.extend(
                [
                    "",
                    (
                        "写作依据："
                        f"{evidence_report.get('local_fulltext_count', 0)} 篇使用本地全文，"
                        f"{evidence_report.get('abstract_only_count', 0)} 篇仅使用摘要，"
                        f"{evidence_report.get('manual_upload_needed_count', 0)} 篇建议手动上传原文补全。"
                    ),
                ]
            )
        return "\n".join(lines)

    def _inspect_generated_document(self, question: str) -> str:
        """Ground generated-output follow-ups in the file contents, not just file metadata."""
        doc = self.state.last_generated_document
        if not doc:
            return self._answer_about_generated_outputs(question)
        markdown = doc.get("markdown") or {}
        filename = Path(str(markdown.get("name") or "")).name
        path = self.output_dir / filename if filename else None
        try:
            content = path.read_text(encoding="utf-8") if path and path.is_file() else ""
        except OSError:
            content = ""
        if not content.strip():
            return (
                f"我检查了上一份生成文件 `{filename or '未知文件'}`：文件内容确实为空或无法读取。"
                "请直接说“重新生成并扩展”，我会基于当前证据重新写作。"
            )

        local_summary = (
            f"我检查了 `{filename}`，它包含 {len(content.strip()):,} 个字符，不是空文件。\n\n"
            f"开头内容：\n{_document_preview(content, limit=900)}"
        )
        if not self.llm.available:
            return local_summary
        system = (
            "You inspect a generated research document for the user. Answer only from the supplied document and conversation. "
            "State clearly whether it is empty, summarize what is actually present, and identify obvious incompleteness. "
            "Do not invent papers, citations, or content that is not in the document. Answer in the user's language."
        )
        user = (
            f"Recent conversation:\n{self._conversation_history_text(limit=8, per_message_limit=600) or 'None'}\n\n"
            f"User question: {question}\n\nDocument filename: {filename}\nDocument content:\n{content[:18000]}"
        )
        try:
            return self.llm.chat_text(
                system=system,
                user=user,
                temperature=0.2,
                max_tokens=1400,
                timeout=150,
                stream=False,
                label="document_inspect",
            ).strip() or local_summary
        except HttpError:
            return local_summary

    def _free_chat(self, message: str) -> str:
        if not self.llm.available:
            return (
                "当前没有配置 Kimi API key，所以不能自由聊天。\n\n"
                "你可以在 `src/paper_agent/local_config.py` 里填写 `KIMI_API_KEY`，"
                "或者在启动服务前设置环境变量 `KIMI_API_KEY`。"
            )
        system, user = self._free_chat_prompt(message)
        try:
            answer = self.llm.chat_text(
                system=system,
                user=user,
                temperature=0.6,
                max_tokens=700,
                timeout=45,
                stream=False,
                label="free_chat",
            )
        except HttpError as exc:
            return (
                "Kimi 这次没有及时返回，可能是网络或模型响应较慢。\n\n"
                f"错误摘要：{exc}\n\n"
                "可以稍后重试，或缩短输入内容再试。"
            )
        # Preserve paragraphs, lists and headings so the browser Markdown
        # renderer receives the document structure the model produced.
        return answer.strip() or "我在。刚刚模型没有返回可展示内容，你可以继续问我检索、阅读或写作相关的问题。"

    def _free_chat_stream(self, message: str) -> Iterator[str]:
        """Create the user-facing chat stream used by the Agent's free_chat tool."""
        if not self.llm.available:
            yield (
                "当前没有配置 Kimi API key，所以不能自由聊天。\n\n"
                "你可以在 `src/paper_agent/local_config.py` 里填写 `KIMI_API_KEY`，"
                "或者在启动服务前设置环境变量 `KIMI_API_KEY`。"
            )
            return
        system, user = self._free_chat_prompt(message)
        stream_text = getattr(self.llm, "stream_text", None)
        if callable(stream_text):
            yield from stream_text(
                system=system,
                user=user,
                temperature=0.6,
                max_tokens=700,
                timeout=90,
                label="free_chat",
            )
            return
        # Lightweight fakes and third-party-compatible clients used in tests
        # may only implement chat_text. They still receive the same response
        # contract, simply as one delta.
        yield self._free_chat(message)

    def _free_chat_prompt(self, message: str) -> tuple[str, str]:
        history = self._conversation_history_text(limit=6, per_message_limit=360)
        system = (
            "You are a helpful, careful AI assistant embedded in a computer-science research workspace. "
            "Answer the user directly and honestly. Tool routing is handled by the host Agent, so never ask the user "
            "to switch modes and never claim a paper search ran unless the host supplied results. "
            "Use clean Markdown: use headings only when useful, leave blank lines between paragraphs, use lists for "
            "sequences, and do not put the whole answer on one line."
        )
        user = f"历史对话：\n{history or '无'}\n\n用户新消息：\n{message}"
        return system, user

    def _generate_document(self, request: str, *, writing_kind: str = "") -> dict:
        intent = self.state.last_intent
        evidence_papers = self._evidence_papers()
        topic = self._writing_topic(request, evidence_papers, intent)
        language = "zh" if _looks_chinese(request) else "en"
        plan = writing_plan_for(request, writing_kind=writing_kind)
        ranked_papers = rank_papers(evidence_papers, topic, limit=24) if topic else list(evidence_papers)
        selected_papers = ranked_papers or list(evidence_papers)
        evidence_query = f"{topic}\n{plan.title_for(language)}\n{request}"
        evidence_notes = self._prepare_writing_evidence(selected_papers, evidence_query)
        evidence_report = _writing_evidence_report(evidence_notes)
        draft = RelatedWorkGenerator().generate(
            query=topic,
            papers=selected_papers,
            language=language,
            use_llm=KimiRelatedWorkWriter().available,
            writing_request=request,
            evidence_notes=evidence_notes,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        md_path = self.output_dir / f"research_document_{stamp}.md"
        bib_path = self.output_dir / f"research_references_{stamp}.bib"
        claim_path = self.output_dir / f"research_claim_map_{stamp}.json"
        md_path.write_text(draft.content_markdown, encoding="utf-8")
        bib_path.write_text(draft.bibtex, encoding="utf-8")
        claim_path.write_text(
            json.dumps(
                {"query": draft.query, "claim_map": draft.claim_map, "writing_evidence": evidence_notes},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        files = [
            {"name": md_path.name, "url": f"/outputs/{md_path.name}", "kind": "markdown"},
            {"name": bib_path.name, "url": f"/outputs/{bib_path.name}", "kind": "bibtex"},
            {"name": claim_path.name, "url": f"/outputs/{claim_path.name}", "kind": "json"},
        ]
        self.state.generated_files = files + self.state.generated_files
        document = {
            "title": draft.title,
            "query": draft.query,
            "preview": _document_preview(draft.content_markdown),
            "preview_markdown": _document_markdown_preview(draft.content_markdown),
            "preview_truncated": len(draft.content_markdown) > 6_000,
            "markdown": files[0],
            "bibtex": files[1],
            "claim_map": files[2],
            "files": files,
            "outline": draft.outline,
            "writing_kind": draft.writing_kind,
            "quality_report": draft.quality_report,
            "evidence_report": evidence_report,
        }
        project_id = next((key for key, state in self.sessions.items() if state is self.state), "default")
        persisted_draft = self.store.create_draft(
            project_id,
            {
                "title": draft.title,
                "writing_kind": draft.writing_kind,
                "content_markdown": draft.content_markdown,
                "bibtex": draft.bibtex,
                "claim_map": draft.claim_map,
                "paper_ids": draft.paper_ids,
                "outline": draft.outline,
                "quality_report": draft.quality_report,
            },
        )
        document["draft_id"] = persisted_draft.get("id")
        document["draft_version"] = persisted_draft.get("version", 1)
        self.state.last_generated_document = document
        return document

    def _confirm_writing_request(self, request: str) -> dict[str, object]:
        """Use a narrow model check before a planner can create a manuscript.

        This is intentionally not a keyword or regular-expression gate.  It
        asks Kimi one binary, evidence-free question: did the user explicitly
        request a writing artefact, as opposed to only literature retrieval?
        """
        if not self.llm.available:
            # The model-driven Agent is unavailable in this case, so preserve
            # the existing offline writing behaviour for direct CLI callers.
            return {"should_write": True, "reason": "Kimi 不可用，保留显式写作调用。"}
        system = (
            "You are a strict writing-admission checker for a research assistant. Return only valid JSON. "
            "Decide whether the user explicitly requested a manuscript artefact to be written, generated, exported, "
            "or revised. Do not infer a writing request from words such as research, investigate, survey as a verb, "
            "study progress, literature retrieval, or latest developments. "
            "A request to investigate recent progress is a search request, even if it could later support a report."
        )
        user = f"""
Original user request:
{request}

Return exactly:
{{"should_write": true or false, "reason": "short Chinese explanation"}}

Examples:
- “调研一下近三年目标检测领域进展” -> {{"should_write":false,"reason":"只要求检索和调研进展"}}
- “查近三年目标检测论文，并写一份研究报告” -> {{"should_write":true,"reason":"明确要求研究报告"}}
- “基于当前文献库写一篇综述” -> {{"should_write":true,"reason":"明确要求写综述"}}
- “这些论文有什么进展” -> {{"should_write":false,"reason":"只是证据问答"}}
"""
        try:
            data = self.llm.chat_json(
                system=system,
                user=user,
                temperature=0.1,
                max_tokens=220,
                timeout=45,
                stream=False,
                label="writing_admission",
            )
        except Exception as exc:
            return {"should_write": False, "reason": f"写作准入判断未完成：{type(exc).__name__}"}
        return {
            "should_write": bool(data.get("should_write") is True),
            "reason": normalize_text(str(data.get("reason") or "未检测到明确写作交付物。")),
        }

    def _search_only_completion_answer(self, reason: object = "") -> str:
        intent = self.state.last_intent
        papers = self._evidence_papers()
        topic = getattr(intent, "display_topic", "") or getattr(intent, "normalized_topic", "") or "当前主题"
        extra = normalize_text(str(reason or ""))
        lines = [f"已完成“{topic}”的文献调研，当前结果已进入文献库，共 {len(papers)} 篇候选论文。"]
        if extra:
            lines.append(f"本次没有生成文稿：{extra}")
        lines.append("你可以继续要求筛选论文、比较研究路线，或明确提出要生成的章节/报告类型。")
        return "\n\n".join(lines)

    def _writing_topic(self, request: str, papers: list[Paper], intent) -> str:
        """Do not reuse a previous full user request as a manuscript research topic."""
        topic = normalize_text(getattr(intent, "normalized_topic", ""))
        if topic and not _looks_like_writing_request_echo(topic):
            return topic
        inferred = _topic_from_papers(papers)
        return inferred if inferred != "computer science" else normalize_text(request)

    def _prepare_writing_evidence(self, papers: list[Paper], request: str) -> list[dict]:
        """Build writing evidence that prefers local full text over abstracts."""
        notes: list[dict] = []
        updated: list[Paper] = []
        for paper in papers[:18]:
            chunks: list[PdfChunk] = []
            note = ""
            evidence_level = "metadata_only"
            manual_upload_needed = False
            local_pdf_exists = bool(paper.local_pdf_path and Path(paper.local_pdf_path).exists())
            local_text_exists = bool(paper.local_text_path and Path(paper.local_text_path).exists())
            if local_text_exists or local_pdf_exists:
                result = self.fulltext_manager.ensure_text(paper)
                updated.append(paper)
                chunks = relevant_chunks(request, result.chunks, limit=5, max_chars=9_000) if result.chunks else []
                if chunks:
                    evidence_level = "local_fulltext"
                    note = result.summary
                else:
                    evidence_level = "local_pdf_without_text"
                    manual_upload_needed = True
                    note = result.summary
            elif paper.abstract:
                evidence_level = "abstract_only"
                note = "数据库中有摘要，但当前没有本地全文。详细方法、实验和局限需要下载或手动上传原文后再确认。"
            elif paper.pdf_url and paper.fulltext_status != "failed":
                evidence_level = "downloadable_metadata"
                note = "当前只有元数据，但存在开放 PDF 链接；可先下载 PDF 到个人文献库后再生成更可靠草稿。"
            else:
                evidence_level = "metadata_only"
                manual_upload_needed = True
                note = paper.fulltext_error or "当前只有元数据，且没有可直接下载的开放 PDF。建议手动上传原文补全。"

            excerpts = [
                {"page": chunk.page, "text": chunk.text[:1800]}
                for chunk in chunks
            ]
            notes.append(
                {
                    "paper_id": paper.id,
                    "title": paper.title,
                    "evidence_level": evidence_level,
                    "fulltext_status": paper.fulltext_status,
                    "local_pdf_path": paper.local_pdf_path,
                    "local_text_path": paper.local_text_path,
                    "excerpt_count": len(excerpts),
                    "excerpts": excerpts,
                    "manual_upload_needed": manual_upload_needed or paper.fulltext_status == "failed",
                    "note": note,
                }
            )
        if updated:
            self.store.save_papers(updated)
            for paper in updated:
                self._upsert_session_paper(paper)
        return notes

    def _generate_pdf_document(self, request: str) -> dict:
        selected = self._relevant_pdf_chunks(request)
        latest_document = self.state.uploaded_documents[-1] if self.state.uploaded_documents else None
        if latest_document and latest_document.full_reading_notes:
            excerpts = "全文阅读笔记（已覆盖整篇论文）：\n\n" + "\n\n".join(latest_document.full_reading_notes)
        else:
            excerpts = "\n\n".join(
                f"[文档：{document.name}，第 {chunk.page} 页]\n{chunk.text}"
                for document, chunk in selected
            )
        if self.llm.available:
            system = (
                "You are a careful academic writing assistant. Draft only the section requested by the user using the supplied "
                "PDF excerpts. Do not invent external papers, authors, results, or bibliography entries. Cite every factual claim "
                "as 【filename，第 N 页】. For a related work request, write a coherent related-work-style synthesis of how the uploaded "
                "paper positions prior work; clearly state that this is grounded only in the uploaded paper, not an external literature search. "
                "Do not use Markdown footnotes, URLs, square-bracket citations, or a reference list. Answer in the user's language."
            )
            user = f"用户请求：{request}\n\n可用 PDF 证据：\n{excerpts}"
            try:
                content = self.llm.chat_text(
                    system=system,
                    user=user,
                    temperature=0.2,
                    max_tokens=2400,
                    timeout=180,
                    stream=False,
                    label="pdf_document_writer",
                )
                content = _clean_pdf_answer(content)
                if not content.strip():
                    content = "无法从当前 PDF 证据生成正文：模型返回了空内容，请重试或缩小写作范围。"
            except HttpError as exc:
                content = f"无法完成基于 PDF 的文档生成：{exc}"
        else:
            content = (
                "# 基于上传论文的写作草稿\n\n"
                "当前未配置 Kimi API key，无法生成完整草稿。以下是已提取的相关页面：\n\n"
                + "\n\n".join(f"【{document.name}，第 {chunk.page} 页】\n{chunk.text[:700]}" for document, chunk in selected)
            )

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"uploaded_pdf_document_{stamp}.md"
        path.write_text(content, encoding="utf-8")
        file = {"name": path.name, "url": f"/outputs/{path.name}", "kind": "markdown"}
        self.state.generated_files = [file] + self.state.generated_files
        document = {
            "title": "基于上传论文的写作草稿",
            "query": request,
            "preview": _document_preview(content),
            "preview_markdown": _document_markdown_preview(content),
            "preview_truncated": len(content) > 6_000,
            "markdown": file,
            "bibtex": None,
            "claim_map": None,
            "files": [file],
            "outline": [],
            "writing_kind": "uploaded_pdf_section",
            "quality_report": {
                "writing_kind": "uploaded_pdf_section",
                "word_count": len(re.sub(r"\s+", "", content)),
                "citation_uses": len(re.findall(r"【[^】]+，第\s*\d+\s*页】", content)),
                "cited_paper_count": 0,
                "available_paper_count": 0,
                "fulltext_paper_count": 1 if selected else 0,
                "weak_evidence_count": 0 if selected else 1,
                "warnings": [] if selected else ["没有找到与请求直接相关的 PDF 页面，草稿需要人工核对。"],
            },
        }
        project_id = next((key for key, state in self.sessions.items() if state is self.state), "default")
        persisted_draft = self.store.create_draft(
            project_id,
            {
                "title": document["title"],
                "writing_kind": document["writing_kind"],
                "content_markdown": content,
                "outline": [],
                "quality_report": document["quality_report"],
            },
        )
        document["draft_id"] = persisted_draft.get("id")
        document["draft_version"] = persisted_draft.get("version", 1)
        self.state.last_generated_document = document
        return document

    def _remember_conversation(self, role: str, content: str) -> None:
        text = normalize_text(content or "")
        if not text:
            return
        self.state.conversation_history.append({"role": role, "content": text[:5000]})
        self.state.conversation_history = self.state.conversation_history[-28:]
        project_id = next((key for key, state in self.sessions.items() if state is self.state), "default")
        self.store.append_project_message(project_id, role, text[:5000])

    def _remember_research_topic(self, topic: str, display_topic: str = "") -> None:
        topic = normalize_text(topic)
        if not topic:
            return
        existing = [item for item in self.state.research_topics if item.lower() != topic.lower()]
        self.state.research_topics = [topic, *existing][:12]

    def _discovery_profile(self, state: AssistantState, *, topic: str = "") -> dict:
        self._repair_legacy_echoed_topic(state)
        topics: list[str] = []
        for candidate in [topic, *(state.research_topics or [])]:
            candidate = normalize_text(candidate)
            if candidate and candidate.lower() not in {item.lower() for item in topics}:
                topics.append(candidate)
        if state.last_intent and state.last_intent.normalized_topic:
            candidate = normalize_text(state.last_intent.normalized_topic)
            if candidate and candidate.lower() not in {item.lower() for item in topics}:
                topics.append(candidate)
        if not topics and state.papers:
            topics.append(_topic_from_papers(state.papers))
        primary_topic = topics[0] if topics else "computer science"
        display_topic = normalize_text(getattr(state.last_intent, "display_topic", "")) if state.last_intent else ""
        if not display_topic:
            display_topic = primary_topic
        evidence = _state_evidence_papers(state)
        seed_titles = [paper.title for paper in evidence[:8] if paper.title]
        seed_abstracts = [paper.abstract for paper in evidence[:5] if paper.abstract]
        return {
            "primary_topic": primary_topic,
            "display_topic": display_topic,
            "topics": topics[:5] or [primary_topic],
            "seed_titles": seed_titles,
            "seed_abstracts": seed_abstracts,
            "paper_count": len(state.papers),
        }

    def _repair_legacy_echoed_topic(self, state: AssistantState) -> None:
        """Repair persisted sessions produced before topic fields were strict."""
        intent = state.last_intent
        if not intent or not _is_request_echo_topic(intent.normalized_topic, intent.original_request):
            return
        repaired = self.intent_analyzer.repair_echoed_topic(intent.original_request, intent.normalized_topic)
        normalized_topic = normalize_text(str(repaired.get("normalized_topic") or ""))
        if not normalized_topic:
            return
        previous_topic = intent.normalized_topic
        intent.normalized_topic = normalized_topic
        intent.display_topic = normalize_text(str(repaired.get("display_topic") or ""))
        if repaired.get("keywords"):
            intent.keywords = [str(item) for item in repaired["keywords"]]
        if repaired.get("queries"):
            intent.queries = [str(item) for item in repaired["queries"]]
        if repaired.get("cs_area"):
            intent.cs_area = normalize_text(str(repaired["cs_area"]))
        state.research_topics = [
            normalized_topic,
            *(item for item in state.research_topics if item.casefold() not in {previous_topic.casefold(), normalized_topic.casefold()}),
        ][:12]

    def _conversation_history_text(self, *, limit: int, per_message_limit: int) -> str:
        rows = []
        for item in self.state.conversation_history[-limit:]:
            role = "用户" if item.get("role") == "user" else "助手"
            content = normalize_text(str(item.get("content") or ""))[:per_message_limit]
            if content:
                rows.append(f"{role}: {content}")
        return "\n".join(rows)

    def _conversation_context(self) -> str:
        parts = []
        history = self._conversation_history_text(limit=10, per_message_limit=520)
        if history:
            parts.append(f"Recent conversation:\n{history}")
        if self.state.uploaded_documents:
            documents = "; ".join(
                f"{document.name} ({document.page_count} pages)" for document in self.state.uploaded_documents[-3:]
            )
            parts.append(f"Uploaded PDFs: {documents}")
        evidence = self._evidence_papers()
        if evidence:
            titles = "; ".join(
                f"{paper.id}: {paper.title[:120]} [fulltext={paper.fulltext_status or 'none'}]"
                for paper in evidence[:6]
            )
            parts.append(f"Current retrieved papers ({len(evidence)}): {titles}")
        doc = self.state.last_generated_document
        if doc:
            markdown = doc.get("markdown") or {}
            filename = Path(str(markdown.get("name") or "")).name
            preview = normalize_text(str(doc.get("preview") or ""))[:900]
            parts.append(
                f"Latest generated document: {filename or 'unknown'}, request={normalize_text(str(doc.get('query') or ''))[:240]}; "
                f"preview={preview or 'no preview'}"
            )
        return "\n\n".join(parts)[:7000]

    def _paper_payloads(self, papers: list[Paper]) -> list[dict]:
        return [
            {
                "id": paper.id,
                "title": paper.title,
                "authors": paper.authors[:8],
                "year": paper.year,
                "venue": paper.venue,
                "venue_rank": paper.venue_rank,
                "source": paper.source,
                "source_url": paper.source_url,
                "pdf_url": paper.pdf_url,
                "doi": paper.doi,
                "abstract": paper.abstract,
                "abstract_status": paper.abstract_status,
                "local_pdf_path": paper.local_pdf_path,
                "local_pdf_display_path": str(Path(paper.local_pdf_path).resolve()) if paper.local_pdf_path else None,
                "local_pdf_url": (
                    f"/api/library/paper/file?paper_id={paper.id}"
                    if paper.local_pdf_path and Path(paper.local_pdf_path).exists()
                    else None
                ),
                "local_text_path": paper.local_text_path,
                "fulltext_status": paper.fulltext_status,
                "fulltext_error": paper.fulltext_error,
                "manual_upload_needed": (
                    paper.fulltext_status == "failed"
                    or (not paper.local_pdf_path and not paper.pdf_url)
                ),
                "fulltext_tip": _paper_fulltext_tip(paper),
                "fulltext_downloaded_at": paper.fulltext_downloaded_at,
                "reading_status": paper.reading_status,
                "importance": paper.importance,
                "user_tags": paper.user_tags,
                "excluded": paper.excluded,
                "exclusion_reason": paper.exclusion_reason,
                "user_notes": paper.user_notes,
                "used_in_sections": paper.used_in_sections,
                "relevance_score": paper.relevance_score,
                "added_at": paper.added_at,
                "updated_at": paper.updated_at,
                "retrieved_at": paper.retrieved_at,
            }
            for paper in papers
        ]

    def _intent_payload(self, intent: ResearchIntent) -> dict:
        return asdict(intent)

    def _source_summary(self, papers: list[Paper]) -> str:
        counts: dict[str, int] = {}
        for paper in papers:
            key = paper.source or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return "，".join(f"{source} {count} 篇" for source, count in sorted(counts.items()))

    def _evidence_papers(self) -> list[Paper]:
        active_papers = [paper for paper in self.state.papers if not paper.excluded]
        # A systematic-review inclusion decision is stronger than the general
        # project selection. Once the user has screened at least one study as
        # included, manuscript generation must not quietly reintroduce studies
        # they explicitly excluded or left outside the final evidence set.
        project_id = next((key for key, state in self.sessions.items() if state is self.state), "default")
        included_ids = {
            str(item.get("paper_id"))
            for item in self.store.list_review_screenings(project_id)
            if item.get("decision") == "include"
        }
        if included_ids:
            included = [paper for paper in active_papers if paper.id in included_ids]
            if included:
                return included
        if not self.state.evidence_paper_ids:
            return active_papers
        wanted = set(self.state.evidence_paper_ids)
        evidence = [paper for paper in active_papers if paper.id in wanted]
        return evidence or active_papers

    def _match_uploaded_pdf(
        self,
        extraction,
        filename: str,
        *,
        candidates: list[Paper] | None = None,
    ) -> tuple[Paper | None, dict]:
        """Match a PDF to a library record without relying on an LLM or network call."""
        context = _pdf_match_context(extraction, filename)
        title_candidates: dict[str, Paper] = {}
        candidate_records = candidates if candidates is not None else [*self.state.papers, *self.store.load_papers()]
        for paper in candidate_records:
            if paper.id and not paper.excluded:
                # The same work can survive a migration under multiple IDs. Score
                # one representative per normalized title so duplicate records do
                # not turn an otherwise exact match into a false ambiguity.
                title_key = _paper_title_key(paper.title)
                if title_key:
                    existing = title_candidates.get(title_key)
                    if existing is None or _paper_match_record_score(paper) > _paper_match_record_score(existing):
                        title_candidates[title_key] = paper
        scored = sorted(
            ((paper, _pdf_title_match_score(paper.title, context)) for paper in title_candidates.values()),
            key=lambda item: item[1],
            reverse=True,
        )
        if not scored or scored[0][1] < 0.82:
            return None, {"status": "unmatched", "confidence": round(scored[0][1], 3) if scored else 0.0, "reason": "未从文件名或 PDF 首页识别出足够接近的论文标题。"}
        best_paper, best_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        if second_score >= 0.74 and best_score - second_score < 0.10:
            return None, {"status": "ambiguous", "confidence": round(best_score, 3), "reason": "匹配到多个相近标题，未自动绑定以避免误关联。"}
        return best_paper, {
            "status": "matched",
            "confidence": round(best_score, 3),
            "reason": "已根据文件名和 PDF 首页文本自动匹配论文标题。",
        }

    def _paper_by_id(self, paper_id: str) -> Paper | None:
        clean_id = normalize_text(paper_id)
        if not clean_id:
            return None
        canonical_id = self.store.resolve_paper_id(clean_id) or clean_id
        for paper in self.state.papers:
            if paper.id == canonical_id:
                return paper
        for paper in self.store.load_papers():
            if paper.id == canonical_id:
                return paper
        return None

    def _upsert_session_paper(self, updated: Paper) -> None:
        for index, paper in enumerate(self.state.papers):
            if paper.id == updated.id:
                self.state.papers[index] = updated
                return
        self.state.papers.append(updated)

    def _resolve_fulltext_paper(self, arguments: dict) -> Paper | None:
        papers = self._evidence_papers()
        if not papers:
            return None
        paper_id = normalize_text(str(arguments.get("paper_id") or ""))
        if paper_id:
            for paper in papers:
                if paper.id == paper_id:
                    return paper
        title = normalize_text(str(arguments.get("title") or "")).lower()
        if title:
            for paper in papers:
                if title in paper.title.lower() or paper.title.lower() in title:
                    return paper
        index_value = arguments.get("index")
        try:
            index = int(index_value)
        except (TypeError, ValueError):
            index = 1
        index = max(1, min(index, len(papers)))
        return papers[index - 1]

    def _paper_hint(self, paper: Paper) -> str:
        if paper.abstract:
            sentence = re.split(r"(?<=[.!?。！？])\s+", paper.abstract.strip())[0]
            return sentence[:180]
        return "DBLP 元数据结果，建议点击来源链接查看摘要或全文。"

    def _finalize_debug(self, *, action: str, kimi_log_start: int, search_used: bool) -> dict:
        kimi_calls = KimiClient.calls_since(kimi_log_start)
        successful_calls = [call for call in kimi_calls if call.get("success")]
        failed_calls = [call for call in kimi_calls if not call.get("success")]
        debug = {
            "action": action,
            "tool_plan": self.state.last_tool_plan,
            "action_source": getattr(self.intent_analyzer, "last_action_trace", {}),
            "intent_source": getattr(self.intent_analyzer, "last_research_trace", {}) if search_used else {},
            "search": self._last_search_debug if search_used else {
                "used_existing_evidence": bool(self._evidence_papers() or self.state.uploaded_documents),
                "uploaded_documents": len(self.state.uploaded_documents),
            },
            "kimi": {
                "configured": self.llm.available,
                "called": bool(kimi_calls),
                "success_count": len(successful_calls),
                "failure_count": len(failed_calls),
                "calls": kimi_calls[-6:],
            },
        }
        self.state.last_debug = debug
        return debug

def _looks_chinese(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", value))


def _safe_pdf_filename(value: str) -> str:
    name = Path(value or "uploaded-paper.pdf").name
    name = re.sub(r"[^A-Za-z0-9._() -]+", "_", name).strip(" .")
    return name[:160] or "uploaded-paper.pdf"


def _pdf_match_context(extraction, filename: str) -> str:
    """Use only the first page and filename for a cheap, conservative match."""
    first_page = "\n".join(
        chunk.text for chunk in getattr(extraction, "chunks", [])
        if getattr(chunk, "page", 0) == 1
    )[:8_000]
    filename_stem = Path(filename or "").stem.replace("_", " ").replace("-", " ")
    return f"{filename_stem}\n{first_page}"


_GENERIC_PDF_HEADING_TOKENS = {
    "abstract", "acknowledgement", "acknowledgements", "acknowledgment", "acknowledgments",
    "appendix", "background", "conclusion", "conclusions", "contents", "discussion",
    "experiment", "experiments", "introduction", "keyword", "keywords", "method", "methods",
    "overview", "preliminary", "references", "related", "result", "results", "survey", "work",
}
_TITLE_MATCH_STOP_TOKENS = {
    "a", "an", "and", "based", "by", "for", "from", "in", "of", "on", "that", "the",
    "this", "to", "toward", "towards", "using", "via", "with",
}


def _paper_title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def _paper_match_record_score(paper: Paper) -> tuple[int, int, int, int, int]:
    return (
        int(bool(paper.doi or paper.arxiv_id)),
        int(bool(paper.abstract)),
        int(bool(paper.source_url)),
        int(bool(paper.pdf_url)),
        int(paper.year or 0),
    )


def _pdf_title_match_score(title: str, context: str) -> float:
    title_key = _paper_title_key(title)
    context_key = re.sub(r"[^a-z0-9]+", "", (context or "").lower())
    if len(title_key) < 12 or not context_key:
        return 0.0

    title_tokens = {
        token for token in re.findall(r"[a-z][a-z0-9]+", (title or "").lower())
        if len(token) >= 3 and token not in _TITLE_MATCH_STOP_TOKENS and token not in _GENERIC_PDF_HEADING_TOKENS
    }
    # Headings such as "Introduction" and "Related Work" occur in nearly every
    # PDF. They must never compete with a real paper title during auto-binding.
    if len(title_tokens) < 2:
        return 0.0
    if title_key in context_key:
        return 1.0

    context_tokens = set(re.findall(r"[a-z][a-z0-9]+", (context or "").lower()))
    coverage = len(title_tokens & context_tokens) / len(title_tokens) if title_tokens else 0.0
    lines = [re.sub(r"[^a-z0-9]+", "", line.lower()) for line in (context or "").splitlines() if line.strip()]
    sequence = max((SequenceMatcher(None, title_key, line).ratio() for line in lines if len(line) >= 12), default=0.0)
    return 0.62 * coverage + 0.38 * sequence


def _looks_like_pdf_question(question: str) -> bool:
    lowered = question.lower()
    markers = ("pdf", "这篇", "本文", "文章", "论文中", "上传", "第几页", "作者", "摘要", "方法", "实验")
    return any(marker in lowered for marker in markers)


def _is_full_pdf_analysis_request(question: str) -> bool:
    lowered = question.lower().strip()
    broad_markers = (
        "全文",
        "通读",
        "完整解析",
        "全面解析",
        "整体分析",
        "解析一下",
        "解读一下",
        "总结这篇",
        "概括这篇",
        "analyze this paper",
        "parse this paper",
        "paper overview",
        "full paper",
    )
    specific_markers = ("第", "page", "章节", "section", "具体", "术语", "公式")
    return any(marker in lowered for marker in broad_markers) and not any(marker in lowered for marker in specific_markers)


def _pdf_reading_batches(chunks: list[PdfChunk], *, target_batches: int = 12) -> list[list[PdfChunk]]:
    """Partition all extracted chunks into consecutive model-sized reading segments."""
    if not chunks:
        return []
    total_chars = sum(len(chunk.text) for chunk in chunks)
    target_chars = max(12_000, min(26_000, (total_chars + target_batches - 1) // target_batches))
    batches: list[list[PdfChunk]] = []
    current: list[PdfChunk] = []
    current_size = 0
    for chunk in chunks:
        if current and current_size + len(chunk.text) > target_chars:
            batches.append(current)
            current = []
            current_size = 0
        current.append(chunk)
        current_size += len(chunk.text)
    if current:
        batches.append(current)
    return batches


def _requires_generated_file(request: str) -> bool:
    lowered = request.lower()
    return any(marker in lowered for marker in ("导出", "下载", "保存为", "文件", ".md", ".docx", ".bib", "bibtex"))


def _fallback_tool_plan(action: str, *, has_documents: bool, has_evidence: bool) -> list[str]:
    if action == "chat":
        return ["free_chat"]
    if action == "search":
        return ["paper_search"]
    if action == "document":
        plan = []
        if has_documents:
            plan.append("pdf_read")
        if not has_evidence and not has_documents:
            plan.append("paper_search")
        plan.append("write_document")
        return plan
    if has_documents:
        return ["pdf_read"]
    return ["evidence_answer"]


def _agent_action_for_tool(tool_name: str) -> str:
    if tool_name == "paper_search":
        return "search"
    if tool_name == "write_document":
        return "document"
    if tool_name == "free_chat":
        return "chat"
    return "answer"


def _agent_tool_status(tool_name: str) -> str:
    messages = {
        "paper_search": "Agent 正在检索本地缓存及外部论文来源。",
        "pdf_read": "Agent 正在阅读上传 PDF 并提取证据。",
        "evidence_answer": "Agent 正在依据当前证据池组织答案。",
        "paper_fulltext_read": "Agent 正在读取本地全文缓存，必要时下载开放 PDF。",
        "write_document": "Agent 正在根据已收集证据生成文档。",
        "document_inspect": "Agent 正在核对上一份生成文档的实际内容。",
        "free_chat": "Agent 正在进行自由聊天。",
    }
    return messages.get(tool_name, "Agent 正在调用工具。")


def _tool_call_signature(tool_name: str, arguments: dict) -> str:
    try:
        payload = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
    except TypeError:
        payload = repr(arguments)
    return f"{tool_name}:{payload}"


def _is_source_rate_or_permission_error(value: str) -> bool:
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in (
            "429",
            "too many requests",
            "403",
            "forbidden",
            "rate exceeded",
            "anonymous search is paused",
            "search temporarily unavailable",
        )
    )


def _search_answer_summary(intent: ResearchIntent, papers: list[Paper], source_summary: str) -> str:
    if not papers:
        return "没有在当前约束下找到可用论文。可以放宽年份、会议/期刊限制，或换一个更短的英文关键词重试。"
    lines = [
        f"已完成检索：**{intent.normalized_topic}**。",
        "",
        f"- 当前证据池得到 {len(papers)} 篇论文。",
    ]
    if source_summary:
        lines.append(f"- 来源状态：{source_summary}")
    abstract_count = sum(1 for paper in papers if paper.abstract)
    fulltext_ready = sum(1 for paper in papers if paper.fulltext_status == "extracted")
    lines.append(f"- 摘要可用：{abstract_count} 篇；本地全文已缓存：{fulltext_ready} 篇。")
    lines.append("")
    lines.append("代表论文：")
    for index, paper in enumerate(papers[:6], start=1):
        lines.append(f"{index}. {paper.title}（{paper.year or 'n.d.'}，{paper.venue or paper.source}）")
    return "\n".join(lines)


def _search_source_summary(reports: dict) -> str:
    labels = {
        "openalex": "OpenAlex",
        "dblp": "DBLP",
        "arxiv": "arXiv",
        "semantic_scholar": "Semantic Scholar",
        "google_scholar": "Google Scholar",
    }
    rows = []
    for source, label in labels.items():
        report = reports.get(source) or {}
        count = report.get("successful_queries", 0)
        total = report.get("queries", 0)
        empty = report.get("empty_queries", 0)
        failures = report.get("failures") or []
        if failures and all("SERPAPI_API_KEY" in str(error) for error in failures):
            rows.append(f"{label} 未配置")
        elif failures:
            reason = _summarize_source_failures(failures)
            rows.append(f"{label} {count}/{total} 个查询有结果，{empty} 个空结果，{len(failures)} 个失败：{reason}")
        elif total:
            rows.append(f"{label} {count}/{total} 个查询有结果")
    return "；".join(rows)


def _summarize_source_failures(failures: list) -> str:
    cleaned = []
    for failure in failures:
        text = normalize_text(str(failure))
        text = re.sub(r"https?://\S+", "", text).strip()
        text = text.replace("Paper search failed:", "").strip()
        if text and text not in cleaned:
            cleaned.append(text[:120])
    return "；".join(cleaned[:2]) or "来源请求失败"


def _clean_pdf_answer(answer: str) -> str:
    """Normalize model citation variants into plain text that the chat renderer cannot misinterpret."""
    cleaned = (answer or "").replace("【回答完毕】", "").strip()
    citation_pattern = re.compile(
        r"\[\s*\^?\s*([^,\]\n]+?)\s*,\s*(?:p(?:age)?\.?|第)?\s*(\d+)\s*(?:页)?\s*\^?\s*\]",
        flags=re.IGNORECASE,
    )
    cleaned = citation_pattern.sub(lambda match: f"【{match.group(1).strip()}，第 {match.group(2)} 页】", cleaned)
    cleaned = re.sub(r"(?:\n|\s)\*{1,2}\s*\d*\.?\s*$", "", cleaned).rstrip()
    return cleaned


def _source_query_plan(intent: ResearchIntent, *, max_queries: int = 6) -> dict[str, list[str]]:
    fallback = _expanded_queries(intent)
    source_queries = intent.source_queries or {}
    facet_queries = _facet_query_variants(intent)
    plan = {
        "openalex": _covered_source_queries(source_queries.get("openalex"), _merge_queries(facet_queries, fallback), max_queries=max_queries),
        "dblp": _covered_source_queries(source_queries.get("dblp"), fallback, _relaxed_topic_queries(intent.normalized_topic), max_queries=max_queries),
        "arxiv": _covered_source_queries(source_queries.get("arxiv"), fallback, max_queries=max_queries),
        "semantic_scholar": _covered_source_queries(source_queries.get("semantic_scholar"), _merge_queries(facet_queries, fallback), max_queries=max_queries),
        "google_scholar": _covered_source_queries(source_queries.get("google_scholar"), _merge_queries(facet_queries, fallback), max_queries=max_queries),
    }
    return plan


def _facet_query_variants(intent: ResearchIntent) -> list[str]:
    groups = _topic_constraint_groups(intent)
    if len(groups) < 2:
        return []
    short_groups: list[list[str]] = []
    for group in groups:
        terms = [str(term) for term in group.get("terms", []) if str(term).isascii()]
        if not terms:
            continue
        compact = []
        for term in terms:
            if len(compact) >= 4:
                break
            if " " in term:
                compact.append(f'"{term}"')
            else:
                compact.append(term)
        if compact:
            short_groups.append(compact)
    if len(short_groups) < 2:
        return []
    boolean_query = " AND ".join("(" + " OR ".join(group) + ")" for group in short_groups)
    head_terms = " ".join(group[0].strip('"') for group in short_groups)
    return [boolean_query, head_terms]


def _enrich_missing_metadata(search_service, papers: list[Paper], *, max_items: int = 12) -> dict[str, object]:
    enrich = getattr(search_service, "enrich_missing_metadata", None)
    if not callable(enrich):
        return {"attempted": 0, "enriched": 0, "failures": ["当前检索服务不支持元数据补全。"]}
    try:
        return enrich(papers, max_items=max_items)
    except Exception as exc:
        return {"attempted": 0, "enriched": 0, "failures": [str(exc)[:300]]}


def _topic_constraint_groups(intent: ResearchIntent) -> list[dict[str, object]]:
    topic_text = " ".join([intent.original_request, intent.normalized_topic, *intent.keywords, *intent.queries]).lower()
    groups: list[dict[str, object]] = []
    if _contains_any(topic_text, ("debias", "bias", "fair", "unbias", "公平", "去偏", "偏见", "无偏")):
        groups.append(
            {
                "name": "debiasing_or_fairness",
                "terms": [
                    "debias",
                    "de-bias",
                    "bias mitigation",
                    "bias-aware",
                    "unbiased",
                    "fairness",
                    "fair",
                    "equity",
                    "equitable",
                    "discrimination",
                    "去偏",
                    "公平",
                    "偏见",
                    "无偏",
                ],
            }
        )
    if _contains_any(topic_text, ("dynamic", "temporal", "time-aware", "sequential", "session-based", "动态", "时序", "序列")):
        groups.append(
            {
                "name": "dynamic_or_temporal",
                "terms": [
                    "dynamic",
                    "temporal",
                    "time-aware",
                    "time aware",
                    "sequential",
                    "session-based",
                    "session based",
                    "evolving",
                    "drift",
                    "streaming",
                    "动态",
                    "时序",
                    "序列",
                ],
            }
        )
    if _contains_any(topic_text, ("recommender", "recommendation", "推荐")):
        groups.append(
            {
                "name": "recommender_or_recommendation",
                "terms": [
                    "recommender",
                    "recommendation",
                    "recommend",
                    "recsys",
                    "collaborative filtering",
                    "推荐",
                ],
            }
        )
    if _contains_any(topic_text, ("open-vocabulary", "open vocabulary", "开放词汇")):
        groups.append(
            {
                "name": "open_vocabulary",
                "terms": ["open-vocabulary", "open vocabulary", "open-set", "open set", "开放词汇"],
            }
        )
    if _contains_any(topic_text, ("object detection", "目标检测")):
        groups.append(
            {
                "name": "object_detection",
                "terms": ["object detection", "detector", "detection", "目标检测", "检测"],
            }
        )
    return groups


def _passes_topic_constraints(paper: Paper, groups: list[dict[str, object]]) -> bool:
    if not groups:
        return True
    evidence_text = " ".join(
        [
            paper.title,
            paper.abstract or "",
            paper.venue or "",
            " ".join(paper.fields_of_study),
        ]
    ).lower()
    return all(_contains_any(evidence_text, tuple(group.get("terms", []))) for group in groups)


def _contains_any(text: str, terms: tuple | list) -> bool:
    return any(str(term).lower() in text for term in terms if str(term or "").strip())


def _covered_source_queries(
    preferred: list[str] | None,
    fallback: list[str],
    relaxed: list[str] | None = None,
    *,
    max_queries: int = 6,
) -> list[str]:
    """Keep a source-specific synonym while guaranteeing a canonical fallback."""
    preferred = [query for query in preferred or [] if query]
    fallback = [query for query in fallback if query]
    relaxed = [query for query in relaxed or [] if query]
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred[0])
    if fallback:
        candidates.append(fallback[0])
    candidates.extend(preferred[1:])
    candidates.extend(relaxed)
    candidates.extend(fallback[1:])
    return list(dict.fromkeys(candidates))[:max_queries]


def _low_recall_threshold(limit: int) -> int:
    return min(12, max(6, limit // 3))


def _expanded_queries(intent: ResearchIntent) -> list[str]:
    queries = [intent.normalized_topic, *intent.queries]
    keyword_query = " ".join(intent.keywords[:5])
    if keyword_query:
        queries.append(keyword_query)
    queries.extend(_relaxed_topic_queries(intent.normalized_topic))
    for query in list(queries):
        queries.extend(_abbreviation_queries(query))
    return list(dict.fromkeys(query for query in queries if query))


def _relaxed_topic_queries(topic: str) -> list[str]:
    tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9-]+", topic.lower()) if len(token) > 2]
    queries: list[str] = []
    if len(tokens) >= 4:
        queries.append(" ".join(tokens[:4]))
    if len(tokens) >= 3:
        queries.append(" ".join(tokens[-3:]))
    if len(tokens) >= 2:
        queries.append(" ".join(tokens[:2]))
    return queries


def _abbreviation_queries(query: str) -> list[str]:
    text = query.lower()
    variants: list[str] = []
    if "dynamic" in text and ("recommender" in text or "recommendation" in text or "rec" in text):
        variants.extend(
            [
                "dynamic rec",
                "dynamic recommender",
                "dynamic recommendation",
                "dynamic recommender systems",
                "temporal recommendation",
                "time-aware recommendation",
                "sequential rec",
            ]
        )
    if "recommender systems" in text:
        variants.append(text.replace("recommender systems", "rec"))
        variants.append(text.replace("recommender systems", "recommendation"))
    if "recommendation" in text and "rec" not in text:
        variants.append(text.replace("recommendation", "rec"))
    if "large language model" in text:
        variants.append(text.replace("large language model", "llm"))
    if "retrieval augmented generation" in text or "retrieval-augmented generation" in text:
        variants.append(text.replace("retrieval augmented generation", "rag").replace("retrieval-augmented generation", "rag"))
    return variants


def _merge_queries(*groups: list[str] | None) -> list[str]:
    merged: list[str] = []
    for group in groups:
        merged.extend(group or [])
    return list(dict.fromkeys(query for query in merged if query))


def _clean_id_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = normalize_text(str(item or ""))
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _looks_like_generated_output_question(question: str) -> bool:
    lowered = question.lower()
    file_terms = [
        "related work",
        "bib",
        "bibtex",
        "markdown",
        "claim map",
        "文件",
        "文档",
        "草稿",
        "生成",
        "导出",
        "在哪",
        "哪里",
        "没看到",
        "没有生成",
        "没生成",
    ]
    return any(term in lowered for term in file_terms)


def _document_preview(content: str, *, limit: int = 520) -> str:
    text = normalize_text(re.sub(r"[#*_`>-]+", " ", content or ""))
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _document_markdown_preview(content: str, *, limit: int = 6_000) -> str:
    """Keep manuscript structure for the in-app preview instead of flattening Markdown."""
    text = (content or "").strip()
    if len(text) <= limit:
        return text
    boundary = text.rfind("\n\n", 0, limit)
    if boundary < limit // 2:
        boundary = limit
    return text[:boundary].rstrip() + "\n\n> 预览已截断，请打开完整 Markdown 文件查看剩余内容。"


def _bibtex_keys(bibtex: str) -> list[str]:
    return re.findall(r"@\w+\s*\{\s*([^,\s]+)", bibtex or "", flags=re.I)


def _clean_session_id(session_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "-", session_id or "default")[:80] or "default"


def _is_path_inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except (OSError, ValueError):
        return False


def _state_evidence_papers(state: AssistantState) -> list[Paper]:
    if not state.evidence_paper_ids:
        return state.papers
    wanted = set(state.evidence_paper_ids)
    evidence = [paper for paper in state.papers if paper.id in wanted]
    return evidence or state.papers


def _topic_from_papers(papers: list[Paper]) -> str:
    corpus = " ".join(f"{paper.title} {paper.abstract or ''}".lower() for paper in papers[:18])
    if re.search(r"\b(debias|debiasing|unbiased|fairness|exposure bias)\b", corpus) and re.search(
        r"\b(recommend|recommender|recommendation)\b", corpus
    ):
        return "debiasing recommender systems"
    tokens: dict[str, int] = {}
    stopwords = {
        "paper", "papers", "study", "studies", "system", "systems", "using", "based",
        "towards", "toward", "with", "from", "large", "language", "model", "models",
    }
    for paper in papers[:12]:
        text = f"{paper.title} {paper.abstract or ''}".lower()
        for token in re.findall(r"[a-z][a-z0-9\-]{3,}", text):
            if token not in stopwords:
                tokens[token] = tokens.get(token, 0) + 1
    top = [token for token, _ in sorted(tokens.items(), key=lambda item: item[1], reverse=True)[:5]]
    return " ".join(top) or "computer science"


def _looks_like_writing_request_echo(value: str) -> bool:
    compact = re.sub(r"\s+", "", value or "")
    return not compact or any(marker in compact for marker in ("调研", "工作进展", "最近", "近三年", "查一下", "我想", "文献库", "写一篇"))


def _is_request_echo_topic(topic: str, request: str) -> bool:
    """Only identify an exact stale request echo; never classify a topic locally."""
    normalized_topic = normalize_text(topic).casefold()
    normalized_request = normalize_text(request).casefold()
    return bool(normalized_topic and normalized_request and normalized_topic == normalized_request)


def _writing_evidence_report(notes: list[dict]) -> dict:
    local_fulltext_count = sum(1 for item in notes if item.get("evidence_level") == "local_fulltext")
    abstract_only_count = sum(1 for item in notes if item.get("evidence_level") == "abstract_only")
    metadata_only_count = sum(
        1
        for item in notes
        if item.get("evidence_level") in {"metadata_only", "downloadable_metadata", "local_pdf_without_text"}
    )
    manual_items = [
        {"paper_id": item.get("paper_id"), "title": item.get("title"), "note": item.get("note")}
        for item in notes
        if item.get("manual_upload_needed")
    ]
    return {
        "paper_count": len(notes),
        "local_fulltext_count": local_fulltext_count,
        "abstract_only_count": abstract_only_count,
        "metadata_only_count": metadata_only_count,
        "manual_upload_needed_count": len(manual_items),
        "manual_upload_needed": manual_items[:12],
    }


def _paper_fulltext_tip(paper: Paper) -> str:
    if paper.fulltext_status == "extracted":
        return "写作时会优先使用本地全文文本。"
    if paper.local_pdf_path:
        return "PDF 已保存，本地写作前会尝试抽取全文文本。"
    if paper.fulltext_status == "failed":
        return "自动下载或解析失败，请手动上传该论文 PDF 补全。"
    if paper.pdf_url:
        return "可先下载 PDF 到个人文献库，再生成更可靠的写作草稿。"
    if paper.abstract:
        return "当前只有摘要；详细方法、实验和局限建议上传原文后再写。"
    return "当前只有元数据；请手动上传原文补全。"


def _assistant_state_to_dict(state: AssistantState) -> dict:
    return {
        "papers": [paper.to_dict() for paper in state.papers],
        "last_intent": state.last_intent.to_dict() if state.last_intent else None,
        "last_answer": state.last_answer,
        "generated_files": state.generated_files,
        "last_generated_document": state.last_generated_document,
        "last_debug": state.last_debug,
        "last_tool_plan": state.last_tool_plan,
        "conversation_history": state.conversation_history,
        "evidence_paper_ids": state.evidence_paper_ids,
        "research_topics": state.research_topics,
        "uploaded_documents": [_uploaded_document_to_dict(document) for document in state.uploaded_documents],
    }


def _assistant_state_from_dict(data: dict) -> AssistantState:
    state = AssistantState()
    state.papers = [Paper.from_dict(item) for item in data.get("papers", []) if isinstance(item, dict)]
    intent_data = data.get("last_intent")
    if isinstance(intent_data, dict):
        allowed = ResearchIntent.__dataclass_fields__.keys()
        clean_intent = {key: value for key, value in intent_data.items() if key in allowed}
        state.last_intent = ResearchIntent(**clean_intent)
    state.last_answer = str(data.get("last_answer") or "")
    state.generated_files = _list_of_dicts(data.get("generated_files"))
    last_generated_document = data.get("last_generated_document")
    state.last_generated_document = last_generated_document if isinstance(last_generated_document, dict) else None
    last_debug = data.get("last_debug")
    state.last_debug = last_debug if isinstance(last_debug, dict) else {}
    state.last_tool_plan = [str(item) for item in data.get("last_tool_plan", []) if item]
    state.conversation_history = _list_of_dicts(data.get("conversation_history"))[-28:]
    state.evidence_paper_ids = [str(item) for item in data.get("evidence_paper_ids", []) if item]
    state.research_topics = [str(item) for item in data.get("research_topics", []) if item][:12]
    state.uploaded_documents = [
        _uploaded_document_from_dict(item)
        for item in data.get("uploaded_documents", [])
        if isinstance(item, dict)
    ]
    return state


def _uploaded_document_to_dict(document: UploadedDocument) -> dict:
    return {
        "id": document.id,
        "name": document.name,
        "path": document.path,
        "page_count": document.page_count,
        "char_count": document.char_count,
        "chunks": [asdict(chunk) for chunk in document.chunks],
        "uploaded_at": document.uploaded_at,
        "full_reading_notes": document.full_reading_notes,
        "full_reading_analysis": document.full_reading_analysis,
    }


def _uploaded_document_from_dict(data: dict) -> UploadedDocument:
    chunks = [
        PdfChunk(
            text=str(item.get("text") or ""),
            page=int(item.get("page") or 0),
            index=int(item.get("index") or index),
        )
        for index, item in enumerate(data.get("chunks", []))
        if isinstance(item, dict)
    ]
    return UploadedDocument(
        id=str(data.get("id") or uuid4().hex),
        name=str(data.get("name") or "uploaded-paper.pdf"),
        path=str(data.get("path") or ""),
        page_count=int(data.get("page_count") or 0),
        char_count=int(data.get("char_count") or sum(len(chunk.text) for chunk in chunks)),
        chunks=chunks,
        uploaded_at=str(data.get("uploaded_at") or utc_now_iso()),
        full_reading_notes=[str(item) for item in data.get("full_reading_notes", []) if item],
        full_reading_analysis=str(data.get("full_reading_analysis") or ""),
    )


def _list_of_dicts(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
