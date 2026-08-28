from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from urllib.parse import quote_plus

from ..tools.llm import KimiClient
from .models import normalize_text
from .routing import RoutingContext, compile_route


# Routing only selects an existing deterministic plan. The language model owns
# semantic interpretation; an unavailable model safely falls back to chat.
ROUTING_LLM_TIMEOUT = 12


@dataclass
class ResearchIntent:
    original_request: str
    normalized_topic: str
    cs_area: str
    # English canonical form drives retrieval; the short Chinese label is only
    # for the workspace and discovery UI.
    display_topic: str = ""
    keywords: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    source_queries: dict[str, list[str]] = field(default_factory=dict)
    source_urls: dict[str, list[str]] = field(default_factory=dict)
    target_venues: list[str] = field(default_factory=list)
    target_venue_ranks: list[str] = field(default_factory=list)
    recent_years: int | None = None
    from_year: int | None = None
    to_year: int | None = None
    constraints: list[str] = field(default_factory=lambda: ["arxiv", "ccf_a_b", "sci_q1_q3"])
    confidence: float = 0.0
    source: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActionIntent:
    original_request: str
    action: str
    reason: str = ""
    confidence: float = 0.0
    source: str = "unknown"


@dataclass
class RequestAnalysis:
    action: ActionIntent
    research: ResearchIntent | None = None
    tools: list[str] = field(default_factory=list)
    category: str = ""
    subtask: str = ""
    deliverable: str = ""
    evidence_scope: str = "none"
    search_required: bool = False


class IntentAnalyzer:
    def __init__(self, llm: KimiClient | None = None) -> None:
        self.llm = llm or KimiClient()
        self.last_action_trace: dict = {}
        self.last_research_trace: dict = {}

    def analyze_request(
        self,
        request: str,
        *,
        mode: str = "auto",
        has_papers: bool = False,
        has_documents: bool = False,
        has_generated_document: bool = False,
        conversation_context: str = "",
    ) -> RequestAnalysis:
        """Classify request semantics with Kimi, then compile a safe local tool plan."""
        if not self.llm.available:
            self.last_action_trace = {"requested": "kimi", "used": "unavailable", "fallback": True, "error": "Kimi API key is not configured."}
            self.last_research_trace = dict(self.last_action_trace)
            return self._safe_chat_analysis("Kimi 路由不可用，未执行关键词或正则兜底判断。", request)
        try:
            analysis = self._analyze_request_with_kimi(
                request,
                mode=mode,
                has_documents=has_documents,
                has_papers=has_papers,
                has_generated_document=has_generated_document,
                conversation_context=conversation_context,
            )
            self.last_action_trace = {"requested": "kimi", "used": "kimi", "fallback": False, "error": ""}
            self.last_research_trace = {"requested": "kimi", "used": "kimi", "fallback": False, "error": ""}
            return analysis
        except Exception as exc:
            self.last_action_trace = {"requested": "kimi", "used": "safe_chat", "fallback": True, "error": str(exc)}
            self.last_research_trace = dict(self.last_action_trace)
            return self._safe_chat_analysis("Kimi 路由未返回有效结构化结果，已安全保留自由问答入口。", request)

    def analyze(self, request: str) -> ResearchIntent:
        if self.llm.available:
            try:
                intent = self._analyze_with_kimi(request)
                self.last_research_trace = {"requested": "kimi", "used": "kimi", "fallback": False, "error": ""}
                return intent
            except Exception as exc:
                self.last_research_trace = {"requested": "kimi", "used": "safe_default", "fallback": True, "error": str(exc)}
                return self._safe_research_intent(request)
        self.last_research_trace = {"requested": "kimi", "used": "unavailable", "fallback": True, "error": "Kimi API key is not configured."}
        return self._safe_research_intent(request)

    def research_from_plan(self, request: str, plan: dict, *, source: str = "agent") -> ResearchIntent:
        """Convert a validated Agent tool payload into the existing search intent contract.

        The iterative Agent already used Kimi to plan this payload, so this method
        intentionally performs no additional model call. Missing optional fields
        are filled from the local parser only to keep search executable.
        """
        data = dict(plan or {})
        proposed_topic = normalize_text(str(data.get("normalized_topic") or data.get("topic") or ""))
        # A planner occasionally echoed the entire request into this field.
        # Repair only that unambiguous malformed shape with Kimi; Python does
        # not try to infer a topic from the user's wording.
        if _is_request_echo(proposed_topic, request):
            repaired = self.repair_echoed_topic(request, proposed_topic)
            if repaired:
                data.update(repaired)
                proposed_topic = normalize_text(str(data.get("normalized_topic") or ""))
            else:
                raise ValueError("paper_search 的主题字段回显了完整用户请求，未获得可用的规范研究主题。")
        if not proposed_topic:
            raise ValueError("paper_search 缺少规范化研究主题；请重新提供 normalized_topic。")
        data["normalized_topic"] = proposed_topic
        if not data.get("queries"):
            data["queries"] = [proposed_topic]
        if not data.get("keywords"):
            data["keywords"] = []
        if not data.get("source_queries"):
            data["source_queries"] = {}
        if not data.get("cs_area"):
            data["cs_area"] = "Interdisciplinary CS"
        intent = self._research_intent_from_data(request, data, source=source)
        self.last_research_trace = {"requested": "agent", "used": source, "fallback": False, "error": ""}
        return intent

    def repair_echoed_topic(self, request: str, proposed_topic: str = "") -> dict:
        """Ask Kimi to repair an unambiguous instruction-as-topic mistake.

        This is deliberately a narrow data-quality repair, not a keyword or
        regular-expression intent fallback. It also lets old persisted sessions
        recover their discovery profile without asking users to create a new one.
        """
        if not self.llm.available:
            return {}
        system = (
            "You normalize the research subject for a computer-science literature assistant. "
            "Return only valid JSON. The proposed topic is an invalid echo of a whole user instruction. "
            "Extract only the academic research object; do not include action verbs, time ranges, venue constraints, "
            "words such as research/investigate/progress/latest, or output deliverables. "
            "Keep essential technical modifiers."
        )
        user = f"""
User request: {request}
Invalid echoed topic: {proposed_topic or request}

Return exactly:
{{
  "normalized_topic": "concise English canonical research topic",
  "display_topic": "concise Chinese research topic for the UI",
  "keywords": ["4-8 English technical keywords"],
  "queries": ["2-4 focused English literature queries"],
  "cs_area": "one CS area"
}}

Example: "调研一下最近三年去偏推荐系统工作进展" ->
{{"normalized_topic":"debiasing recommender systems","display_topic":"去偏推荐系统","keywords":["debiasing","recommendation","recommender systems"],"queries":["debiasing recommender systems","debiasing recommendation"],"cs_area":"AI"}}
"""
        try:
            data = self.llm.chat_json(
                system=system,
                user=user,
                temperature=0.1,
                max_tokens=500,
                timeout=ROUTING_LLM_TIMEOUT,
                stream=False,
                label="topic_repair",
            )
        except Exception:
            return {}
        normalized_topic = normalize_text(str(data.get("normalized_topic") or ""))
        if not normalized_topic or _is_request_echo(normalized_topic, request):
            return {}
        return {
            "normalized_topic": normalized_topic,
            "display_topic": normalize_text(str(data.get("display_topic") or "")),
            "keywords": _clean_list(data.get("keywords")),
            "queries": _clean_list(data.get("queries")),
            "cs_area": normalize_text(str(data.get("cs_area") or "")),
        }

    def analyze_action(self, request: str, *, mode: str = "auto", has_papers: bool = False) -> ActionIntent:
        if mode == "chat":
            self.last_action_trace = {"requested": "mode", "used": "mode", "fallback": False, "error": ""}
            return ActionIntent(request, "chat", "用户选择自由聊天模式。", 1.0, "mode")
        if mode == "research":
            self.last_action_trace = {"requested": "mode", "used": "mode", "fallback": False, "error": ""}
            return ActionIntent(request, "search", "用户选择论文调研模式。", 1.0, "mode")
        if self.llm.available:
            try:
                action = self._analyze_action_with_kimi(request, has_papers=has_papers)
                self.last_action_trace = {"requested": "kimi", "used": "kimi", "fallback": False, "error": ""}
                return action
            except Exception as exc:
                self.last_action_trace = {"requested": "kimi", "used": "safe_chat", "fallback": True, "error": str(exc)}
                return ActionIntent(request, "chat", "Kimi 路由失败，安全保留自由问答。", 0.0, "safe_chat")
        self.last_action_trace = {"requested": "kimi", "used": "unavailable", "fallback": True, "error": "Kimi API key is not configured."}
        return ActionIntent(request, "chat", "Kimi 路由不可用，安全保留自由问答。", 0.0, "safe_chat")

    def _analyze_action_with_kimi(self, request: str, *, has_papers: bool) -> ActionIntent:
        system = (
            "You are the intent router for a CS paper research assistant. "
            "Return only valid JSON. Do not answer the user. "
            "Choose exactly one action from: chat, search, answer, document. "
            "chat means casual conversation or general help not requiring the current paper pool. "
            "search means the user explicitly asks to find, search, investigate, refresh, change topic, change venue, or change year range. "
            "answer means the user asks a follow-up question, asks why something happened, asks to inspect current results, or can be answered from the current retrieved paper pool. "
            "document means the user asks to generate/export/write a survey, paper section, abstract, introduction, method, experiment, discussion, related work, summary, report, bibtex, markdown, or other files. "
            "If papers already exist and the user asks a short follow-up like 'related work呢', '怎么没有生成', '为什么没有', choose answer unless they explicitly ask to regenerate. "
            "Do not treat output words like bib, file, markdown, document as research topics."
        )
        user = f"""
User request:
{request}

Current conversation has retrieved papers: {has_papers}

Return JSON:
{{
  "action": "chat|search|answer|document",
  "reason": "short Chinese reason",
  "confidence": 0.0
}}

Routing examples:
- "查一下近两年ICLR上关于动态推荐系统的论文，形成一份.bib文件" -> document
- "基于这些论文写一段引言" -> document
- "生成一份综述大纲" -> document
- "换成目标检测方向重新查" -> search
- "related work呢，怎么没有生成" -> answer, if has_papers is true
- "为什么没有生成bib" -> answer, if has_papers is true
- "这些论文里哪篇最适合先读" -> answer, if has_papers is true
- "你好，解释一下你能做什么" -> chat
"""
        data = self.llm.chat_json(
            system=system,
            user=user,
            temperature=0.1,
            max_tokens=500,
            timeout=ROUTING_LLM_TIMEOUT,
            label="intent_action",
        )
        action = normalize_text(str(data.get("action") or "")).lower()
        if action not in {"chat", "search", "answer", "document"}:
            action = "chat"
        return ActionIntent(
            original_request=request,
            action=action,
            reason=normalize_text(str(data.get("reason") or "")),
            confidence=_safe_float(data.get("confidence"), default=0.7),
            source="kimi",
        )

    def _analyze_request_with_kimi(
        self,
        request: str,
        *,
        mode: str,
        has_documents: bool,
        has_papers: bool,
        has_generated_document: bool,
        conversation_context: str,
    ) -> RequestAnalysis:
        system = (
            "You are the hierarchical routing and research-intent analyzer for a CS literature assistant. "
            "Return only one valid JSON object. Do not answer the user and never invent papers. "
            "Analyze in two stages: first classify the request category, then identify the concrete subtask and deliverable. "
            "The application, not you, chooses executable tools. Extract a clean research topic only if a literature search is actually needed. "
            "Use the supplied conversation and workspace state as first-class context. Do not force a request into a fixed keyword class. "
            "Infer exact CS topics while preserving modifiers such as dynamic, sequential, debiasing, causal, privacy, "
            "multimodal, open-vocabulary, small-object, and medical. Deliverable words such as survey, review, related work, "
            "bib, file, markdown, export, section, chapter, method introduction, and document are not research topics."
        )
        user = f"""
User request:
{request}

Mode: {mode}
Current session has uploaded PDF: {has_documents}
Current session has retrieved papers: {has_papers}
Current session has a generated Markdown document: {has_generated_document}
Conversation and workspace context (may be empty):
{conversation_context or 'None'}

Return JSON with exactly these keys:
- category: one of "chat", "literature_search", "evidence_qa", "paper_reading", "document_writing", "document_inspection", "discovery"
- subtask: concise subtask such as "fresh_search", "current_evidence_survey", "related_work", "bibtex", "uploaded_pdf_summary", "generated_file_check"
- deliverable: one of "none", "survey", "related_work", "introduction", "method_section", "experiment_section", "summary", "bibtex", "markdown", "report", "answer", "outline"
- evidence_scope: one of "none", "current_evidence", "uploaded_pdf", "mixed", "fresh_search", "generated_document"
- tool_plan: a non-empty ordered list using only "free_chat", "paper_search", "pdf_read", "paper_fulltext_read", "evidence_answer", "write_document", "document_inspect"
- needs_fresh_literature: true only when the tool_plan contains paper_search; otherwise false.
- topic_relation: one of "current_workspace", "new_topic", "follow_up", or "unknown".
- reason: short Chinese reason
- confidence: 0 to 1
- normalized_topic: concise English topic
- display_topic: concise Chinese UI label for the same research topic
- cs_area: NLP, AI, ML, CV, DB, SE, Security, Systems, Networks, HCI, Graphics, Theory, Robotics, or Interdisciplinary CS
- keywords: 4-8 English keywords
- queries: 2-4 precise English queries
- source_queries: object with openalex, dblp, arxiv, semantic_scholar, google_scholar; each has 1-3 focused English queries
- target_venues: explicitly requested venue names only, such as ACL, ICLR, KDD, T-PAMI
- target_venue_ranks: only CCF-A, CCF-B, SCI-Q1-Q3, or arXiv when explicitly requested
- recent_years, from_year, to_year: integer or null

Rules:
- Stage 1 categories:
  - chat: ordinary conversation, no evidence needed.
  - literature_search: user asks to find/search/retrieve/investigate new papers or change topic/year/venue/source.
  - evidence_qa: user asks about current retrieved papers or why a previous result looked wrong.
  - paper_reading: user asks to parse/read/summarize uploaded PDF or a specific retrieved paper.
  - document_writing: user asks to write/export/generate a survey, related work, section, chapter, report, BibTeX, markdown, or summary.
  - document_inspection: user asks where the generated file is, whether it is empty, or what it contains.
  - discovery: user asks for latest trends/news/research progress recommendations.
- Tool-plan contract:
  - Ordinary greeting, brainstorming, coding-independent help, or open conversation -> category chat, evidence_scope none, tool_plan ["free_chat"]. This is the permanent free-chat entry; do not turn "hello" into evidence QA or search merely because papers exist.
  - Fresh literature retrieval -> literature_search, fresh_search, ["paper_search"].
  - Fresh retrieval followed by a requested document -> document_writing, fresh_search, ["paper_search", "write_document"].
  - Question answerable from current papers -> evidence_qa, current_evidence, ["evidence_answer"].
  - Uploaded PDF explanation -> paper_reading, uploaded_pdf, ["pdf_read"].
  - Uploaded PDF plus a requested document -> document_writing, uploaded_pdf, ["pdf_read", "write_document"].
  - Current evidence writing -> document_writing, current_evidence, ["write_document"].
  - Inspecting the last generated output -> document_inspection, generated_document, ["document_inspect"].
  - A detailed question about a cached paper's method or experiment may use ["paper_fulltext_read"] when no uploaded PDF is the target.
- Stage 2: decide whether new retrieval is needed. If the request says “按现在已调研的文献”, “基于已有论文”, “基于这些论文”, “根据当前证据”, or similar, set needs_fresh_literature to false.
- Treat “目前文献库/现有文献库/我的文献库/已选论文/当前论文池” as current workspace references. For example, “我想基于目前文献库写一篇调研” is document_writing using current_workspace, not a new literature search. The word “调研” may describe a report or survey deliverable; it is not by itself proof that external search is needed.
- If an existing paper pool is about one topic but the user clearly names a different new academic topic, set topic_relation to new_topic and needs_fresh_literature to true. If the request only says “写一篇调研/综述” without naming a new topic, use current_workspace.
- Stage 3: extract research_topic only from the academic object. Never include deliverables or instructions in normalized_topic or queries.
- The conversation context is authoritative for references such as “这个”, “上一份”, “为什么是空的”, and “继续”. Do not start a new search just because the newest user message is short.
- When an uploaded PDF exists, requests such as “解析一下这篇文章”, “总结本文”, “讲讲方法”, or “实验结果如何” are paper_reading, not literature_search.
- If a generated document exists and the user asks whether it is empty, asks for its content, asks what it generated, or asks why it failed, choose document_inspection. Choose document_writing only when the user explicitly asks to rewrite, expand, or regenerate it.
- If the user asks an ordinary follow-up about current papers, choose evidence_qa; do not set needs_fresh_literature unless the user explicitly asks for new, different, refreshed, or broader literature.
- If current retrieved papers exist and the user asks to write a survey/综述/章节/related work from current or already-investigated literature, choose document_writing and set needs_fresh_literature to false.
- "近两年 ICLR 上动态推荐系统论文并生成 bib" means category document_writing, normalized_topic "dynamic recommender systems", target_venues ["ICLR"], recent_years 2. Never put bib in a query.
- "调研一下最近三年去偏推荐系统工作进展" means normalized_topic "debiasing recommender systems" and display_topic "去偏推荐系统". The full Chinese instruction must never be a topic.
- "按现在已调研的文献，写一篇survey综述方法介绍章节" means category document_writing, subtask current_evidence_survey, deliverable survey, needs_fresh_literature false, normalized_topic should reuse context or be empty. Never search for "survey".
- "A会和B会的目标检测" means target_venue_ranks ["CCF-A", "CCF-B"], not literal venue names.
- For dynamic recommendation include a short query such as "dynamic rec" as one variant, but do not drop the word dynamic.
- If the user asks to parse or explain an uploaded PDF and also generate related work, choose document_writing and set needs_fresh_literature false unless fresh external literature is explicitly requested.
"""
        raw_data = self.llm.chat_json(
            system=system,
            user=user,
            temperature=0.1,
            max_tokens=900,
            timeout=ROUTING_LLM_TIMEOUT,
            stream=False,
            label="intent_request",
        )
        data = raw_data if isinstance(raw_data, dict) else {}
        return self._compile_request_analysis(
            request,
            data=data,
            has_papers=has_papers,
            has_documents=has_documents,
            has_generated_document=has_generated_document,
            source="kimi",
        )

    def _compile_request_analysis(
        self,
        request: str,
        *,
        data: dict,
        has_papers: bool,
        has_documents: bool,
        has_generated_document: bool,
        source: str,
        fallback_reason: str = "",
    ) -> RequestAnalysis:
        """Turn semantic intent into one executable plan owned by the application."""
        decision = compile_route(
            context=RoutingContext(
                has_papers=has_papers,
                has_documents=has_documents,
                has_generated_document=has_generated_document,
            ),
            llm_category=str(data.get("category") or ""),
            llm_subtask=str(data.get("subtask") or ""),
            llm_deliverable=str(data.get("deliverable") or ""),
            llm_evidence_scope=str(data.get("evidence_scope") or ""),
            llm_tool_plan=data.get("tool_plan") if isinstance(data.get("tool_plan"), list) else data.get("tools"),
            llm_reason=str(data.get("reason") or fallback_reason),
        )
        action_name = {
            "document_writing": "document",
            "literature_search": "search",
            "paper_reading": "answer",
            "evidence_qa": "answer",
            "document_inspection": "answer",
            "discovery": "chat",
            "chat": "chat",
        }[decision.category]
        action = ActionIntent(
            request,
            action_name,
            decision.reason,
            _safe_float(data.get("confidence"), default=0.6 if source != "kimi" else 0.75),
            source,
        )
        research = self._research_intent_from_data(request, data, source=source) if decision.search_required else None
        return RequestAnalysis(
            action,
            research,
            list(decision.tools),
            category=decision.category,
            subtask=decision.subtask,
            deliverable=decision.deliverable,
            evidence_scope=decision.evidence_scope,
            search_required=decision.search_required,
        )

    @staticmethod
    def _safe_chat_analysis(reason: str, request: str = "") -> RequestAnalysis:
        action = ActionIntent(request, "chat", reason, 0.0, "safe_chat")
        return RequestAnalysis(
            action=action,
            research=None,
            tools=["free_chat"],
            category="chat",
            subtask="general_chat",
            deliverable="none",
            evidence_scope="none",
            search_required=False,
        )

    def _analyze_with_kimi(self, request: str) -> ResearchIntent:
        system = (
            "You are a computer-science research intent analyzer. "
            "Return only valid JSON. Do not invent papers. "
            "Convert the user's Chinese or English request into source-specific English literature search queries. "
            "Infer the exact research subfield from the user's text, not just the broad parent field. "
            "Preserve all important modifiers such as dynamic, sequential, session-based, debiasing, fairness, privacy, robustness, causal, multimodal, graph, medical, open-vocabulary, small-object, and security. "
            "Generate abbreviation and short-form variants when common in papers, such as rec, recommender, recommendation, RS, seq rec, dyn rec, LLM, RAG, GNN, VLM, MLLM. "
            "Do not simplify a specific topic into a generic one. "
            "Ignore output-format and task-delivery words such as survey, review, related work, chapter, section, method introduction, .bib, BibTeX, document, report, export, generate, write, and file; these are not research topics."
        )
        user = f"""
Analyze the user's research request for a CS literature investigation agent.

User request:
{request}

Return JSON with these keys:
- normalized_topic: concise English research topic
- display_topic: concise Chinese UI label for the same research topic
- cs_area: one of NLP, AI, ML, CV, DB, SE, Security, Systems, Networks, HCI, Graphics, Theory, Robotics, Interdisciplinary CS
- keywords: 6-12 English keywords
- queries: 4-8 precise English paper-search queries
- source_queries: object with keys "openalex", "dblp", "arxiv", "semantic_scholar", "google_scholar"; each value is a list of 3-6 source-specific queries.
  - OpenAlex queries should preserve compound facets and may use boolean operators for essential modifiers.
  - DBLP queries should be short and title/venue-friendly.
  - arXiv queries should include important technical qualifiers.
  - Google Scholar queries can be broader and include synonyms.
- target_venues: venue names explicitly requested by the user, such as ACL, ICML, T-PAMI, TKDE, KDD. Empty list if none.
- target_venue_ranks: venue-quality filters explicitly requested by the user. Allowed values: "CCF-A", "CCF-B", "SCI-Q1-Q3", "arXiv".
  - If the user says "A会", "CCF A", "A类会议", use ["CCF-A"], not target_venues.
  - If the user says "B会", "CCF B", "B类会议", use ["CCF-B"], not target_venues.
  - If the user says "A/B会", "A会和B会", use ["CCF-A", "CCF-B"].
- recent_years: integer if the user asks for recent N years, otherwise null
- from_year: integer if user specifies a start year, otherwise null
- to_year: integer if user specifies an end year, otherwise null
- confidence: number from 0 to 1

Examples of preserving the real field:
- "动态推荐系统" -> normalized_topic "dynamic recommender systems"; queries should include "dynamic recommendation", not only "recommender systems".
- "动态推荐系统" -> DBLP queries should include "dynamic rec", "dynamic recommendation", "dynamic recommender", and "dynamic recommender systems".
- "去偏推荐系统" -> normalized_topic "debiasing recommender systems"; queries should include "debiasing recommendation" and "fairness-aware recommender systems".
- "目标检测" -> normalized_topic "object detection".
- "开放词汇目标检测" -> normalized_topic "open-vocabulary object detection"; keep "open-vocabulary".
- "医学图像分割" -> normalized_topic "medical image segmentation"; keep "medical".
- "查一下近两年ICLR上关于动态推荐系统的论文，形成一份.bib文件" -> normalized_topic "dynamic recommender systems", target_venues ["ICLR"], recent_years 2. Do not include "bib" in any query.
- "按现在已调研的文献，写一篇survey综述方法介绍章节" -> this is not a search request. If forced to extract a topic, do not include "survey", "review", "method introduction", or "chapter".
- "近三年A会和B会的目标检测论文" -> target_venue_ranks ["CCF-A", "CCF-B"], target_venues []. Do not search for a literal venue named "A会" or "B会".

The agent will search DBLP, arXiv, and Google Scholar. Return JSON only.
"""
        data = self.llm.chat_json(
            system=system,
            user=user,
            temperature=0.1,
            timeout=ROUTING_LLM_TIMEOUT,
            label="intent_research",
        )
        return self._research_intent_from_data(request, data, source="kimi")

    def _research_intent_from_data(self, request: str, data: dict, *, source: str) -> ResearchIntent:
        normalized_topic = normalize_text(str(data.get("normalized_topic") or data.get("topic") or ""))
        queries = _clean_list(data.get("queries")) or ([normalized_topic] if normalized_topic else [])
        if not normalized_topic and queries:
            normalized_topic = queries[0]
        if not normalized_topic:
            # Never persist a complete user instruction as a research topic.
            normalized_topic = "computer science"
        keywords = _clean_list(data.get("keywords"))
        source_queries = _clean_source_queries(data.get("source_queries"), queries or [normalized_topic])
        recent_years = _safe_int(data.get("recent_years"))
        from_year = _safe_int(data.get("from_year"))
        to_year = _safe_int(data.get("to_year"))
        return ResearchIntent(
            original_request=request,
            normalized_topic=normalized_topic,
            display_topic=normalize_text(str(data.get("display_topic") or "")),
            cs_area=normalize_text(str(data.get("cs_area") or "Interdisciplinary CS")),
            keywords=keywords,
            queries=queries,
            source_queries=source_queries,
            source_urls=_source_urls(source_queries, recent_years=recent_years, from_year=from_year, to_year=to_year),
            target_venues=_clean_list(data.get("target_venues")),
            target_venue_ranks=_clean_venue_ranks(data.get("target_venue_ranks")),
            recent_years=recent_years,
            from_year=from_year,
            to_year=to_year,
            confidence=_safe_float(data.get("confidence"), default=0.7),
            source=source,
        )

    @staticmethod
    def _safe_research_intent(request: str) -> ResearchIntent:
        topic = "computer science"
        source_queries = _clean_source_queries({}, [topic])
        return ResearchIntent(
            original_request=request,
            normalized_topic=topic,
            display_topic="",
            cs_area="Interdisciplinary CS",
            queries=[topic],
            source_queries=source_queries,
            source_urls=_source_urls(source_queries, recent_years=None, from_year=None, to_year=None),
            confidence=0.0,
            source="safe_default",
        )

def _is_request_echo(topic: str, request: str) -> bool:
    """Detect only the exact malformed planner shape, without interpreting text."""
    normalized_topic = normalize_text(topic).casefold()
    normalized_request = normalize_text(request).casefold()
    return bool(normalized_topic and normalized_request and normalized_topic == normalized_request)


def _clean_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = [normalize_text(str(item)) for item in value]
    return [item for item in cleaned if item]


def _clean_source_queries(value, fallback_queries: list[str]) -> dict[str, list[str]]:
    source_queries = {"openalex": [], "dblp": [], "arxiv": [], "semantic_scholar": [], "google_scholar": []}
    if isinstance(value, dict):
        for source in source_queries:
            source_queries[source] = _clean_list(value.get(source))
    for source in source_queries:
        if not source_queries[source]:
            source_queries[source] = fallback_queries[:4]
    return source_queries


def _clean_venue_ranks(value) -> list[str]:
    allowed = {
        "ccfa": "CCF-A",
        "ccf-a": "CCF-A",
        "a": "CCF-A",
        "ccfb": "CCF-B",
        "ccf-b": "CCF-B",
        "b": "CCF-B",
        "sciq1q3": "SCI-Q1-Q3",
        "sci-q1-q3": "SCI-Q1-Q3",
        "arxiv": "arXiv",
    }
    ranks = []
    for item in _clean_list(value):
        key = "".join(character for character in item.lower() if character.isascii() and (character.isalnum() or character == "-"))
        rank = allowed.get(key)
        if rank and rank not in ranks:
            ranks.append(rank)
    return ranks


def _normalize_tool_plan_for_request(
    tools: list[str],
    request: str,
    *,
    has_papers: bool,
    has_documents: bool,
) -> list[str]:
    plan = list(tools)
    if _uses_current_evidence_for_writing(request, has_papers=has_papers):
        plan = [tool for tool in plan if tool != "paper_search"]
        if has_documents and _looks_like_pdf_read_request(request) and "pdf_read" not in plan:
            plan.insert(0, "pdf_read")
        if "write_document" not in plan:
            plan.append("write_document")
    if _looks_like_document_request(request) and has_papers and "paper_search" in plan and not _has_fresh_search_intent(request):
        plan = [tool for tool in plan if tool != "paper_search"]
        if "write_document" not in plan:
            plan.append("write_document")
    if not plan:
        if _looks_like_document_request(request) and (has_papers or has_documents):
            plan = ["write_document"]
        elif has_papers:
            plan = ["evidence_answer"]
        else:
            plan = ["free_chat"]
    return plan


def _heuristic_category(action: str, tools: list[str]) -> str:
    if "document_inspect" in tools:
        return "document_inspection"
    if "write_document" in tools:
        return "document_writing"
    if "pdf_read" in tools or "paper_fulltext_read" in tools:
        return "paper_reading"
    if "paper_search" in tools:
        return "literature_search"
    if "evidence_answer" in tools:
        return "evidence_qa"
    if action == "chat" or "free_chat" in tools:
        return "chat"
    return action or "chat"


def _uses_current_evidence_for_writing(request: str, *, has_papers: bool) -> bool:
    if not has_papers or not _looks_like_document_request(request):
        return False
    lowered = request.lower()
    compact = re.sub(r"\s+", "", lowered)
    current_markers = [
        "按现在",
        "已调研",
        "已经调研",
        "已有",
        "当前",
        "这些论文",
        "这些文献",
        "已选论文",
        "当前证据",
        "检索到的",
        "基于这些",
        "基于当前",
        "根据这些",
        "根据当前",
        "current",
        "retrieved",
        "existing",
    ]
    return any(marker in compact or marker in lowered for marker in current_markers)


def _has_fresh_search_intent(value: str) -> bool:
    lowered = value.lower()
    compact = re.sub(r"\s+", "", lowered)
    fresh_markers = [
        "重新检索",
        "重新搜索",
        "重新查",
        "再检索",
        "再搜索",
        "再查",
        "补充检索",
        "补充搜索",
        "最新论文",
        "新的论文",
        "外部文献",
        "fresh",
        "new papers",
        "refresh",
    ]
    if any(marker in compact or marker in lowered for marker in fresh_markers):
        return True
    explicit = ["检索", "搜索", "查找", "查一下", "搜一下", "find", "search", "retrieve"]
    return any(marker in lowered for marker in explicit)


def _sanitize_research_topic(value: str, request: str) -> str:
    text = normalize_text(str(value or ""))
    if not text:
        return ""
    text = _remove_deliverable_terms(text, aggressive=_looks_like_document_request(request))
    text = re.sub(r"\b(current|existing|retrieved|investigated|literature|papers?|article|articles)\b", " ", text, flags=re.I)
    text = re.sub(r"\b(write|generate|draft|export|form|create|based|according|introduction|chapter|section)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" -_,.;:，。")
    return text


def _prefer_rule_intent(request: str, topic: str) -> bool:
    """Prefer deterministic parsing when a request contains a known concrete topic."""
    compact_request = re.sub(r"\s+", "", request or "")
    compact_topic = re.sub(r"\s+", "", topic or "")
    known_topics = (
        "去偏推荐", "消偏推荐", "公平推荐", "无偏推荐", "推荐去偏",
        "动态推荐", "序列推荐", "会话推荐", "目标检测", "图异常检测",
        "医学图像分割", "开放词汇目标检测",
    )
    request_echo_markers = ("调研", "工作进展", "最近", "近三年", "查一下", "搜一下", "文献库", "写一篇")
    return any(marker in compact_request for marker in known_topics) or any(
        marker in compact_topic for marker in request_echo_markers
    )


def _sanitize_queries(values: list[str], request: str) -> list[str]:
    cleaned = []
    for value in values:
        query = _sanitize_research_topic(value, request)
        if query and query.lower() not in {item.lower() for item in cleaned}:
            cleaned.append(query)
    return cleaned or [_sanitize_research_topic(normalize_text(request), request) or "computer science"]


def _sanitize_keywords(values: list[str], request: str) -> list[str]:
    cleaned = []
    for value in values:
        keyword = _remove_deliverable_terms(value, aggressive=_looks_like_document_request(request)).strip()
        if keyword and keyword.lower() not in _TASK_TERM_SET and keyword.lower() not in {item.lower() for item in cleaned}:
            cleaned.append(keyword)
    return cleaned


def _sanitize_source_queries(source_queries: dict[str, list[str]], request: str) -> dict[str, list[str]]:
    return {source: _sanitize_queries(queries, request) for source, queries in source_queries.items()}


_TASK_TERM_SET = {
    "survey", "review", "综述", "related", "work", "related work", "method", "methods",
    "introduction", "chapter", "section", "bib", "bibtex", "markdown", "document",
    "report", "file", "summary", "write", "generate", "export",
}


def _remove_deliverable_terms(value: str, *, aggressive: bool) -> str:
    text = normalize_text(str(value or ""))
    phrase_patterns = [
        r"\brelated\s+work\b",
        r"\bmethod\s+introduction\b",
        r"\bmethods?\s+introduction\b",
        r"\bsurvey\s+review\b",
        r"\bsurvey\s+section\b",
        r"\breview\s+section\b",
        r"\bbibtex\b",
        r"\bmarkdown\b",
        r"\bchapter\b",
        r"\bsection\b",
        r"方法介绍章节",
        r"方法介绍",
        r"介绍章节",
        r"综述章节",
        r"文献综述",
        r"相关工作",
        r"引用文件",
    ]
    for pattern in phrase_patterns:
        text = re.sub(pattern, " ", text, flags=re.I)
    if aggressive:
        words = [
            "survey", "review", "method", "methods", "introduction", "chapter", "section",
            "summary", "report", "document", "file", "bib", "write", "generate", "draft",
            "综述", "章节", "方法", "介绍", "文档", "文件", "写", "生成", "导出",
        ]
        for word in words:
            text = re.sub(rf"\b{re.escape(word)}\b", " ", text, flags=re.I)
            text = text.replace(word, " ")
    return re.sub(r"\s+", " ", text).strip()


def _safe_int(value) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(number, 1.0))


def _safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return normalize_text(str(value or "")).lower() in {"1", "true", "yes", "是"}


def _looks_like_pdf_read_request(value: str) -> bool:
    lowered = value.lower()
    markers = ("解析", "解读", "总结", "概括", "分析", "这篇", "本文", "文章", "论文", "pdf", "方法", "实验")
    return any(marker in lowered for marker in markers)


def _contextual_follow_up_action(value: str, *, has_papers: bool) -> ActionIntent | None:
    if not has_papers:
        return None
    lowered = value.lower().strip()
    compact = re.sub(r"\s+", "", lowered)
    if _has_explicit_search_intent(value):
        return None
    complaint_markers = [
        "怎么没有",
        "为什么没有",
        "没生成",
        "没有生成",
        "没看到",
        "看不到",
        "在哪里",
        "在哪",
        "文件呢",
        "bib呢",
        "relatedwork呢",
        "related work呢",
    ]
    if any(marker in compact or marker in lowered for marker in complaint_markers):
        return ActionIntent(value, "answer", "检测到基于当前上下文的连续追问。", 0.76)
    if re.search(r"(生成|写|撰写|导出|形成|整理|起草)", value) and not re.search(r"(没有|没|为什么|怎么)", value):
        return None
    follow_up_markers = [
        "呢",
        "什么意思",
        "解释",
        "上一轮",
        "刚才",
        "这些",
        "当前",
        "结果",
        "引用文件",
    ]
    if any(marker in compact or marker in lowered for marker in follow_up_markers):
        return ActionIntent(value, "answer", "检测到基于当前上下文的连续追问。", 0.72)
    if len(value) <= 28 and any(token in lowered for token in ["related work", "bib", "引用", "文档", "文件"]):
        return ActionIntent(value, "answer", "短追问优先基于当前结果回答。", 0.65)
    return None


def _has_explicit_search_intent(value: str) -> bool:
    lowered = value.lower()
    explicit_words = [
        "检索",
        "搜索",
        "查一下",
        "搜一下",
        "查询",
        "查找",
        "调研",
        "重新查",
        "重搜",
        "再搜",
        "换成",
        "改成",
        "找",
        "find",
        "search",
        "retrieve",
        "look up",
        "survey papers",
    ]
    has_time_or_venue = bool(re.search(r"近\s*[\d一二两三四五六七八九十]+\s*年|20\d{2}|aaai|iclr|icml|kdd|acl|emnlp|cvpr|iccv|eccv", lowered, flags=re.I))
    return any(word in lowered for word in explicit_words) or has_time_or_venue


def _looks_like_document_request(value: str) -> bool:
    lowered = value.lower()
    document_words = [
        "生成文档",
        "写文档",
        "写一段",
        "写一节",
        "写章节",
        "章节",
        "论文章节",
        "摘要",
        "引言",
        "背景",
        "方法",
        "实验",
        "讨论",
        "总结",
        "论文总结",
        "科研方向",
        "文档",
        "导出",
        "报告",
        "综述",
        "related work",
        "bibtex",
        ".bib",
        "bib",
        "文献综述",
        "markdown",
    ]
    return any(word in lowered for word in document_words)


def _requires_multi_tool_plan(value: str, *, has_documents: bool) -> bool:
    if not has_documents:
        return False
    lowered = value.lower()
    pdf_read_words = ["pdf", "上传", "这篇论文", "本文", "解析", "解读", "总结", "概括", "分析"]
    return _looks_like_document_request(value) and any(word in lowered for word in pdf_read_words)


def _translate_common_terms(value: str) -> str:
    replacements = {
        "大模型": "large language model",
        "大型语言模型": "large language model",
        "检索增强": "retrieval augmented generation",
        "知识增强": "knowledge augmented",
        "去偏动态推荐系统": "debiasing dynamic recommender systems",
        "去偏动态推荐": "debiasing dynamic recommendation",
        "公平动态推荐系统": "fairness-aware dynamic recommender systems",
        "公平动态推荐": "fairness-aware dynamic recommendation",
        "无偏动态推荐系统": "unbiased dynamic recommender systems",
        "无偏动态推荐": "unbiased dynamic recommendation",
        "动态推荐系统": "dynamic recommender systems",
        "动态推荐": "dynamic recommendation",
        "序列推荐系统": "sequential recommender systems",
        "序列推荐": "sequential recommendation",
        "会话推荐系统": "session-based recommender systems",
        "会话推荐": "session-based recommendation",
        "上下文感知推荐": "context-aware recommendation",
        "去偏推荐系统": "debiasing recommender systems",
        "去偏推荐": "debiasing recommendation",
        "消偏推荐系统": "debiasing recommender systems",
        "消偏推荐": "debiasing recommendation",
        "公平推荐系统": "fairness-aware recommender systems",
        "公平推荐": "fairness-aware recommendation",
        "无偏推荐": "unbiased recommender systems",
        "推荐去偏": "recommendation debiasing",
        "智能体": "agent",
        "多智能体": "multi-agent",
        "问答": "question answering",
        "代码生成": "code generation",
        "漏洞检测": "vulnerability detection",
        "软件工程": "software engineering",
        "数据库": "database",
        "图神经网络": "graph neural network",
        "图异常检测": "graph anomaly detection",
        "异常检测": "anomaly detection",
        "计算机视觉": "computer vision",
        "开放词汇目标检测": "open-vocabulary object detection",
        "小目标检测": "small object detection",
        "目标检测": "object detection",
        "实例分割": "instance segmentation",
        "语义分割": "semantic segmentation",
        "医学图像分割": "medical image segmentation",
        "医学图像": "medical image",
        "多模态": "multimodal",
        "医学": "medical",
        "推荐系统": "recommender system",
        "网络安全": "cybersecurity",
        "隐私": "privacy",
        "联邦学习": "federated learning",
        "强化学习": "reinforcement learning",
    }
    normalized = value
    for src, dst in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = normalized.replace(src, f" {dst} ")
    return normalized


def _keyword_tokens(value: str) -> list[str]:
    stopwords = {
        "bib",
        "bibtex",
        "bibliography",
        "reference",
        "references",
        "document",
        "report",
        "file",
        "export",
        "write",
        "generate",
        "format",
        "want",
        "research",
        "paper",
        "papers",
        "survey",
        "related",
        "work",
        "direction",
        "latest",
        "recent",
        "about",
        "using",
        "based",
        "with",
        "and",
        "the",
        "for",
        "想",
        "研究",
        "调研",
        "论文",
        "方向",
        "最新",
        "近年",
        "会议",
        "期刊",
        "形成",
        "一份",
        "文件",
        "文档",
        "生成",
        "导出",
        "写",
    }
    tokens = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", value.lower()):
        if len(token) <= 2 or token in stopwords:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens


def _extract_recent_years(value: str) -> int | None:
    patterns = [
        r"近\s*(\d{1,2})\s*年",
        r"最近\s*(\d{1,2})\s*年",
        r"past\s*(\d{1,2})\s*years?",
        r"last\s*(\d{1,2})\s*years?",
        r"recent\s*(\d{1,2})\s*years?",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.I)
        if match:
            return int(match.group(1))
    chinese_numbers = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    match = re.search(r"(近|最近)\s*([一二两三四五六七八九十])\s*年", value)
    if match:
        return chinese_numbers.get(match.group(2))
    return None


def _extract_year_range(value: str) -> tuple[int | None, int | None]:
    range_match = re.search(r"\b((?:19|20)\d{2})\s*[-~至到]\s*((?:19|20)\d{2})\b", value)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))
    since_match = re.search(r"\b((?:19|20)\d{2})\s*(?:以来|以后|之后|onward|since)\b", value, flags=re.I)
    if since_match:
        return int(since_match.group(1)), None
    return None, None


def _extract_target_venues(value: str) -> list[str]:
    normalized = value.lower()
    found: list[str] = []
    for venue in KNOWN_VENUES:
        pattern = r"(?<![a-z0-9])" + re.escape(venue.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            found.append(venue)
    return found


def _extract_target_venue_ranks(value: str) -> list[str]:
    text = value.lower()
    ranks: list[str] = []
    if re.search(r"(ccf\s*)?a\s*(类)?\s*(会|会议)|a/b\s*(会|会议)|a\s*和\s*b\s*(会|会议)|a\s*、\s*b\s*(会|会议)", text):
        ranks.append("CCF-A")
    if re.search(r"(ccf\s*)?b\s*(类)?\s*(会|会议)|a/b\s*(会|会议)|a\s*和\s*b\s*(会|会议)|a\s*、\s*b\s*(会|会议)", text):
        ranks.append("CCF-B")
    if re.search(r"sci\s*(q?1\s*[-到至~]?\s*q?3|三区以上|1区到3区|一区到三区)", text):
        ranks.append("SCI-Q1-Q3")
    return ranks


def _guess_area(value: str) -> str:
    text = value.lower()
    area_rules = [
        ("Security", ["security", "vulnerability", "privacy", "attack", "malware"]),
        ("SE", ["software", "code", "program", "bug", "repository"]),
        ("NLP", ["language model", "nlp", "text", "question answering", "rag"]),
        ("CV", ["vision", "image", "video", "detection", "segmentation"]),
        ("DB", ["database", "query", "transaction", "index"]),
        ("AI", ["recommender", "recommendation", "debiasing", "fairness-aware"]),
        ("CV", ["object detection", "segmentation", "open-vocabulary", "small object", "image"]),
        ("Systems", ["system", "os", "distributed", "storage", "compiler"]),
        ("Networks", ["network", "wireless", "routing", "internet"]),
        ("HCI", ["human", "interaction", "user interface"]),
        ("ML", ["learning", "neural", "model", "training"]),
    ]
    for area, needles in area_rules:
        if any(needle in text for needle in needles):
            return area
    return "Interdisciplinary CS"


def _build_queries(topic: str, tokens: list[str]) -> list[str]:
    base = normalize_text(topic)
    queries = [base]
    if tokens:
        queries.append(" ".join(tokens[:6]))
    if "large" in tokens and "language" in tokens and "model" in tokens:
        queries.append(base.replace("large language model", "LLM"))
    if "retrieval" in tokens or "augmented" in tokens:
        queries.append(base.replace("retrieval augmented generation", "RAG"))
    if "dynamic" in tokens and ("recommender" in tokens or "recommendation" in tokens):
        queries.extend(
            [
                "dynamic rec",
                "dynamic recommender",
                "dynamic recommender systems",
                "dynamic recommendation",
                "time-aware recommender systems",
                "temporal recommendation",
                "sequential rec",
                "sequential dynamic recommendation",
            ]
        )
    if "debiasing" in tokens and ("recommender" in tokens or "recommendation" in tokens):
        queries.extend(
            [
                "debiasing recommender systems",
                "debiasing recommendation",
                "fairness-aware recommender systems",
                "unbiased recommendation",
                "bias mitigation in recommender systems",
            ]
        )
    if "object" in tokens and "detection" in tokens:
        if "open-vocabulary" in tokens:
            queries.extend(["open-vocabulary object detection", "open vocabulary detection"])
        elif "small" in tokens:
            queries.extend(["small object detection", "tiny object detection"])
        else:
            queries.extend(["object detection", "visual object detection"])
    return list(dict.fromkeys(query for query in queries if query))[:6]


def _build_source_queries(queries: list[str], tokens: list[str]) -> dict[str, list[str]]:
    source_queries = {
        "openalex": queries[:4],
        "dblp": queries[:4],
        "arxiv": queries[:4],
        "semantic_scholar": queries[:4],
        "google_scholar": queries[:4],
    }
    if "debiasing" in tokens and "dynamic" in tokens and ("recommender" in tokens or "recommendation" in tokens):
        source_queries = {
            "openalex": [
                '(debiasing OR fairness OR unbiased OR "bias mitigation") AND (dynamic OR temporal OR sequential OR "time-aware") AND (recommender OR recommendation)',
                "debiasing dynamic recommender systems",
                "fairness dynamic recommender systems",
                "unbiased dynamic recommendation",
            ],
            "dblp": [
                "fairness dynamic recommender systems",
                "fairness dynamic recommendation",
                "unbiased dynamic recommendation",
                "debiasing dynamic recommendation",
                "dynamic recommender fairness",
            ],
            "arxiv": [
                "debiasing dynamic recommender systems",
                "fairness-aware dynamic recommender systems",
                "unbiased dynamic recommendation",
                "bias mitigation dynamic recommendation",
            ],
            "semantic_scholar": [
                "debiasing dynamic recommender systems",
                "fairness dynamic recommender systems",
                "unbiased dynamic recommendation",
                "bias mitigation dynamic recommendation",
            ],
            "google_scholar": [
                '"debiasing" "dynamic recommender systems"',
                '"fairness" "dynamic recommender systems"',
                '"unbiased" "dynamic recommendation"',
                '"bias mitigation" "dynamic recommendation"',
            ],
        }
    elif "debiasing" in tokens and ("recommender" in tokens or "recommendation" in tokens):
        source_queries = {
            "openalex": [
                '(debiasing OR fairness OR unbiased OR "bias mitigation") AND (recommender OR recommendation)',
                "debiasing recommender systems",
                "fairness-aware recommender systems",
                "unbiased recommendation",
            ],
            "dblp": [
                "debiasing recommender systems",
                "debiasing recommendation",
                "fairness-aware recommendation",
                "unbiased recommendation",
            ],
            "arxiv": [
                "debiasing recommender systems",
                "bias mitigation recommender systems",
                "fairness-aware recommender systems",
                "unbiased recommendation",
            ],
            "semantic_scholar": [
                "debiasing recommender systems",
                "bias mitigation recommender systems",
                "fairness-aware recommendation",
                "unbiased recommendation",
            ],
            "google_scholar": [
                '"debiasing recommender systems"',
                '"debiasing recommendation"',
                '"fairness-aware recommender systems"',
                '"bias mitigation" "recommender systems"',
            ],
        }
    elif "dynamic" in tokens and ("recommender" in tokens or "recommendation" in tokens):
        source_queries = {
            "openalex": [
                '(dynamic OR temporal OR sequential OR "time-aware") AND (recommender OR recommendation)',
                "dynamic recommender systems",
                "dynamic recommendation",
                "temporal recommendation",
            ],
            "dblp": [
                "dynamic rec",
                "dynamic recommender",
                "dynamic recommender systems",
                "dynamic recommendation",
                "time-aware recommendation",
                "temporal recommendation",
                "sequential rec",
                "sequential recommendation",
            ],
            "arxiv": [
                "dynamic rec",
                "dynamic recommender",
                "dynamic recommender systems",
                "dynamic recommendation",
                "time-aware recommender systems",
                "temporal recommendation",
                "sequential dynamic recommendation",
            ],
            "semantic_scholar": [
                "dynamic rec",
                "dynamic recommender",
                "dynamic recommender systems",
                "dynamic recommendation",
                "time-aware recommender systems",
                "temporal recommendation",
                "sequential recommendation",
            ],
            "google_scholar": [
                '"dynamic rec"',
                '"dynamic recommender"',
                '"dynamic recommender systems"',
                '"dynamic recommendation"',
                '"time-aware recommender systems"',
                '"temporal recommendation"',
                '"sequential recommendation"',
            ],
        }
    elif "object" in tokens and "detection" in tokens:
        if "open-vocabulary" in tokens:
            source_queries = {
                "openalex": [
                    '("open-vocabulary" OR "open vocabulary" OR "open-set") AND "object detection"',
                    "open-vocabulary object detection",
                    "open vocabulary object detection",
                ],
                "dblp": ["open-vocabulary object detection", "open vocabulary detection", "object detection"],
                "arxiv": ["open-vocabulary object detection", "open vocabulary object detection", "open-set object detection"],
                "semantic_scholar": ["open-vocabulary object detection", "open vocabulary object detection", "open-set object detection"],
                "google_scholar": ['"open-vocabulary object detection"', '"open vocabulary object detection"', '"open-set object detection"'],
            }
        elif "small" in tokens:
            source_queries = {
                "openalex": ["small object detection", "tiny object detection", "small target detection"],
                "dblp": ["small object detection", "tiny object detection", "object detection"],
                "arxiv": ["small object detection", "tiny object detection", "small target detection"],
                "semantic_scholar": ["small object detection", "tiny object detection", "small target detection"],
                "google_scholar": ['"small object detection"', '"tiny object detection"', '"small target detection"'],
            }
    return source_queries


def _source_urls(
    source_queries: dict[str, list[str]],
    *,
    recent_years: int | None,
    from_year: int | None,
    to_year: int | None,
) -> dict[str, list[str]]:
    current_year = datetime.now(timezone.utc).year
    ylo = from_year
    if ylo is None and recent_years:
        ylo = current_year - recent_years + 1
    yhi = to_year
    urls = {"openalex": [], "dblp": [], "arxiv": [], "semantic_scholar": [], "google_scholar": []}
    for query in source_queries.get("openalex", []):
        urls["openalex"].append(f"https://openalex.org/works?search={quote_plus(query)}")
    for query in source_queries.get("dblp", []):
        urls["dblp"].append(f"https://dblp.org/search?q={quote_plus(query)}")
    for query in source_queries.get("arxiv", []):
        urls["arxiv"].append(f"https://arxiv.org/search/?query={quote_plus(query)}&searchtype=all")
    for query in source_queries.get("semantic_scholar", []):
        url = f"https://www.semanticscholar.org/search?q={quote_plus(query)}"
        if ylo:
            url += f"&year%5B0%5D={ylo}"
            if yhi:
                url += f"&year%5B1%5D={yhi}"
        urls["semantic_scholar"].append(url)
    for query in source_queries.get("google_scholar", []):
        url = f"https://scholar.google.com/scholar?q={quote_plus(query)}"
        if ylo:
            url += f"&as_ylo={ylo}"
        if yhi:
            url += f"&as_yhi={yhi}"
        urls["google_scholar"].append(url)
    return urls


KNOWN_VENUES = [
    "AAAI",
    "ACL",
    "ASPLOS",
    "CCS",
    "CHI",
    "CIKM",
    "COLING",
    "CVPR",
    "ECCV",
    "EMNLP",
    "ICCV",
    "ICDE",
    "ICLR",
    "ICML",
    "ICSE",
    "IJCAI",
    "INFOCOM",
    "ISCA",
    "ISSTA",
    "KDD",
    "MICRO",
    "MobiCom",
    "NAACL",
    "NDSS",
    "NeurIPS",
    "OSDI",
    "PLDI",
    "POPL",
    "SIGCOMM",
    "SIGGRAPH",
    "SIGIR",
    "SIGMOD",
    "SOSP",
    "T-PAMI",
    "TKDE",
    "TOIS",
    "TIFS",
    "TNNLS",
    "USENIX Security",
    "VLDB",
    "WSDM",
    "WWW",
]
