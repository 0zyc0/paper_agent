from __future__ import annotations

from dataclasses import dataclass
import re
from textwrap import fill
from typing import Any

from ..tools.kimi_writer import KimiRelatedWorkWriter
from ..skills.catalog import AcademicWritingSkillRouter
from .models import Paper, RelatedWorkDraft
from .rank import split_classic_and_recent


@dataclass(frozen=True)
class WritingPlan:
    """A small, explicit contract for the manuscript artefact being drafted."""

    kind: str
    title_zh: str
    title_en: str
    outline_zh: tuple[str, ...]
    outline_en: tuple[str, ...]

    def title_for(self, language: str) -> str:
        return self.title_zh if language.lower().startswith("zh") else self.title_en

    def outline_for(self, language: str) -> list[str]:
        return list(self.outline_zh if language.lower().startswith("zh") else self.outline_en)


_WRITING_PLANS: dict[str, WritingPlan] = {
    "related_work": WritingPlan(
        "related_work", "相关工作", "Related Work",
        ("研究脉络与问题定位", "方法路线与代表性工作", "近期进展与本文定位"),
        ("Research lineage and problem framing", "Method families and representative work", "Recent advances and positioning"),
    ),
    "survey": WritingPlan(
        "survey", "研究综述", "Literature Survey",
        ("范围与分类框架", "主要研究路线", "比较、挑战与研究机会"),
        ("Scope and taxonomy", "Major research lines", "Comparison, challenges, and opportunities"),
    ),
    "introduction": WritingPlan(
        "introduction", "引言", "Introduction",
        ("研究背景与问题", "现有工作与不足", "本文目标与贡献边界"),
        ("Background and problem", "Prior work and limitations", "Study objective and contribution boundary"),
    ),
    "method_section": WritingPlan(
        "method_section", "方法章节", "Methodology",
        ("设计目标与符号", "方法框架", "实现假设与可验证要点"),
        ("Design objective and notation", "Method framework", "Assumptions and verification points"),
    ),
    "experiment_section": WritingPlan(
        "experiment_section", "实验章节", "Experimental Design",
        ("评测目标", "数据、指标与对比设置", "结果解读边界"),
        ("Evaluation objective", "Data, metrics, and baselines", "Result interpretation boundary"),
    ),
    "summary": WritingPlan(
        "summary", "文献总结", "Literature Summary",
        ("核心问题", "主要发现", "仍待核实的问题"),
        ("Core problem", "Main findings", "Open verification items"),
    ),
    "outline": WritingPlan(
        "outline", "论文大纲", "Paper Outline",
        ("研究背景", "相关工作", "方法", "实验", "讨论与结论"),
        ("Introduction", "Related Work", "Method", "Experiments", "Discussion and Conclusion"),
    ),
    "bibliography": WritingPlan(
        "bibliography", "引用文献清单", "Bibliography Export",
        ("已导出 BibTeX", "引用键说明", "使用前核查"),
        ("Exported BibTeX", "Citation-key notes", "Verification before use"),
    ),
    "report": WritingPlan(
        "report", "研究报告", "Research Report",
        ("研究范围与证据边界", "主要研究路线", "可确认观察与后续工作"),
        ("Research scope and evidence boundary", "Major research lines", "Supported observations and next steps"),
    ),
}


class RelatedWorkGenerator:
    """Generate citation-aware academic prose for several manuscript sections.

    The class name is preserved for compatibility with the CLI and existing callers,
    but its output is no longer forced into a Related Work template.
    """

    def generate(
        self,
        *,
        query: str,
        papers: list[Paper],
        language: str = "en",
        max_papers: int = 18,
        use_llm: bool = False,
        writing_request: str | None = None,
        evidence_notes: list[dict] | None = None,
    ) -> RelatedWorkDraft:
        selected = papers[:max_papers]
        plan = writing_plan_for(writing_request or "")
        citation_keys = _citation_keys(selected)
        bibtex = "\n\n".join(_bibtex_for(paper, citation_keys[paper.id or ""]) for paper in selected)
        content = ""
        draft_source = "fallback"
        llm_attempted = False
        llm_error = ""

        if plan.kind == "bibliography":
            content = _bibliography_note(plan, selected, citation_keys, language)
        elif use_llm and selected:
            writer = KimiRelatedWorkWriter()
            if writer.available:
                llm_attempted = True
                try:
                    llm_content = writer.draft(
                        topic=query,
                        papers=selected,
                        citation_keys=citation_keys,
                        language=language,
                        writing_request=writing_request,
                        writing_plan=_writer_plan_payload(plan, language),
                        evidence_notes=evidence_notes,
                    )
                    content = _normalize_llm_section(llm_content, plan=plan, language=language)
                    _validate_llm_draft(content, citation_keys, plan)
                    draft_source = "llm"
                except Exception as exc:
                    # The draft remains available through the deterministic
                    # path, but the UI must be able to explain why it is a
                    # fallback rather than silently presenting it as model work.
                    llm_error = type(exc).__name__
                    content = ""

        if not content:
            content = self._fallback_draft(plan, query, selected, citation_keys, language)

        claim_map = _claim_map_from_content(content, selected, citation_keys, evidence_notes)
        quality_report = _quality_report(
            content,
            selected,
            citation_keys,
            evidence_notes,
            plan,
            draft_source=draft_source,
            llm_attempted=llm_attempted,
            llm_error=llm_error,
        )
        return RelatedWorkDraft(
            title=plan.title_for(language),
            query=query,
            content_markdown=content,
            bibtex=bibtex,
            paper_ids=[paper.id or "" for paper in selected],
            claim_map=claim_map,
            writing_kind=plan.kind,
            outline=plan.outline_for(language),
            quality_report=quality_report,
        )

    def _fallback_draft(
        self,
        plan: WritingPlan,
        query: str,
        papers: list[Paper],
        citation_keys: dict[str, str],
        language: str,
    ) -> str:
        if plan.kind == "outline":
            return _outline_draft(plan, query, papers, citation_keys, language)
        if plan.kind == "bibliography":
            return _bibliography_note(plan, papers, citation_keys, language)
        if plan.kind == "related_work":
            return self._related_work_fallback(query, papers, citation_keys, language)
        if plan.kind == "survey":
            return _survey_fallback(plan, query, papers, citation_keys, language)
        if plan.kind == "report":
            return _report_fallback(plan, query, papers, citation_keys, language)
        if plan.kind == "introduction":
            return _introduction_fallback(plan, query, papers, citation_keys, language)
        if plan.kind == "method_section":
            return _method_fallback(plan, query, papers, citation_keys, language)
        if plan.kind == "experiment_section":
            return _experiment_fallback(plan, query, papers, citation_keys, language)
        if plan.kind == "summary":
            return _summary_fallback(plan, query, papers, citation_keys, language)
        return _section_fallback(plan, query, papers, citation_keys, language)

    def _related_work_fallback(
        self,
        query: str,
        papers: list[Paper],
        citation_keys: dict[str, str],
        language: str,
    ) -> str:
        classic, recent = split_classic_and_recent(papers)
        classic = classic[:5]
        recent = recent[:7]
        remainder = [paper for paper in papers if paper not in classic and paper not in recent][:4]
        title = "相关工作" if language.lower().startswith("zh") else "Related Work"
        paragraphs: list[str] = []

        if language.lower().startswith("zh"):
            if classic:
                paragraphs.append(
                    _paragraph(
                        f"### 研究脉络与问题定位\n\n围绕 {query}，已有研究为问题定义、常见建模假设和评测边界提供了基础。"
                        f"本文可将这些工作作为研究脉络的起点，并据此说明后续方法比较的范围 {_cites(citation_keys, classic)}。"
                    )
                )
            if recent:
                paragraphs.append(
                    _paragraph(
                        "### 方法路线与代表性工作\n\n近期工作从不同任务设定和建模选择推进这一方向。"
                        f"代表性研究包括 {_named_papers(recent[:4])}；这些记录可用于组织方法路线，而不应被写成逐篇罗列 {_cites(citation_keys, recent)}。"
                    )
                )
            if remainder or papers:
                positioned = remainder or papers[:6]
                paragraphs.append(
                    _paragraph(
                        "### 近期进展与本文定位\n\n综合现有证据，正文应明确区分已有工作解决的问题、当前研究的差异，以及尚需全文核实的实验或局限性结论。"
                        f"在缺少完整原文的记录上，宜保持高层次表述 {_cites(citation_keys, positioned)}。"
                    )
                )
        else:
            if classic:
                paragraphs.append(
                    _paragraph(
                        f"### Research Lineage and Problem Framing\n\nWork on {query} establishes the problem definitions, modelling assumptions, and evaluation boundaries that frame this section {_cites(citation_keys, classic)}."
                    )
                )
            if recent:
                paragraphs.append(
                    _paragraph(
                        "### Method Families and Representative Work\n\nRecent studies advance the area through distinct task formulations and modelling choices. "
                        f"Representative records include {_named_papers(recent[:4])}; they should be synthesized as method families rather than enumerated paper by paper {_cites(citation_keys, recent)}."
                    )
                )
            if remainder or papers:
                positioned = remainder or papers[:6]
                paragraphs.append(
                    _paragraph(
                        "### Recent Advances and Positioning\n\nThe manuscript should separate what prior work addresses from the present study's intended distinction, while reserving detailed performance or limitation claims for full-text verification "
                        f"{_cites(citation_keys, positioned)}."
                    )
                )
        body = "\n\n".join(paragraphs) or _no_evidence_text(language)
        return f"## {title}\n\n{body}\n"


def writing_plan_for(request: str, *, writing_kind: str = "") -> WritingPlan:
    """Resolve the writer skill selected by the Agent, with legacy fallback.

    The normal Agent route supplies ``writing_kind`` as a constrained enum.
    The text fallback remains for existing CLI callers and older saved tasks.
    """
    normalized_kind = str(writing_kind or "").strip().lower()
    if normalized_kind in _WRITING_PLANS:
        return _WRITING_PLANS[normalized_kind]
    if normalized_kind in {"general", "markdown"}:
        return _WRITING_PLANS["report"]
    text = (request or "").lower()
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return _WRITING_PLANS["related_work"]
    if any(token in compact for token in ("bibtex", ".bib", "引用文件", "参考文献导出")) and not any(
        token in compact for token in ("relatedwork", "相关工作", "综述", "引言", "方法", "实验", "章节")
    ):
        return _WRITING_PLANS["bibliography"]
    if any(token in compact for token in ("大纲", "outline", "结构")):
        return _WRITING_PLANS["outline"]
    if any(token in compact for token in ("relatedwork", "相关工作")):
        return _WRITING_PLANS["related_work"]
    if any(token in compact for token in ("survey", "综述", "文献回顾", "调研")):
        return _WRITING_PLANS["survey"]
    if any(token in compact for token in ("introduction", "引言", "研究背景")):
        return _WRITING_PLANS["introduction"]
    if any(token in compact for token in ("method", "方法", "模型设计")):
        return _WRITING_PLANS["method_section"]
    if any(token in compact for token in ("experiment", "实验", "评测", "evaluation")):
        return _WRITING_PLANS["experiment_section"]
    if any(token in compact for token in ("summary", "总结", "概述")):
        return _WRITING_PLANS["summary"]
    return _WRITING_PLANS["report"]


def _writer_plan_payload(plan: WritingPlan, language: str) -> dict[str, Any]:
    route = AcademicWritingSkillRouter().route(plan.kind)
    return {
        "kind": plan.kind,
        "title": plan.title_for(language),
        "outline": plan.outline_for(language),
        "skill": "academic-writing",
        "resource": route.payload()["resource"],
        "routing_contract": route.routing,
        "instruction": route.instruction,
        "markdown_contract": route.markdown_contract,
    }


def _outline_draft(plan: WritingPlan, query: str, papers: list[Paper], citation_keys: dict[str, str], language: str) -> str:
    title = plan.title_for(language)
    lines = [f"## {title}", "", f"**Topic:** {query}", ""]
    for index, heading in enumerate(plan.outline_for(language), start=1):
        paper_slice = papers[max(0, index - 1): index + 1] or papers[:2]
        citation = _cites(citation_keys, paper_slice)
        if language.lower().startswith("zh"):
            lines.append(f"{index}. **{heading}**：围绕该部分明确研究问题、证据来源和待验证假设 {citation}".rstrip())
        else:
            lines.append(f"{index}. **{heading}**: state the question, evidence base, and assumptions to be verified {citation}".rstrip())
    return "\n".join(lines).strip() + "\n"


def _bibliography_note(plan: WritingPlan, papers: list[Paper], citation_keys: dict[str, str], language: str) -> str:
    title = plan.title_for(language)
    if language.lower().startswith("zh"):
        body = f"已为 {len(papers)} 篇当前证据论文生成 BibTeX 文件。正文引用请使用下列键："
    else:
        body = f"A BibTeX file has been generated for the {len(papers)} current evidence papers. Use the following keys in the manuscript:"
    keys = ", ".join(f"`{citation_keys[paper.id or paper.title]}`" for paper in papers[:18]) or "—"
    return f"## {title}\n\n{body}\n\n{keys}\n"


def _family_groups(papers: list[Paper]) -> list[tuple[str, list[Paper]]]:
    groups: dict[str, list[Paper]] = {}
    for paper in papers:
        groups.setdefault(_survey_family(paper), []).append(paper)
    return sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[:4]


def _family_synthesis(family: str, *, language: str) -> str:
    """Provide a distinct, conservative synthesis frame for each research line."""
    zh = language.lower().startswith("zh")
    frames = {
        "因果与反事实校正": (
            "这一路线把曝光、选择或观测偏差视为需要显式处理的机制，通常要求交代可观测的偏差信号与反事实/倾向性假设。",
            "This line treats exposure, selection, or observation bias as an explicit mechanism and requires the observable signal and counterfactual or propensity assumptions to be stated.",
        ),
        "公平性、曝光与偏差校正": (
            "这一路线关注推荐结果在用户、物品或曝光机会之间的分配，并需要同时说明优化目标与可能的效用权衡。",
            "This line focuses on how recommendations are distributed across users, items, or exposure opportunities, and requires both the optimization target and utility trade-offs to be made explicit.",
        ),
        "反馈回路与交互环境中的偏差": (
            "这一路线将用户交互与后续推荐相互影响的过程纳入问题定义，适合进一步核对时间切分、模拟环境或在线反馈假设。",
            "This line includes the mutual influence between user interaction and later recommendations, making temporal splits, simulation environments, or online-feedback assumptions central verification items.",
        ),
        "新闻推荐中的内容与媒体偏差": (
            "这一路线把内容呈现和媒体来源的偏差纳入推荐目标，需要区分用户偏好、内容属性与平台曝光之间的关系。",
            "This line includes content presentation and media-source bias in the recommendation target, requiring a distinction among user preference, content attributes, and platform exposure.",
        ),
        "大模型增强的推荐建模": (
            "这一路线探索语言模型在表示、推理或重排序中的作用，应重点核对模型输入、外部知识来源与可复现评测设置。",
            "This line explores language models for representation, reasoning, or reranking; its inputs, external knowledge sources, and reproducible evaluation settings need careful verification.",
        ),
        "时序与动态推荐场景": (
            "这一路线强调兴趣、上下文或候选集合随时间变化，后续比较应明确序列长度、时间窗口和冷启动处理。",
            "This line emphasizes changing interests, contexts, or candidate sets over time; subsequent comparisons should specify sequence length, temporal windows, and cold-start treatment.",
        ),
        "检索增强与知识访问": (
            "这一路线通过外部证据或索引补充模型知识，比较时应区分检索质量、生成质量和证据可追溯性。",
            "This line supplements model knowledge with external evidence or indexes; comparisons should distinguish retrieval quality, generation quality, and evidence traceability.",
        ),
    }
    generic = (
        "这一路线代表了一组可比较的任务建模选择；后续写作应先固定问题定义、建模假设与评测对象，再讨论方法差异。",
        "This line represents a comparable set of task-modelling choices; subsequent writing should fix the problem definition, modelling assumptions, and evaluation target before discussing method differences.",
    )
    return (frames.get(family, generic))[0 if zh else 1]


def _report_fallback(
    plan: WritingPlan,
    query: str,
    papers: list[Paper],
    citation_keys: dict[str, str],
    language: str,
) -> str:
    """Write an evidence-led report instead of reusing a generic section template."""
    title = plan.title_for(language)
    if not papers:
        return f"## {title}\n\n{_no_evidence_text(language)}\n"
    headings = plan.outline_for(language)
    families = _family_groups(papers)
    if language.lower().startswith("zh"):
        lines = [
            f"## {title}", "", f"### {headings[0]}", "",
            f"本报告围绕 {query}，综合当前文献库中 {len(papers)} 篇可用记录。"
            "以下判断以题名、摘要和已提取的本地全文片段为限；未获得全文的论文只用于研究定位，"
            f"不据此推断具体实验效果。 {_cites(citation_keys, papers[:min(6, len(papers))])}", "",
            f"### {headings[1]}", "",
        ]
        for family, group in families:
            lines.extend([
                f"#### {family}", "",
                _family_synthesis(family, language=language)
                + f"当前证据可将其作为独立研究路线加以比较 {_cites(citation_keys, group)}。", "",
            ])
        lines.extend([
            f"### {headings[2]}", "",
            "现有证据表明，后续研究报告应优先建立“偏差/任务定义、可观测反馈、建模假设、评测协议”四列对照，"
            "再针对可获得全文的工作核验方法细节和实验结论。对于尚无全文的记录，保留为待核验线索而非形成确定性结论。"
            f" {_cites(citation_keys, papers[-min(6, len(papers)):])}",
        ])
    else:
        lines = [
            f"## {title}", "", f"### {headings[0]}", "",
            f"This report examines {query} using {len(papers)} records from the current library. "
            "Its claims are bounded by titles, abstracts, and available local full-text excerpts; records without full text are used for positioning rather than detailed empirical conclusions. "
            f"{_cites(citation_keys, papers[:min(6, len(papers))])}", "",
            f"### {headings[1]}", "",
        ]
        for family, group in families:
            lines.extend([
                f"#### {family}", "",
                _family_synthesis(family, language=language)
                + f"The current evidence supports treating it as a distinct research line for comparison {_cites(citation_keys, group)}.", "",
            ])
        lines.extend([
            f"### {headings[2]}", "",
            "The next report iteration should compare bias or task definitions, observable feedback, modelling assumptions, and evaluation protocols across the available lines. "
            f"Records without full text remain verification leads rather than conclusive empirical evidence. {_cites(citation_keys, papers[-min(6, len(papers)):])}",
        ])
    return "\n".join(lines).strip() + "\n"


def _introduction_fallback(plan: WritingPlan, query: str, papers: list[Paper], citation_keys: dict[str, str], language: str) -> str:
    title, headings = plan.title_for(language), plan.outline_for(language)
    evidence = _family_groups(papers)
    if not papers:
        return f"## {title}\n\n{_no_evidence_text(language)}\n"
    if language.lower().startswith("zh"):
        lines = [
            f"## {title}", "", f"### {headings[0]}", "",
            f"{query} 涉及如何在具体任务约束下建立可靠的建模与评测过程。当前文献显示，该问题已在多种研究路线中被讨论，"
            f"为研究动机和问题边界提供了基础 {_cites(citation_keys, papers[:min(5, len(papers))])}。", "",
            f"### {headings[1]}", "",
            "现有记录覆盖 " + "、".join(name for name, _ in evidence) + " 等路线，但不同工作采用的任务假设与评测边界未必一致。"
            "因此，不能仅依据题名或摘要直接比较优劣；需要在全文层面核对方法条件与实验协议。"
            f" {_cites(citation_keys, papers[-min(6, len(papers)):])}", "",
            f"### {headings[2]}", "",
            "据此，本文可将研究目标限定为：明确目标问题的定义，选择可验证的建模路线，并在一致的评测协议下说明其与现有工作的差异。"
            "具体方法、数据和贡献仍需由作者补充后再写成确定性陈述。",
        ]
    else:
        lines = [
            f"## {title}", "", f"### {headings[0]}", "",
            f"{query} concerns how reliable modelling and evaluation can be established under task-specific constraints. The current literature provides several starting points for framing the problem {_cites(citation_keys, papers[:min(5, len(papers))])}.", "",
            f"### {headings[1]}", "",
            "The available records span " + ", ".join(name for name, _ in evidence) + ", but their assumptions and evaluation boundaries need not be comparable. Detailed comparisons therefore require verification against the original papers. " + _cites(citation_keys, papers[-min(6, len(papers)):]), "",
            f"### {headings[2]}", "",
            "This study can consequently define the target problem, select a verifiable modelling route, and state its distinction from prior work under a consistent evaluation protocol. Specific methods, datasets, and contributions must be supplied by the author before they are presented as established facts.",
        ]
    return "\n".join(lines).strip() + "\n"


def _method_fallback(plan: WritingPlan, query: str, papers: list[Paper], citation_keys: dict[str, str], language: str) -> str:
    title, headings = plan.title_for(language), plan.outline_for(language)
    groups = _family_groups(papers)
    if language.lower().startswith("zh"):
        lines = [
            f"## {title}", "", f"### {headings[0]}", "",
            f"本节是围绕 {query} 的方法设计草案，而非对已实现系统的描述。设计时应先固定任务输入、预测目标、可观测反馈和约束条件，"
            f"以便与现有研究进行可核验的比较 {_cites(citation_keys, papers[:min(5, len(papers))])}。", "",
            f"### {headings[1]}", "",
            "可从现有文献的 " + "、".join(name for name, _ in groups) + " 等路线中选择主框架，并明确每个模块服务的假设。"
            "当前证据只支持将这些路线作为设计候选，不能据此声称本文已经采用或优于任何一种方法。"
            f" {_cites(citation_keys, papers[-min(6, len(papers)):])}", "",
            f"### {headings[2]}", "",
            "在形成正式方法章节前，作者还需要补充符号定义、模型输入输出、优化目标、训练/推理流程，以及与研究问题对应的可证伪假设。",
        ]
    else:
        lines = [
            f"## {title}", "", f"### {headings[0]}", "",
            f"This is a method-design draft for {query}, not a description of an implemented system. The design should first fix task inputs, prediction targets, observable feedback, and constraints {_cites(citation_keys, papers[:min(5, len(papers))])}.", "",
            f"### {headings[1]}", "",
            "The literature offers candidate framing through " + ", ".join(name for name, _ in groups) + ". These are design options only and do not establish that the present study adopts or outperforms any approach. " + _cites(citation_keys, papers[-min(6, len(papers)):]), "",
            f"### {headings[2]}", "",
            "Before this becomes a formal method section, the author must provide notation, module inputs and outputs, objective functions, training/inference steps, and falsifiable assumptions aligned with the research question.",
        ]
    return "\n".join(lines).strip() + "\n"


def _experiment_fallback(plan: WritingPlan, query: str, papers: list[Paper], citation_keys: dict[str, str], language: str) -> str:
    title, headings = plan.title_for(language), plan.outline_for(language)
    if language.lower().startswith("zh"):
        lines = [
            f"## {title}", "", f"### {headings[0]}", "",
            f"针对 {query}，实验应回答：所提出设计是否改善目标任务，同时是否在关键约束下保持稳健。当前文献可用于界定需要比较的任务与评测边界 {_cites(citation_keys, papers[:min(6, len(papers))])}。", "",
            f"### {headings[1]}", "",
            "正式实验前需由作者确定数据来源、时间切分或交互切分方式、评价指标、可复现基线和资源预算。"
            "对于与偏差、公平性或动态反馈相关的主题，建议同时报告总体效果与分组/时间维度的稳健性检查。"
            f" {_cites(citation_keys, papers[-min(6, len(papers)):])}", "",
            f"### {headings[2]}", "",
            "本草稿不包含任何实验数值或优劣结论。结果章节应仅在完成预注册式的比较设置、消融实验和误差分析后填写。",
        ]
    else:
        lines = [
            f"## {title}", "", f"### {headings[0]}", "",
            f"For {query}, the evaluation should ask whether the proposed design improves the target task while remaining robust under the relevant constraints. The current literature helps define the task and evaluation boundary {_cites(citation_keys, papers[:min(6, len(papers))])}.", "",
            f"### {headings[1]}", "",
            "Before running experiments, the author must specify data sources, temporal or interaction splits, metrics, reproducible baselines, and a resource budget. Topics involving bias, fairness, or dynamic feedback should report both aggregate outcomes and grouped or temporal robustness checks. " + _cites(citation_keys, papers[-min(6, len(papers)):]), "",
            f"### {headings[2]}", "",
            "This draft contains no numerical result or performance claim. The results section should be filled only after comparison settings, ablations, and error analyses have been completed.",
        ]
    return "\n".join(lines).strip() + "\n"


def _summary_fallback(plan: WritingPlan, query: str, papers: list[Paper], citation_keys: dict[str, str], language: str) -> str:
    title, headings = plan.title_for(language), plan.outline_for(language)
    groups = _family_groups(papers)
    if language.lower().startswith("zh"):
        lines = [
            f"## {title}", "", f"### {headings[0]}", "",
            f"当前文献围绕 {query} 展开，涵盖 " + "、".join(name for name, _ in groups) + " 等问题视角。" + _cites(citation_keys, papers[:min(6, len(papers))]), "",
            f"### {headings[1]}", "",
            "可确认的共同点是：不同路线都需要将研究问题、假设和评测协议一并说明，才具有可比较性。"
            f" {_cites(citation_keys, papers[-min(6, len(papers)):])}", "",
            f"### {headings[2]}", "",
            "仍待核实的是各论文的具体算法细节、数据处理、实验设置与限制。缺少全文的记录不应被用来支撑细粒度结论。",
        ]
    else:
        lines = [
            f"## {title}", "", f"### {headings[0]}", "",
            f"The current literature on {query} spans " + ", ".join(name for name, _ in groups) + ". " + _cites(citation_keys, papers[:min(6, len(papers))]), "",
            f"### {headings[1]}", "",
            "A supported common observation is that problem definitions, assumptions, and evaluation protocols must be reported together before methods can be compared. " + _cites(citation_keys, papers[-min(6, len(papers)):]), "",
            f"### {headings[2]}", "",
            "Algorithmic details, data treatment, experimental settings, and limitations remain to be verified in the original texts. Records without full text should not support fine-grained conclusions.",
        ]
    return "\n".join(lines).strip() + "\n"


def _section_fallback(plan: WritingPlan, query: str, papers: list[Paper], citation_keys: dict[str, str], language: str) -> str:
    title = plan.title_for(language)
    headings = plan.outline_for(language)
    groups = [papers[:5], papers[5:10], papers[10:15]]
    paragraphs: list[str] = []
    for heading, group in zip(headings, groups, strict=False):
        evidence = group or papers[:5]
        citations = _cites(citation_keys, evidence)
        if language.lower().startswith("zh"):
            evidence_summary = _abstract_evidence_summary(evidence[:2], language)
            text = (
                f"### {heading}\n\n围绕 {query}，本节基于当前证据整理可确认的研究线索。"
                f"其中 {_named_papers(evidence[:3]) or '现有记录'} 为该部分提供定位依据 {citations}。"
                f"{evidence_summary} 涉及具体机制、实验提升或局限性时，仍应回到原文逐项核查。"
            )
        else:
            evidence_summary = _abstract_evidence_summary(evidence[:2], language)
            text = (
                f"### {heading}\n\nFor {query}, this section synthesizes the research signals supported by the current evidence set. "
                f"{_named_papers(evidence[:3]) or 'The available records'} provide the immediate positioning for this part {citations}. "
                f"{evidence_summary} Detailed mechanisms, empirical gains, and limitations should be verified against the original papers."
            )
        paragraphs.append(_paragraph(text))
    return f"## {title}\n\n" + "\n\n".join(paragraphs or [_no_evidence_text(language)]) + "\n"


def _survey_fallback(
    plan: WritingPlan,
    query: str,
    papers: list[Paper],
    citation_keys: dict[str, str],
    language: str,
) -> str:
    """Produce an evidence-led survey when the writing model is unavailable."""
    title = plan.title_for(language)
    if not papers:
        return f"## {title}\n\n{_no_evidence_text(language)}\n"

    groups: dict[str, list[Paper]] = {}
    for paper in papers:
        groups.setdefault(_survey_family(paper), []).append(paper)
    chosen = list(groups.items())[:4]
    headings = plan.outline_for(language)

    if language.lower().startswith("zh"):
        family_names = "、".join(name for name, _ in chosen)
        lines = [
            f"## {title}", "", f"### {headings[0]}", "",
            f"本调研聚焦 {query}，以当前文献库中的 {len(papers)} 篇记录为基础。"
            f"从题名和摘要可归纳出 {family_names} 等研究路线；以下只综合可由现有记录支持的研究定位，"
            f"不将摘要信息扩展为未经核验的实验结论。 {_cites(citation_keys, papers[:min(6, len(papers))])}", "",
        ]
        for family, group in chosen:
            evidence_summary = _abstract_evidence_summary(group[:2], language)
            lines.extend([
                f"### {family}", "",
                f"该路线在当前文献中包括 {_named_papers(group[:3])}。"
                f"从摘要可确认的共同线索是 {evidence_summary or '它们均围绕该类偏差的识别、建模或评测展开'}。"
                "涉及具体算法细节、指标提升或适用边界时，仍应以原文为准。 "
                f"{_cites(citation_keys, group)}", "",
            ])
        lines.extend([
            f"### {headings[-1]}", "",
            "现有文献库已覆盖多条研究路线，但不同论文的偏差定义、数据分布假设与评测协议未必一致。"
            "后续写作宜先按这些维度建立对照表，再从可获得全文的论文中核验方法细节和实验结论。"
            f" {_cites(citation_keys, papers[-min(6, len(papers)):])}",
        ])
    else:
        family_names = ", ".join(name for name, _ in chosen)
        lines = [
            f"## {title}", "", f"### {headings[0]}", "",
            f"This survey focuses on {query} and synthesizes the {len(papers)} records currently available in the library. "
            f"The titles and abstracts indicate the following research lines: {family_names}. "
            f"{_cites(citation_keys, papers[:min(6, len(papers))])}", "",
        ]
        for family, group in chosen:
            evidence_summary = _abstract_evidence_summary(group[:2], language)
            lines.extend([
                f"### {family}", "",
                f"This line is represented by {_named_papers(group[:3])}. The available records support positioning these papers "
                f"as a comparable modelling perspective. {evidence_summary} Detailed mechanisms and empirical claims require full-text verification. {_cites(citation_keys, group)}", "",
            ])
        lines.extend([
            f"### {headings[-1]}", "",
            "The current library spans several research lines, but their bias definitions, distributional assumptions, and evaluation protocols may differ. "
            f"A final survey should compare those dimensions against the original papers. {_cites(citation_keys, papers[-min(6, len(papers)):])}",
        ])
    return "\n".join(lines).strip() + "\n"


def _survey_family(paper: Paper) -> str:
    text = f"{paper.title} {paper.abstract}".lower()
    if any(token in text for token in ("retrieval augmented", "retrieval-augmented", "retrieval augmentation", "rag", "knowledge-intensive")):
        return "检索增强与知识访问"
    if any(token in text for token in ("object detection", "segmentation", "image classification", "visual recognition")):
        return "视觉任务建模与评测"
    if any(token in text for token in ("graph neural", "gnn", "graph learning", "knowledge graph")):
        return "图结构表示与推理"
    if any(token in text for token in ("news recommender", "news recommendation", "media bias")):
        return "新闻推荐中的内容与媒体偏差"
    if any(token in text for token in ("feedback loop", "feedback-loop", "performative")):
        return "反馈回路与交互环境中的偏差"
    if any(token in text for token in ("causal", "counterfactual", "propensity", "ips", "inverse propensity")):
        return "因果与反事实校正"
    if any(token in text for token in ("fairness", "fair exposure", "popularity bias", "exposure bias", "debias")):
        return "公平性、曝光与偏差校正"
    if any(token in text for token in ("large language model", "llm", "language model", "generative")):
        return "大模型增强的推荐建模"
    if any(token in text for token in ("sequential", "session", "temporal", "dynamic", "streaming")):
        return "时序与动态推荐场景"
    return "推荐建模与评测"


def _abstract_evidence_summary(papers: list[Paper], language: str) -> str:
    """Return a short, attributable signal from abstracts for offline drafts.

    This keeps the deterministic fallback useful without inventing methods or
    results when the writing model is unavailable.
    """
    snippets = [_first_usable_abstract_sentence(paper.abstract) for paper in papers]
    snippets = [snippet for snippet in snippets if snippet]
    if not snippets:
        return ""
    if language.lower().startswith("zh"):
        return "；".join(f"摘要指出“{snippet}”" for snippet in snippets[:2])
    return "Abstract evidence notes that " + "; ".join(snippets[:2])


def _first_usable_abstract_sentence(abstract: str | None) -> str:
    text = re.sub(r"^[\s.…]+", "", abstract or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 40:
        return ""
    sentences = re.split(r"(?<=[.!?。])\s+", text)
    for sentence in sentences:
        sentence = sentence.strip(" .。")
        if 40 <= len(sentence) <= 260 and "…" not in sentence:
            return sentence
    return ""


def _normalize_llm_section(content: str, *, plan: WritingPlan, language: str) -> str:
    text = (content or "").strip()
    if not text:
        raise ValueError("LLM returned an empty draft.")
    text = _strip_source_sections(text)
    canonical_heading = f"## {plan.title_for(language)}"
    first_heading = re.search(r"^#{1,3}\s+.+$", text, flags=re.M)
    if first_heading:
        text = text[:first_heading.start()] + canonical_heading + text[first_heading.end():]
    else:
        text = f"{canonical_heading}\n\n{text}"
    return text.strip() + "\n"


def _validate_llm_draft(content: str, citation_keys: dict[str, str], plan: WritingPlan) -> None:
    if _looks_like_source_listing(content):
        raise ValueError("LLM returned a source list instead of an academic draft.")
    used = _citation_keys_in_text(content)
    allowed = set(citation_keys.values())
    if any(key not in allowed for key in used):
        raise ValueError("LLM returned a citation key outside the evidence set.")
    if plan.kind not in {"outline", "bibliography"} and citation_keys and not used:
        raise ValueError("LLM draft did not cite the supplied evidence.")
    if len(content) < 180 and plan.kind not in {"outline", "bibliography"}:
        raise ValueError("LLM draft is too short to be a usable manuscript section.")
    if plan.kind not in {"outline", "bibliography"} and citation_keys:
        minimum_citations = min(2, len(citation_keys))
        if len(set(used)) < minimum_citations:
            raise ValueError("LLM draft cites too little of the selected evidence.")
        subsection_count = len(re.findall(r"^###\s+\S+", content, flags=re.M))
        if subsection_count < 2:
            raise ValueError("LLM draft does not follow the required section structure.")
        sections = _prose_sections(content)
        if len(sections) < 2 or any(not _is_substantive_prose(section) for section in sections):
            raise ValueError("LLM draft contains an empty or outline-only subsection.")
        body_size = sum(len(re.sub(r"\s+|\\cite\{[^}]+\}", "", section)) for section in sections)
        if body_size < 320:
            raise ValueError("LLM draft lacks enough manuscript prose.")
        if _has_repeated_paragraphs(content):
            raise ValueError("LLM draft repeats a paragraph instead of developing the section.")


def _prose_sections(content: str) -> list[str]:
    """Return bodies following level-3 headings for quality validation."""
    matches = list(re.finditer(r"^###\s+.+$", content or "", flags=re.M))
    bodies: list[str] = []
    for index, heading in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        body = content[heading.end():end].strip()
        if body:
            bodies.append(body)
        else:
            bodies.append("")
    return bodies


def _is_substantive_prose(value: str) -> bool:
    clean = re.sub(r"\\cite\{[^}]+\}", "", value or "")
    clean = re.sub(r"[#*_`>-]", "", clean)
    compact = re.sub(r"\s+", "", clean)
    return len(compact) >= 70 and not re.fullmatch(r"(?:\d+[.)]?\s*)+", compact or "")


def _strip_source_sections(content: str) -> str:
    pattern = re.compile(
        r"\n{0,2}#{1,3}\s*(?:Retrieved Sources|已检索来源|References|参考文献|Bibliography|文献列表)\b[\s\S]*$",
        flags=re.IGNORECASE,
    )
    return pattern.sub("", content).strip()


def _looks_like_source_listing(content: str) -> bool:
    text = re.sub(r"\s+", " ", content or "").strip()
    lowered = text.lower()
    if len(text) < 80:
        return True
    if lowered.startswith(("retrieved sources", "references", "bibliography")):
        return True
    metadata_markers = sum(lowered.count(marker) for marker in ("source:", "url:", "doi:", "retrieved from", "@article"))
    cite_count = len(_citation_keys_in_text(text))
    numbered_items = len(re.findall(r"(?:^|\n)\s*\d+\.\s+\S+", content or ""))
    return metadata_markers >= 2 or (numbered_items >= 4 and cite_count == 0)


def _citation_keys_in_text(content: str) -> list[str]:
    keys: list[str] = []
    for match in re.findall(r"\\cite\{([^}]+)\}", content or ""):
        keys.extend(part.strip() for part in match.split(",") if part.strip())
    return keys


def _has_repeated_paragraphs(content: str) -> bool:
    """Reject low-effort LLM outputs that reuse the same prose across headings."""
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", content):
        clean = re.sub(r"^#{1,4}\s+.*$", "", paragraph, flags=re.M)
        clean = re.sub(r"\\cite\{[^}]+\}", "", clean)
        clean = re.sub(r"\s+", " ", clean.lower()).strip()
        if len(clean) >= 60:
            paragraphs.append(clean)
    for index, left in enumerate(paragraphs):
        for right in paragraphs[index + 1:]:
            if left == right:
                return True
            # Character-set overlap marks almost every English academic
            # paragraph as a duplicate because they share the same alphabet.
            # Compare words for alphabetic text and 3-grams for Chinese text.
            left_tokens = _repeat_tokens(left)
            right_tokens = _repeat_tokens(right)
            overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
            if overlap > 0.88:
                return True
    return False


def _repeat_tokens(value: str) -> set[str]:
    words = set(re.findall(r"[a-z][a-z0-9-]{2,}", value.lower()))
    if words:
        return words
    compact = re.sub(r"[^\u4e00-\u9fff]", "", value)
    return {compact[index:index + 3] for index in range(max(0, len(compact) - 2))}


def _claim_map_from_content(
    content: str,
    papers: list[Paper],
    citation_keys: dict[str, str],
    evidence_notes: list[dict] | None,
) -> list[dict[str, Any]]:
    id_by_key = {key: paper.id for paper in papers if (key := citation_keys.get(paper.id or paper.title))}
    evidence_by_id = {str(item.get("paper_id") or ""): item for item in evidence_notes or [] if isinstance(item, dict)}
    claims: list[dict[str, Any]] = []
    for paragraph in re.split(r"\n\s*\n", content):
        clean = re.sub(r"^#{1,4}\s+.*$", "", paragraph, flags=re.M).strip()
        keys = _citation_keys_in_text(clean)
        paper_ids = [id_by_key[key] for key in keys if id_by_key.get(key)]
        if not clean or not paper_ids:
            continue
        levels = [str((evidence_by_id.get(paper_id) or {}).get("evidence_level") or "metadata_only") for paper_id in paper_ids]
        if levels and all(level == "local_fulltext" for level in levels):
            status = "local_fulltext"
        elif levels and all(level == "abstract_only" for level in levels):
            status = "abstract_only"
        elif levels and all(level in {"metadata_only", "downloadable_metadata", "local_pdf_without_text"} for level in levels):
            status = "metadata_only"
        else:
            status = "mixed_evidence"
        claims.append(
            {
                "claim": re.sub(r"\\cite\{[^}]+\}", "", clean).replace("\n", " ").strip()[:500],
                "paper_ids": list(dict.fromkeys(paper_ids)),
                "citation_keys": list(dict.fromkeys(key for key in keys if id_by_key.get(key))),
                "papers": [
                    _paper_reference(next((paper for paper in papers if paper.id == paper_id), None))
                    for paper_id in list(dict.fromkeys(paper_ids))
                ],
                "evidence_status": status,
            }
        )
    if claims:
        return claims
    return [{
        "claim": "The generated draft is limited to the selected evidence set.",
        "paper_ids": [paper.id for paper in papers],
        "citation_keys": [citation_keys.get(paper.id or paper.title) for paper in papers],
        "papers": [_paper_reference(paper) for paper in papers],
        "evidence_status": "metadata_only",
    }] if papers else []


def _paper_reference(paper: Paper | None) -> dict[str, str]:
    if paper is None:
        return {}
    return {
        "paper_id": paper.id or "",
        "title": paper.title,
        "doi": paper.doi or "",
        "url": paper.source_url or paper.pdf_url or "",
    }


def _quality_report(
    content: str,
    papers: list[Paper],
    citation_keys: dict[str, str],
    evidence_notes: list[dict] | None,
    plan: WritingPlan,
    *,
    draft_source: str,
    llm_attempted: bool,
    llm_error: str,
) -> dict[str, Any]:
    used = set(_citation_keys_in_text(content))
    evidence = [item for item in evidence_notes or [] if isinstance(item, dict)]
    fulltext_count = sum(item.get("evidence_level") == "local_fulltext" for item in evidence)
    weak_count = sum(
        item.get("evidence_level") in {"abstract_only", "metadata_only", "downloadable_metadata", "local_pdf_without_text"}
        for item in evidence
    )
    warnings: list[str] = []
    if weak_count:
        warnings.append(f"{weak_count} 篇论文缺少可用全文，正文中的细节性结论应在投稿前核对原文。")
    if papers and len(used) < min(2, len(papers)) and plan.kind not in {"outline", "bibliography"}:
        warnings.append("当前草稿引用覆盖较少，建议补充或筛选证据论文。")
    metadata_issues = _citation_metadata_issues(papers)
    if metadata_issues:
        warnings.append(f"{len(metadata_issues)} 篇论文的作者或出版物信息被来源截断，BibTeX 已保守省略不完整字段，投稿前应从出版社页面补全。")
    if draft_source != "llm" and plan.kind not in {"outline", "bibliography"}:
        warnings.append("当前为基于题名和摘要的结构化兜底草稿，不应直接作为投稿版本；连接写作模型或补充全文后可生成更深入的综合。")
    if llm_attempted and draft_source != "llm":
        warnings.append(f"写作模型未返回可用草稿（{llm_error or 'unknown error'}），已自动使用结构化兜底草稿。")
    return {
        "writing_kind": plan.kind,
        "word_count": len(re.sub(r"\s+", "", re.sub(r"[#*_`>-]", "", content))),
        "citation_uses": len(_citation_keys_in_text(content)),
        "cited_paper_count": len(used),
        "available_paper_count": len(papers),
        "fulltext_paper_count": fulltext_count,
        "weak_evidence_count": weak_count,
        "draft_source": draft_source,
        "llm_attempted": llm_attempted,
        "llm_error": llm_error,
        "citation_coverage": round(len(used) / len(papers), 3) if papers else 0.0,
        "citation_metadata_issues": metadata_issues,
        "warnings": warnings,
    }


def _citation_keys(papers: list[Paper]) -> dict[str, str]:
    keys: dict[str, str] = {}
    used: set[str] = set()
    for paper in papers:
        base = _base_citation_key(paper)
        key = base
        index = 2
        while key in used:
            key = f"{base}{index}"
            index += 1
        used.add(key)
        keys[paper.id or paper.title] = key
    return keys


def _base_citation_key(paper: Paper) -> str:
    author = "unknown"
    if paper.authors:
        author = re.sub(r"[^A-Za-z0-9]", "", paper.authors[0].split()[-1]).lower() or "unknown"
    year = str(paper.year or "nd")
    title_word = next((token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", paper.title) if len(token) > 3), "paper")
    return f"{author}{year}{title_word}"


def _bibtex_for(paper: Paper, key: str) -> str:
    # A retrieval source (for example DBLP or OpenAlex) is not a publication
    # venue, so do not turn it into a misleading journal field.
    venue = _complete_metadata_value(paper.venue)
    conference_markers = ("conference", "proceedings", "iclr", "neurips", "icml", "aaai", "ijcai", "acl", "emnlp", "kdd", "sigir", "recsys", "www", "cvpr", "iccv", "eccv")
    is_conference = any(marker in (venue or "").lower() for marker in conference_markers)
    entry_type = "inproceedings" if is_conference else "article"
    container_key = "booktitle" if is_conference else "journal"
    authors = [author for author in paper.authors if _complete_metadata_value(author)]
    doi = paper.doi or _doi_from_url(paper.source_url)
    fields = {
        "title": paper.title,
        "author": " and ".join(authors),
        "year": str(paper.year or ""),
        container_key: venue,
        "doi": doi or "",
        "url": paper.source_url or paper.pdf_url or "",
    }
    metadata_note = _bibtex_metadata_note(paper, authors, venue)
    if metadata_note:
        fields["note"] = metadata_note
    lines = [f"@{entry_type}{{{key},"]
    for field, value in fields.items():
        if value:
            lines.append(f"  {field} = {{{_bibtex_escape(str(value))}}},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


def _bibtex_escape(value: str) -> str:
    return value.replace("\\", "\\textbackslash{}").replace("{", "").replace("}", "").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#").replace("_", "\\_")


def _complete_metadata_value(value: str | None) -> str:
    text = (value or "").strip()
    return "" if "…" in text or "..." in text else text


def _doi_from_url(url: str | None) -> str:
    match = re.search(r"(?:doi(?:/abs)?/|doi\.org/)(10\.\d{4,9}/[^?#\s]+)", url or "", flags=re.I)
    return match.group(1).rstrip(".,)") if match else ""


def _citation_metadata_issues(papers: list[Paper]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for paper in papers:
        fields = []
        if not paper.authors or any(not _complete_metadata_value(author) for author in paper.authors):
            fields.append("authors")
        if paper.venue and not _complete_metadata_value(paper.venue):
            fields.append("venue")
        if fields:
            issues.append({"paper_id": paper.id or "", "title": paper.title, "fields": ",".join(fields)})
    return issues


def _bibtex_metadata_note(paper: Paper, authors: list[str], venue: str) -> str:
    issues = []
    if len(authors) != len(paper.authors):
        issues.append("author list")
    if paper.venue and not venue:
        issues.append("venue")
    if not issues:
        return ""
    return f"Source metadata has an incomplete {' and '.join(issues)}; verify before submission."


def _paragraph(text: str) -> str:
    lines = []
    for block in text.split("\n\n"):
        if block.startswith("#"):
            lines.append(block)
        else:
            lines.append(fill(re.sub(r"\s+", " ", block).strip(), width=100))
    return "\n\n".join(lines)


def _cites(citation_keys: dict[str, str], papers: list[Paper]) -> str:
    keys = [citation_keys.get(paper.id or paper.title) for paper in papers]
    values = [key for key in keys if key]
    return f"\\cite{{{','.join(values)}}}" if values else ""


def _named_papers(papers: list[Paper]) -> str:
    return "；".join(f"《{paper.title}》" for paper in papers if paper.title)


def _no_evidence_text(language: str) -> str:
    return "当前没有可用于写作的证据论文。" if language.lower().startswith("zh") else "No evidence papers are currently available for drafting."
