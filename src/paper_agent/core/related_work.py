from __future__ import annotations

from dataclasses import dataclass
import re
from textwrap import fill
from typing import Any

from ..tools.kimi_writer import KimiRelatedWorkWriter
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
        ("研究范围", "证据综合", "结论与后续工作"),
        ("Research scope", "Evidence synthesis", "Conclusions and next steps"),
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

        if plan.kind == "bibliography":
            content = _bibliography_note(plan, selected, citation_keys, language)
        elif use_llm and selected:
            writer = KimiRelatedWorkWriter()
            if writer.available:
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
                except Exception:
                    content = ""

        if not content:
            content = self._fallback_draft(plan, query, selected, citation_keys, language)

        claim_map = _claim_map_from_content(content, selected, citation_keys, evidence_notes)
        quality_report = _quality_report(content, selected, citation_keys, evidence_notes, plan)
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


def writing_plan_for(request: str) -> WritingPlan:
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
    return {"kind": plan.kind, "title": plan.title_for(language), "outline": plan.outline_for(language)}


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


def _section_fallback(plan: WritingPlan, query: str, papers: list[Paper], citation_keys: dict[str, str], language: str) -> str:
    title = plan.title_for(language)
    headings = plan.outline_for(language)
    groups = [papers[:5], papers[5:10], papers[10:15]]
    paragraphs: list[str] = []
    for heading, group in zip(headings, groups, strict=False):
        evidence = group or papers[:5]
        citations = _cites(citation_keys, evidence)
        if language.lower().startswith("zh"):
            text = (
                f"### {heading}\n\n围绕 {query}，本节基于当前证据整理可确认的研究线索。"
                f"其中 {_named_papers(evidence[:3]) or '现有记录'} 为该部分提供定位依据 {citations}。"
                "涉及具体机制、实验提升或局限性时，仍应回到原文逐项核查。"
            )
        else:
            text = (
                f"### {heading}\n\nFor {query}, this section synthesizes the research signals supported by the current evidence set. "
                f"{_named_papers(evidence[:3]) or 'The available records'} provide the immediate positioning for this part {citations}. "
                "Detailed mechanisms, empirical gains, and limitations should be verified against the original papers."
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
            lines.extend([
                f"### {family}", "",
                f"该路线在当前文献中包括 {_named_papers(group[:3])}。"
                "这些工作构成一组可比较的建模视角；涉及具体算法细节、指标提升或适用边界时，"
                f"仍应以原文为准。 {_cites(citation_keys, group)}", "",
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
            lines.extend([
                f"### {family}", "",
                f"This line is represented by {_named_papers(group[:3])}. The available records support positioning these papers "
                f"as a comparable modelling perspective, while detailed mechanisms and empirical claims require full-text verification. {_cites(citation_keys, group)}", "",
            ])
        lines.extend([
            f"### {headings[-1]}", "",
            "The current library spans several research lines, but their bias definitions, distributional assumptions, and evaluation protocols may differ. "
            f"A final survey should compare those dimensions against the original papers. {_cites(citation_keys, papers[-min(6, len(papers)):])}",
        ])
    return "\n".join(lines).strip() + "\n"


def _survey_family(paper: Paper) -> str:
    text = f"{paper.title} {paper.abstract}".lower()
    if any(token in text for token in ("causal", "counterfactual", "propensity", "ips", "inverse propensity")):
        return "因果与反事实校正"
    if any(token in text for token in ("fairness", "fair exposure", "popularity bias", "exposure bias", "debias")):
        return "公平性、曝光与偏差校正"
    if any(token in text for token in ("large language model", "llm", "language model", "generative")):
        return "大模型增强的推荐建模"
    if any(token in text for token in ("sequential", "session", "temporal", "dynamic", "streaming")):
        return "时序与动态推荐场景"
    return "推荐建模与评测"


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
        status = "local_fulltext" if levels and all(level == "local_fulltext" for level in levels) else "mixed_evidence"
        claims.append(
            {
                "claim": re.sub(r"\\cite\{[^}]+\}", "", clean).replace("\n", " ").strip()[:500],
                "paper_ids": list(dict.fromkeys(paper_ids)),
                "evidence_status": status,
            }
        )
    if claims:
        return claims
    return [{"claim": "The generated draft is limited to the selected evidence set.", "paper_ids": [paper.id for paper in papers], "evidence_status": "metadata_only"}] if papers else []


def _quality_report(
    content: str,
    papers: list[Paper],
    citation_keys: dict[str, str],
    evidence_notes: list[dict] | None,
    plan: WritingPlan,
) -> dict[str, Any]:
    used = set(_citation_keys_in_text(content))
    evidence = [item for item in evidence_notes or [] if isinstance(item, dict)]
    fulltext_count = sum(item.get("evidence_level") == "local_fulltext" for item in evidence)
    weak_count = sum(item.get("evidence_level") in {"metadata_only", "downloadable_metadata", "local_pdf_without_text"} for item in evidence)
    warnings: list[str] = []
    if weak_count:
        warnings.append(f"{weak_count} 篇论文缺少可用全文，正文中的细节性结论应在投稿前核对原文。")
    if papers and len(used) < min(2, len(papers)) and plan.kind not in {"outline", "bibliography"}:
        warnings.append("当前草稿引用覆盖较少，建议补充或筛选证据论文。")
    return {
        "writing_kind": plan.kind,
        "word_count": len(re.sub(r"\s+", "", re.sub(r"[#*_`>-]", "", content))),
        "citation_uses": len(_citation_keys_in_text(content)),
        "cited_paper_count": len(used),
        "available_paper_count": len(papers),
        "fulltext_paper_count": fulltext_count,
        "weak_evidence_count": weak_count,
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
    venue = paper.venue or paper.source
    conference_markers = ("conference", "proceedings", "iclr", "neurips", "icml", "aaai", "ijcai", "acl", "emnlp", "kdd", "sigir", "www", "cvpr", "iccv", "eccv")
    is_conference = any(marker in (venue or "").lower() for marker in conference_markers)
    entry_type = "inproceedings" if is_conference else "article"
    container_key = "booktitle" if is_conference else "journal"
    fields = {
        "title": paper.title,
        "author": " and ".join(paper.authors) if paper.authors else "Unknown",
        "year": str(paper.year or ""),
        container_key: venue,
        "doi": paper.doi or "",
        "url": paper.source_url or paper.pdf_url or "",
    }
    lines = [f"@{entry_type}{{{key},"]
    for field, value in fields.items():
        if value:
            lines.append(f"  {field} = {{{_bibtex_escape(str(value))}}},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


def _bibtex_escape(value: str) -> str:
    return value.replace("\\", "\\textbackslash{}").replace("{", "").replace("}", "").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")


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
