from __future__ import annotations

"""Skill discovery for the paper assistant.

Skills are versioned workflow instructions.  They describe how a model and a
tool chain should work together, while tools remain the only place that can
perform side effects such as retrieval, file reading, and document creation.
"""

from dataclasses import dataclass
from pathlib import Path


SKILL_ROOT = Path(__file__).with_name("definitions")


@dataclass(frozen=True)
class PaperSkill:
    name: str
    title: str
    description: str
    tools: tuple[str, ...]
    path: Path

    def prompt(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return self.description

    def payload(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "tools": list(self.tools),
        }


class PaperSkillCatalog:
    """Select the smallest set of paper-domain workflows for a compiled plan."""

    def __init__(self, root: Path = SKILL_ROOT) -> None:
        self.root = root
        self._skills = {
            "literature-research": PaperSkill(
                "literature-research",
                "论文检索",
                "将研究问题转成可验证的检索约束，并汇总多来源论文。",
                ("paper_search",),
                root / "literature-research" / "SKILL.md",
            ),
            "evidence-grounding": PaperSkill(
                "evidence-grounding",
                "证据问答",
                "只根据当前证据池回答，并明确证据不足之处。",
                ("evidence_answer", "paper_fulltext_read"),
                root / "evidence-grounding" / "SKILL.md",
            ),
            "pdf-reading": PaperSkill(
                "pdf-reading",
                "论文精读",
                "从上传或本地缓存的 PDF 提取可追溯的页码证据。",
                ("pdf_read", "paper_fulltext_read"),
                root / "pdf-reading" / "SKILL.md",
            ),
            "academic-writing": PaperSkill(
                "academic-writing",
                "学术写作",
                "依据当前证据生成具有结构、引用边界和格式约束的草稿。",
                ("write_document", "document_inspect"),
                root / "academic-writing" / "SKILL.md",
            ),
            "research-discovery": PaperSkill(
                "research-discovery",
                "研究发现",
                "基于用户长期研究画像推荐可验证的最新进展与方向。",
                ("discovery_feed",),
                root / "research-discovery" / "SKILL.md",
            ),
        }

    def for_plan(self, tools: list[str], *, category: str = "") -> list[PaperSkill]:
        wanted: list[str] = []
        if "paper_search" in tools:
            wanted.append("literature-research")
        if "pdf_read" in tools or "paper_fulltext_read" in tools:
            wanted.append("pdf-reading")
        if "evidence_answer" in tools:
            wanted.append("evidence-grounding")
        if "write_document" in tools or "document_inspect" in tools:
            wanted.append("academic-writing")
        if category == "discovery":
            wanted.append("research-discovery")
        return [self._skills[name] for name in wanted if name in self._skills]

    def get(self, name: str) -> PaperSkill | None:
        return self._skills.get(name)
