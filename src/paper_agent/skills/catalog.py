from __future__ import annotations

"""Skill discovery for the paper assistant.

Skills are versioned workflow instructions.  They describe how a model and a
tool chain should work together, while tools remain the only place that can
perform side effects such as retrieval, file reading, and document creation.
"""

from dataclasses import dataclass
from pathlib import Path


SKILL_ROOT = Path(__file__).with_name("definitions")
ACADEMIC_WRITING_ROOT = SKILL_ROOT / "academic-writing"


@dataclass(frozen=True)
class WritingSkillRoute:
    """Resolved resource bundle for the one academic-writing skill."""

    deliverable: str
    resource: str
    instruction: str
    routing: str
    markdown_contract: str

    def payload(self) -> dict:
        return {
            "skill": "academic-writing",
            "deliverable": self.deliverable,
            "resource": f"resources/{self.resource}.md",
        }


class AcademicWritingSkillRouter:
    """Navigate one Skill.md to its resource-level writing instruction.

    Mirrors Mydex's structured-response pattern: a routing resource defines the
    stable branches, while the runtime injects only the selected branch plus the
    shared Markdown contract into the writer prompt.
    """

    _RESOURCE_BY_DELIVERABLE = {
        "report": "report",
        "survey": "survey",
        "related_work": "related-work",
        "introduction": "introduction",
        "method_section": "method",
        "experiment_section": "experiment",
        "summary": "summary",
        "outline": "outline",
        "bibliography": "bibliography",
        "bibtex": "bibliography",
    }

    def __init__(self, root: Path = ACADEMIC_WRITING_ROOT) -> None:
        self.root = root
        self.resources = root / "resources"

    def route(self, deliverable: str) -> WritingSkillRoute:
        resource = self._RESOURCE_BY_DELIVERABLE.get(deliverable, "general")
        return WritingSkillRoute(
            deliverable=deliverable or "general",
            resource=resource,
            instruction=self._read(f"{resource}.md"),
            routing=self._read("writing-routing.md"),
            markdown_contract=self._read("markdown-contract.md"),
        )

    def _read(self, name: str) -> str:
        try:
            return (self.resources / name).read_text(encoding="utf-8").strip()
        except OSError:
            return ""


@dataclass(frozen=True)
class PaperSkill:
    name: str
    title: str
    description: str
    tools: tuple[str, ...]
    path: Path
    variant: str = ""

    def prompt(self) -> str:
        try:
            router_prompt = self.path.read_text(encoding="utf-8").strip()
        except OSError:
            router_prompt = self.description
        if not self.variant:
            return router_prompt
        try:
            variant_prompt = (self.path.parent / "resources" / f"{self.variant}.md").read_text(encoding="utf-8").strip()
            routing_prompt = (self.path.parent / "resources" / "writing-routing.md").read_text(encoding="utf-8").strip()
            contract_prompt = (self.path.parent / "resources" / "markdown-contract.md").read_text(encoding="utf-8").strip()
        except OSError:
            return router_prompt
        return (
            f"{router_prompt}\n\n## 写作路由\n\n{routing_prompt}"
            f"\n\n## 当前写作分支：{self.variant}\n\n{variant_prompt}"
            f"\n\n## 共同 Markdown 契约\n\n{contract_prompt}"
        ).strip()

    def payload(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "tools": list(self.tools),
            "variant": self.variant or None,
            "instruction_file": f"academic-writing/resources/{self.variant}.md" if self.variant else "academic-writing/SKILL.md",
        }


class PaperSkillCatalog:
    """Select the smallest set of paper-domain workflows for a compiled plan."""

    def __init__(self, root: Path = SKILL_ROOT) -> None:
        self.root = root
        self.writing_router = AcademicWritingSkillRouter(root / "academic-writing")
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
    def for_plan(self, tools: list[str], *, category: str = "", deliverable: str = "") -> list[PaperSkill]:
        wanted: list[str] = []
        if "paper_search" in tools:
            wanted.append("literature-research")
        if "pdf_read" in tools or "paper_fulltext_read" in tools:
            wanted.append("pdf-reading")
        if "evidence_answer" in tools:
            wanted.append("evidence-grounding")
        selected: list[PaperSkill] = []
        for name in wanted:
            skill = self._skills.get(name)
            if skill:
                selected.append(skill)
        if "write_document" in tools or "document_inspect" in tools:
            selected.append(self._academic_writing_skill(deliverable))
        if category == "discovery":
            skill = self._skills.get("research-discovery")
            if skill:
                selected.append(skill)
        return selected

    def _academic_writing_skill(self, deliverable: str) -> PaperSkill:
        base = self._skills["academic-writing"]
        variant = self.writing_router.route(deliverable).resource
        return PaperSkill(
            name=base.name,
            title=base.title,
            description=base.description,
            tools=base.tools,
            path=base.path,
            variant=variant,
        )

    def get(self, name: str) -> PaperSkill | None:
        return self._skills.get(name)
