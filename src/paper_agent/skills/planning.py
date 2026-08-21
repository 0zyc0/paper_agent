from __future__ import annotations


TOOL_SKILLS = {
    "pdf_read": {
        "description": "Read uploaded PDF excerpts and answer or summarize only from those pages.",
        "requires": "an uploaded PDF in the current session",
    },
    "paper_search": {
        "description": "Search fresh CS literature from local cache, DBLP, arXiv, Semantic Scholar, and Google Scholar.",
        "requires": "a research topic and optional time or venue filters",
    },
    "evidence_answer": {
        "description": "Answer a question from the current retrieved paper pool without new retrieval.",
        "requires": "retrieved papers in the current session",
    },
    "paper_fulltext_read": {
        "description": "Download/cache an open-access PDF for a retrieved paper, read local full-text chunks, and answer detailed questions from the paper text.",
        "requires": "a retrieved paper with an open PDF URL or existing local full-text cache",
    },
    "write_document": {
        "description": "Write or export a requested research section, survey, related work, report, or bibliography from available evidence.",
        "requires": "uploaded PDF evidence, retrieved papers, or paper_search earlier in the plan",
    },
    "document_inspect": {
        "description": "Read the latest generated Markdown document and answer questions about its actual content, completeness, or an empty-looking result.",
        "requires": "a generated document in the current session",
    },
    "free_chat": {
        "description": "Handle general conversation that does not require research evidence.",
        "requires": "none",
    },
}


def tool_skills_prompt() -> str:
    lines = []
    for name, skill in TOOL_SKILLS.items():
        lines.append(f"- {name}: {skill['description']} Requires: {skill['requires']}.")
    return "\n".join(lines)


def clean_tool_plan(value, *, has_documents: bool, has_generated_document: bool = False) -> list[str]:
    allowed = list(TOOL_SKILLS)
    raw_plan = value if isinstance(value, list) else []
    plan = []
    for item in raw_plan:
        name = str(item).strip().lower()
        if name in allowed and name not in plan:
            plan.append(name)
    if not has_documents:
        plan = [name for name in plan if name != "pdf_read"]
    if not has_generated_document:
        plan = [name for name in plan if name != "document_inspect"]
    return plan
