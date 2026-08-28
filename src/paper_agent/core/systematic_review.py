from __future__ import annotations

"""Project-scoped screening records and exports for systematic reviews."""

import csv
import io
from pathlib import Path

from .models import Paper


SCREENING_STAGES = ("title_abstract", "full_text")
SCREENING_DECISIONS = ("include", "exclude", "pending")


def review_snapshot(*, papers: list[Paper], protocol: dict | None, screenings: list[dict]) -> dict:
    protocol = protocol or {}
    decisions = {str(item.get("paper_id")): item for item in screenings}
    rows = []
    for paper in papers:
        decision = decisions.get(str(paper.id), {})
        rows.append(
            {
                "paper_id": paper.id,
                "title": paper.title,
                "authors": "; ".join(paper.authors),
                "year": paper.year or "",
                "venue": paper.venue or "",
                "doi": paper.doi or "",
                "url": paper.source_url or paper.pdf_url or "",
                "abstract": paper.abstract or "",
                "fulltext_status": paper.fulltext_status,
                "stage": decision.get("stage") or "title_abstract",
                "decision": decision.get("decision") or "pending",
                "reason": decision.get("reason") or "",
                "updated_at": decision.get("updated_at") or "",
            }
        )
    summary = prisma_summary(rows)
    return {"protocol": protocol, "rows": rows, "summary": summary}


def prisma_summary(rows: list[dict]) -> dict:
    title_excluded = sum(1 for row in rows if row["stage"] == "title_abstract" and row["decision"] == "exclude")
    full_text_excluded = sum(1 for row in rows if row["stage"] == "full_text" and row["decision"] == "exclude")
    included = sum(1 for row in rows if row["decision"] == "include")
    screened = sum(1 for row in rows if row["decision"] != "pending")
    return {
        "identified": len(rows),
        "screened": screened,
        "pending": len(rows) - screened,
        "title_abstract_excluded": title_excluded,
        "full_text_excluded": full_text_excluded,
        "included": included,
    }


def evidence_csv(rows: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    fields = [
        "paper_id", "title", "authors", "year", "venue", "doi", "url", "abstract",
        "fulltext_status", "stage", "decision", "reason", "updated_at",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])
    return buffer.getvalue()


def prisma_markdown(protocol: dict, summary: dict) -> str:
    inclusion = protocol.get("inclusion_criteria") or []
    exclusion = protocol.get("exclusion_criteria") or []
    question = protocol.get("research_question") or "尚未设置研究问题"
    strategy = protocol.get("search_strategy") or "尚未设置检索策略"
    return "\n".join(
        [
            "# Systematic Review Protocol and PRISMA Summary",
            "",
            "## Research Question",
            question,
            "",
            "## Search Strategy",
            strategy,
            "",
            "## Inclusion Criteria",
            *([f"- {item}" for item in inclusion] or ["- 尚未设置"]),
            "",
            "## Exclusion Criteria",
            *([f"- {item}" for item in exclusion] or ["- 尚未设置"]),
            "",
            "## PRISMA Flow Summary",
            f"- Identified records: {summary['identified']}",
            f"- Screened records: {summary['screened']}",
            f"- Pending screening: {summary['pending']}",
            f"- Excluded at title/abstract: {summary['title_abstract_excluded']}",
            f"- Excluded at full text: {summary['full_text_excluded']}",
            f"- Included studies: {summary['included']}",
            "",
        ]
    )


def write_review_export(*, snapshot: dict, format: str, path: Path) -> dict:
    normalized = str(format or "evidence_csv").strip().lower().replace("-", "_")
    path.parent.mkdir(parents=True, exist_ok=True)
    if normalized in {"evidence_csv", "csv"}:
        path.write_text(evidence_csv(snapshot["rows"]), encoding="utf-8-sig")
        kind = "evidence_csv"
    elif normalized in {"prisma", "prisma_markdown", "markdown"}:
        path.write_text(prisma_markdown(snapshot["protocol"], snapshot["summary"]), encoding="utf-8")
        kind = "prisma_markdown"
    else:
        raise ValueError(f"不支持的系统综述导出格式：{format}")
    return {"name": path.name, "path": str(path), "kind": kind}
