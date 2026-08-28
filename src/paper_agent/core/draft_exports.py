from __future__ import annotations

"""Portable manuscript exports for the local-first web application."""

import json
from pathlib import Path
import re
from xml.sax.saxutils import escape as xml_escape
from zipfile import ZIP_DEFLATED, ZipFile

from .models import Paper


_CITATION_PATTERN = re.compile(r"\\cite\{([^}]+)\}")
_BIB_KEY_PATTERN = re.compile(r"@\w+\s*\{\s*([^,\s]+)", flags=re.I)


class DraftExportError(ValueError):
    pass


def export_draft(*, draft: dict, papers: list[Paper], format: str, path: Path) -> dict:
    """Validate citations before emitting a Word, LaTeX, or reference artefact."""
    normalized = str(format or "markdown").strip().lower().replace("-", "_")
    content = str(draft.get("content_markdown") or "")
    bibtex = str(draft.get("bibtex") or "")
    _validate_citations(content, bibtex, writing_kind=str(draft.get("writing_kind") or ""))
    path.parent.mkdir(parents=True, exist_ok=True)

    if normalized == "markdown":
        path.write_text(content, encoding="utf-8")
        return _result(path, "markdown")
    if normalized == "bibtex":
        path.write_text(bibtex, encoding="utf-8")
        return _result(path, "bibtex")
    if normalized in {"latex", "tex"}:
        path.write_text(_latex_document(draft, papers), encoding="utf-8")
        return _result(path, "latex")
    if normalized == "ris":
        path.write_text(_ris(papers), encoding="utf-8")
        return _result(path, "ris")
    if normalized in {"csl", "csl_json", "csljson"}:
        path.write_text(json.dumps(_csl_json(papers), ensure_ascii=False, indent=2), encoding="utf-8")
        return _result(path, "csl_json")
    if normalized == "docx":
        _write_docx(path, draft, papers)
        return _result(path, "docx")
    raise DraftExportError(f"不支持的导出格式：{format}")


def _validate_citations(content: str, bibtex: str, *, writing_kind: str) -> None:
    used = [key.strip() for group in _CITATION_PATTERN.findall(content) for key in group.split(",") if key.strip()]
    allowed = set(_BIB_KEY_PATTERN.findall(bibtex))
    unknown = sorted(set(used) - allowed)
    if unknown:
        raise DraftExportError("草稿含有未在当前 BibTeX 中定义的引用键：" + ", ".join(unknown))
    if writing_kind not in {"outline", "bibliography"} and used and not bibtex.strip():
        raise DraftExportError("草稿使用了引用，但缺少 BibTeX 参考文献。")


def _result(path: Path, kind: str) -> dict:
    return {"name": path.name, "path": str(path), "kind": kind}


def _latex_document(draft: dict, papers: list[Paper]) -> str:
    lines = ["\\documentclass[11pt]{article}", "\\begin{document}", f"\\title{{{_latex_escape(str(draft.get('title') or 'Untitled Draft'))}}}", "\\maketitle", ""]
    for line in str(draft.get("content_markdown") or "").splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
        elif stripped.startswith("### "):
            lines.append(f"\\subsection{{{_latex_escape(stripped[4:])}}}")
        elif stripped.startswith(("## ", "# ")):
            lines.append(f"\\section{{{_latex_escape(stripped.lstrip('# ').strip())}}}")
        else:
            lines.append(_latex_escape(stripped))
    if papers:
        lines.append("\\section*{References}")
        lines.extend(_latex_escape(_reference_text(paper)) + "\\par" for paper in papers)
    return "\n".join(lines + ["\\end{document}", ""])


def _latex_escape(value: str) -> str:
    saved: list[str] = []

    def protect(match: re.Match[str]) -> str:
        saved.append(match.group(0))
        return f"CITATIONTOKEN{len(saved) - 1}"

    text = _CITATION_PATTERN.sub(protect, value)
    for source, replacement in {"\\": r"\\textbackslash{}", "&": r"\\&", "%": r"\\%", "$": r"\\$", "#": r"\\#", "_": r"\\_", "{": r"\\{", "}": r"\\}"}.items():
        text = text.replace(source, replacement)
    for index, citation in enumerate(saved):
        text = text.replace(f"CITATIONTOKEN{index}", citation)
    return text


def _ris(papers: list[Paper]) -> str:
    records: list[str] = []
    for paper in papers:
        lines = ["TY  - JOUR", *[f"AU  - {author}" for author in paper.authors], f"TI  - {paper.title}"]
        if paper.year:
            lines.append(f"PY  - {paper.year}")
        if paper.venue:
            lines.append(f"JO  - {paper.venue}")
        if paper.doi:
            lines.append(f"DO  - {paper.doi}")
        if paper.source_url:
            lines.append(f"UR  - {paper.source_url}")
        if paper.abstract:
            lines.append(f"AB  - {paper.abstract}")
        records.append("\n".join(lines + ["ER  - "]))
    return "\n\n".join(records) + ("\n" if records else "")


def _csl_json(papers: list[Paper]) -> list[dict]:
    records = []
    for paper in papers:
        item = {
            "id": paper.id or paper.title,
            "type": "article-journal",
            "title": paper.title,
            "author": [{"literal": author} for author in paper.authors],
            "container-title": paper.venue or "",
            "DOI": paper.doi or "",
            "URL": paper.source_url or "",
            "abstract": paper.abstract or "",
        }
        if paper.year:
            item["issued"] = {"date-parts": [[paper.year]]}
        records.append(item)
    return records


def _write_docx(path: Path, draft: dict, papers: list[Paper]) -> None:
    paragraphs = [("Title", str(draft.get("title") or "Untitled Draft"))]
    for line in str(draft.get("content_markdown") or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("### "):
            paragraphs.append(("Heading2", stripped[4:]))
        elif stripped.startswith(("## ", "# ")):
            paragraphs.append(("Heading1", stripped.lstrip("# ").strip()))
        else:
            paragraphs.append(("Normal", stripped))
    if papers:
        paragraphs.append(("Heading1", "References"))
        paragraphs.extend(("Normal", _reference_text(paper)) for paper in papers)
    body = "".join(f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr><w:r><w:t xml:space="preserve">{xml_escape(text)}</w:t></w:r></w:p>' for style, text in paragraphs)
    document = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{body}<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body></w:document>'
    styles = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style></w:styles>'
    content_types = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
    root_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
    doc_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", styles)
        archive.writestr("word/_rels/document.xml.rels", doc_rels)


def _reference_text(paper: Paper) -> str:
    authors = ", ".join(paper.authors) or "Unknown author"
    year = str(paper.year) if paper.year else "n.d."
    venue = f". {paper.venue}" if paper.venue else ""
    url = f". {paper.source_url}" if paper.source_url else ""
    return f"{authors} ({year}). {paper.title}{venue}{url}"
