from zipfile import ZipFile

import pytest

from paper_agent.core.draft_exports import DraftExportError, export_draft
from paper_agent.core.models import Paper


def _paper() -> Paper:
    return Paper(
        title="Evidence-Grounded Recommendation",
        authors=["Ada Lovelace", "Grace Hopper"],
        abstract="A study of evidence-grounded recommendation.",
        year=2026,
        venue="RecSys",
        source="dblp",
        source_url="https://example.org/evidence",
        doi="10.1000/evidence",
    )


def _draft() -> dict:
    return {
        "title": "Research Report",
        "writing_kind": "research_report",
        "paper_ids": [_paper().id],
        "content_markdown": "# Introduction\n\nThis claim is grounded in evidence \\cite{lovelace2026evidence}.",
        "bibtex": "@article{lovelace2026evidence,\n  title={Evidence-Grounded Recommendation},\n  author={Lovelace, Ada},\n  year={2026}\n}\n",
    }


@pytest.mark.parametrize(
    ("format_name", "filename"),
    [
        ("markdown", "draft.md"),
        ("bibtex", "draft.bib"),
        ("latex", "draft.tex"),
        ("ris", "draft.ris"),
        ("csl_json", "draft.json"),
        ("docx", "draft.docx"),
    ],
)
def test_draft_exports_cover_standard_manuscript_formats(tmp_path, format_name, filename):
    path = tmp_path / filename

    result = export_draft(draft=_draft(), papers=[_paper()], format=format_name, path=path)

    assert result["path"] == str(path)
    assert path.exists()
    if format_name == "docx":
        with ZipFile(path) as archive:
            assert "word/document.xml" in archive.namelist()
            assert "References" in archive.read("word/document.xml").decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
        assert text
        if format_name == "latex":
            assert "\\cite{lovelace2026evidence}" in text
        if format_name == "ris":
            assert "TI  - Evidence-Grounded Recommendation" in text
        if format_name == "csl_json":
            assert '"DOI": "10.1000/evidence"' in text


def test_draft_export_refuses_unknown_citation_keys(tmp_path):
    draft = _draft()
    draft["content_markdown"] = "A claim \\cite{missing2026}."

    with pytest.raises(DraftExportError, match="missing2026"):
        export_draft(draft=draft, papers=[_paper()], format="docx", path=tmp_path / "draft.docx")
