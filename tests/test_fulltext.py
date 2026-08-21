from paper_agent.core import fulltext as fulltext_module
from paper_agent.core.fulltext import PaperFullTextManager, _open_pdf_candidates
from paper_agent.core.models import Paper


def test_open_pdf_candidates_include_arxiv_alternative():
    paper = Paper(
        title="A Paper",
        authors=["A"],
        source="openalex",
        source_url="https://arxiv.org/abs/2401.01234",
        pdf_url="https://publisher.example.com/paper.pdf",
    )

    candidates = _open_pdf_candidates(paper)

    assert candidates == [
        "https://publisher.example.com/paper.pdf",
        "https://arxiv.org/pdf/2401.01234.pdf",
    ]


def test_pdf_cache_uses_public_arxiv_when_first_source_rejects_request(monkeypatch, tmp_path):
    paper = Paper(
        title="A Paper",
        authors=["A"],
        source="openalex",
        source_url="https://arxiv.org/abs/2401.01234",
        pdf_url="https://publisher.example.com/paper.pdf",
    )
    requested_urls = []

    def fake_download(url, *, referer=None):
        requested_urls.append(url)
        if "publisher" in url:
            raise RuntimeError("HTTP 403 when downloading PDF")
        return b"%PDF-1.4\nplaceholder"

    monkeypatch.setattr(fulltext_module, "_download_pdf", fake_download)
    monkeypatch.setattr(PaperFullTextManager, "_extract_to_cache", lambda self, current, data: [])

    result = PaperFullTextManager(tmp_path).cache_pdf(paper)

    assert result.downloaded is True
    assert requested_urls[-1] == "https://arxiv.org/pdf/2401.01234.pdf"
    assert paper.local_pdf_path
