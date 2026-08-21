from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import ssl
import urllib.error
import urllib.request

from .models import Paper, utc_now_iso
from .pdf_reader import PdfChunk, PdfExtraction, PdfExtractionError, extract_pdf_text


MAX_FULLTEXT_PDF_BYTES = 30 * 1024 * 1024


@dataclass
class PaperFullTextResult:
    paper: Paper
    chunks: list[PdfChunk]
    downloaded: bool
    extracted: bool
    summary: str


class PaperFullTextManager:
    """Cache open full-text PDFs locally and keep SQLite as a metadata index."""

    def __init__(self, root_dir: str | Path = "data") -> None:
        root = Path(root_dir)
        self.pdf_dir = root / "paper_files" / "pdf"
        self.text_dir = root / "paper_texts"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)

    def cache_pdf(self, paper: Paper, *, extract_text: bool = True) -> PaperFullTextResult:
        """Download or reuse a paper PDF in the local personal library."""
        if paper.local_pdf_path and Path(paper.local_pdf_path).exists():
            if extract_text and not paper.local_text_path:
                data = Path(paper.local_pdf_path).read_bytes()
                chunks = self._extract_to_cache(paper, data)
                if chunks:
                    return PaperFullTextResult(paper, chunks, downloaded=False, extracted=True, summary="PDF 已在个人文献库中，已抽取全文文本。")
                paper.fulltext_status = "pdf_cached"
                return PaperFullTextResult(paper, [], downloaded=False, extracted=False, summary=f"PDF 已在个人文献库中，但文本抽取失败：{paper.fulltext_error or '未知错误'}")
            chunks = self.read_chunks(paper) if extract_text else []
            return PaperFullTextResult(paper, chunks, downloaded=False, extracted=bool(chunks), summary="PDF 已在个人文献库中。")

        if not _open_pdf_candidates(paper):
            paper.fulltext_status = "none"
            paper.fulltext_error = "当前论文没有可直接下载的开放 PDF 链接。"
            return PaperFullTextResult(paper, [], downloaded=False, extracted=False, summary=paper.fulltext_error)

        try:
            data = _download_open_pdf(paper)
            self._write_pdf_to_cache(paper, data)
            if not extract_text:
                paper.fulltext_status = "pdf_cached"
                paper.fulltext_error = None
                return PaperFullTextResult(paper, [], downloaded=True, extracted=False, summary="已下载 PDF 到个人文献库。")
            chunks = self._extract_to_cache(paper, data)
            if chunks:
                return PaperFullTextResult(paper, chunks, downloaded=True, extracted=True, summary="已下载开放 PDF 并抽取全文文本。")
            paper.fulltext_status = "pdf_cached"
            return PaperFullTextResult(paper, [], downloaded=True, extracted=False, summary="已下载 PDF 到个人文献库，但全文文本抽取失败。")
        except Exception as exc:
            paper.fulltext_status = "failed"
            paper.fulltext_error = str(exc)[:500]
            return PaperFullTextResult(paper, [], downloaded=False, extracted=False, summary=f"全文缓存失败：{paper.fulltext_error}")

    def ensure_text(self, paper: Paper) -> PaperFullTextResult:
        if paper.local_text_path:
            chunks = self.read_chunks(paper)
            if chunks:
                paper.fulltext_status = "extracted"
                return PaperFullTextResult(paper, chunks, downloaded=False, extracted=False, summary="已使用本地全文文本缓存。")

        if paper.local_pdf_path and Path(paper.local_pdf_path).exists():
            data = Path(paper.local_pdf_path).read_bytes()
            chunks = self._extract_to_cache(paper, data)
            if chunks:
                return PaperFullTextResult(paper, chunks, downloaded=False, extracted=True, summary="已从本地 PDF 抽取全文文本。")
            paper.fulltext_status = "pdf_cached"
            return PaperFullTextResult(paper, [], downloaded=False, extracted=False, summary=f"PDF 已在个人文献库中，但文本抽取失败：{paper.fulltext_error or '未知错误'}")

        return self.cache_pdf(paper, extract_text=True)

    def attach_uploaded_pdf(
        self,
        paper: Paper,
        data: bytes,
        *,
        extraction: PdfExtraction | None = None,
    ) -> PaperFullTextResult:
        """Attach a user-uploaded PDF to a retrieved paper and cache its text."""
        self._write_pdf_to_cache(paper, data)
        try:
            extraction = extraction or extract_pdf_text(data)
        except PdfExtractionError as exc:
            paper.fulltext_status = "pdf_cached"
            paper.fulltext_error = str(exc)
            return PaperFullTextResult(
                paper,
                [],
                downloaded=False,
                extracted=False,
                summary=f"PDF 已保存到个人文献库，但文本抽取失败：{exc}",
            )
        chunks = self._write_text_to_cache(paper, extraction)
        return PaperFullTextResult(
            paper,
            chunks,
            downloaded=False,
            extracted=bool(chunks),
            summary="已上传 PDF、保存到个人文献库，并抽取全文文本。",
        )

    def read_chunks(self, paper: Paper) -> list[PdfChunk]:
        path = Path(paper.local_text_path or "")
        if not path.exists() or not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        chunks = []
        for item in payload.get("chunks", []):
            try:
                chunks.append(PdfChunk(text=str(item["text"]), page=int(item["page"]), index=int(item["index"])))
            except (KeyError, TypeError, ValueError):
                continue
        return chunks

    def _extract_to_cache(self, paper: Paper, data: bytes) -> list[PdfChunk]:
        try:
            extraction = extract_pdf_text(data)
        except PdfExtractionError as exc:
            paper.fulltext_status = "failed"
            paper.fulltext_error = str(exc)
            return []
        return self._write_text_to_cache(paper, extraction)

    def _write_text_to_cache(self, paper: Paper, extraction: PdfExtraction) -> list[PdfChunk]:
        text_path = self.text_dir / f"{paper.id}.json"
        payload = {
            "paper_id": paper.id,
            "title": paper.title,
            "source_file": paper.local_pdf_path,
            "extracted_at": utc_now_iso(),
            "page_count": extraction.page_count,
            "char_count": extraction.char_count,
            "chunks": [
                {"index": chunk.index, "page": chunk.page, "section": "", "text": chunk.text}
                for chunk in extraction.chunks
            ],
        }
        text_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paper.local_text_path = str(text_path)
        paper.fulltext_status = "extracted"
        paper.fulltext_error = None
        return extraction.chunks

    def _write_pdf_to_cache(self, paper: Paper, data: bytes) -> None:
        digest = hashlib.sha256(data).hexdigest()
        pdf_path = self.pdf_dir / _safe_pdf_filename(paper)
        pdf_path.write_bytes(data)
        paper.local_pdf_path = str(pdf_path)
        paper.fulltext_sha256 = digest
        paper.fulltext_downloaded_at = utc_now_iso()
        paper.fulltext_error = None


def _safe_pdf_url(value: str | None) -> bool:
    return bool(value and value.startswith(("http://", "https://")))


def _safe_pdf_filename(paper: Paper) -> str:
    title = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in paper.title.lower())
    title = "_".join(part for part in title.split("_") if part)[:70]
    stem = f"{paper.id or stable_fallback_id(paper)}_{title or 'paper'}"
    return f"{stem}.pdf"


def stable_fallback_id(paper: Paper) -> str:
    digest = hashlib.sha1(paper.title.encode("utf-8")).hexdigest()[:16]
    return digest


def _open_pdf_candidates(paper: Paper) -> list[str]:
    """Return only public-looking PDF alternatives; never bypass a publisher paywall."""
    candidates = [paper.pdf_url or ""]
    if paper.arxiv_id:
        candidates.append(f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf")
    source_match = re.search(r"arxiv\.org/(?:abs|html)/([^/?#]+)", paper.source_url or "", flags=re.I)
    if source_match:
        candidates.append(f"https://arxiv.org/pdf/{source_match.group(1)}.pdf")
    return list(dict.fromkeys(url for url in candidates if _safe_pdf_url(url)))


def _download_open_pdf(paper: Paper) -> bytes:
    errors: list[str] = []
    for url in _open_pdf_candidates(paper):
        try:
            return _download_pdf(url, referer=paper.source_url)
        except RuntimeError as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("无法下载开放 PDF：" + "; ".join(errors[:3]))


def _download_pdf(url: str, *, referer: str | None = None) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if _safe_pdf_url(referer):
        headers["Referer"] = str(referer)
    request = urllib.request.Request(
        url,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read(MAX_FULLTEXT_PDF_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} when downloading PDF") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not download PDF: {exc.reason}") from exc
    except (TimeoutError, ssl.SSLError, OSError) as exc:
        raise RuntimeError(f"Could not download PDF: {exc}") from exc

    if len(data) > MAX_FULLTEXT_PDF_BYTES:
        raise RuntimeError("PDF 超过 30 MB，未缓存全文。")
    if not data.lstrip().startswith(b"%PDF-"):
        raise RuntimeError("下载结果不是有效 PDF，未缓存全文。")
    return data
