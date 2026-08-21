from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re


MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 250
MAX_EXTRACTED_CHARS = 600_000


class PdfExtractionError(ValueError):
    """Raised when an uploaded PDF cannot be safely turned into text."""


@dataclass
class PdfChunk:
    text: str
    page: int
    index: int


@dataclass
class PdfExtraction:
    page_count: int
    char_count: int
    chunks: list[PdfChunk]


def extract_pdf_text(data: bytes) -> PdfExtraction:
    """Extract selectable PDF text and split it into page-aware chunks."""
    if not data:
        raise PdfExtractionError("上传的文件为空。")
    if len(data) > MAX_PDF_BYTES:
        raise PdfExtractionError("PDF 超过 25 MB，暂不支持上传。")
    if not data.lstrip().startswith(b"%PDF-"):
        raise PdfExtractionError("上传的文件不是有效的 PDF。")

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - deployment configuration
        raise PdfExtractionError("服务端未安装 pypdf，无法读取 PDF。") from exc

    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise PdfExtractionError("暂不支持受密码保护的 PDF。")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise PdfExtractionError(f"PDF 共 {len(reader.pages)} 页，当前最多支持 {MAX_PDF_PAGES} 页。")
        page_texts = [_clean_text(page.extract_text() or "") for page in reader.pages]
    except PdfExtractionError:
        raise
    except Exception as exc:
        raise PdfExtractionError(f"PDF 解析失败：{exc}") from exc

    chunks: list[PdfChunk] = []
    total_chars = 0
    for page_number, text in enumerate(page_texts, start=1):
        if not text:
            continue
        for piece in _split_text(text):
            remaining = MAX_EXTRACTED_CHARS - total_chars
            if remaining <= 0:
                break
            piece = piece[:remaining].strip()
            if not piece:
                continue
            chunks.append(PdfChunk(text=piece, page=page_number, index=len(chunks)))
            total_chars += len(piece)
        if total_chars >= MAX_EXTRACTED_CHARS:
            break

    if not chunks:
        raise PdfExtractionError(
            "没有从 PDF 中提取到可读文本。它可能是扫描件，请先进行 OCR 后再上传。"
        )
    return PdfExtraction(page_count=len(reader.pages), char_count=total_chars, chunks=chunks)


def relevant_chunks(question: str, chunks: list[PdfChunk], *, limit: int = 8, max_chars: int = 18_000) -> list[PdfChunk]:
    """Choose a compact, page-aware local context for a PDF question."""
    if not chunks:
        return []
    terms = _query_terms(question)
    lowered_question = question.lower()
    is_overview = any(
        marker in lowered_question
        for marker in ("总结", "概括", "摘要", "综述", "解读", "介绍", "分析", "讲讲", "全文", "summary", "overview")
    )
    if is_overview:
        return _evenly_spaced_chunks(chunks, limit=limit, max_chars=max_chars)

    scored: list[tuple[int, PdfChunk]] = []
    for chunk in chunks:
        text = chunk.text.lower()
        score = sum(text.count(term) * max(1, len(term)) for term in terms)
        if score:
            scored.append((score, chunk))

    if not scored:
        scored = [(0, chunk) for chunk in chunks[:limit]]

    selected: list[PdfChunk] = []
    seen: set[int] = set()
    for _, chunk in sorted(scored, key=lambda item: (-item[0], item[1].index)):
        if chunk.index in seen:
            continue
        if sum(len(item.text) for item in selected) + len(chunk.text) > max_chars and selected:
            continue
        selected.append(chunk)
        seen.add(chunk.index)
        if len(selected) >= limit:
            break
    return selected


def _evenly_spaced_chunks(chunks: list[PdfChunk], *, limit: int, max_chars: int) -> list[PdfChunk]:
    count = min(limit, len(chunks))
    if count == 1:
        return [chunks[0]]
    indices = [round(index * (len(chunks) - 1) / (count - 1)) for index in range(count)]
    selected: list[PdfChunk] = []
    for index in indices:
        chunk = chunks[index]
        if sum(len(item.text) for item in selected) + len(chunk.text) > max_chars and selected:
            continue
        selected.append(chunk)
    return selected or [chunks[0]]


def _clean_text(value: str) -> str:
    return re.sub(r"[ \t]+", " ", value.replace("\x00", "")).strip()


def _split_text(text: str, *, chunk_size: int = 2400, overlap: int = 220) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = max(text.rfind("\n", start + chunk_size // 2, end), text.rfind(". ", start + chunk_size // 2, end))
            if boundary > start:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def _query_terms(question: str) -> list[str]:
    terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}|[\u4e00-\u9fff]{2,}", question.lower())
    return list(dict.fromkeys(term for term in terms if len(term) >= 2))
