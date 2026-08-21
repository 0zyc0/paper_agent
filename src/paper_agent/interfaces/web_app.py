from __future__ import annotations

import argparse
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import sys
import threading
import traceback
from urllib.parse import parse_qs, unquote, urlparse

from ..core.assistant_engine import ResearchAssistantEngine
from ..core.pdf_reader import PdfExtractionError


ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = ROOT / "web"
OUTPUT_DIR = ROOT / "outputs"


class ResearchAssistantHandler(BaseHTTPRequestHandler):
    engine = ResearchAssistantEngine()
    engine_lock = threading.RLock()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        session_id = query.get("session_id", ["default"])[0]
        if path == "/" or path == "/index.html":
            self._serve_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path == "/app.js":
            self._serve_file(WEB_DIR / "app.js", "application/javascript; charset=utf-8")
            return
        if path == "/api.js":
            self._serve_file(WEB_DIR / "api.js", "application/javascript; charset=utf-8")
            return
        if path == "/styles.css":
            self._serve_file(WEB_DIR / "styles.css", "text/css; charset=utf-8")
            return
        if path.startswith("/assets/"):
            name = Path(unquote(path.removeprefix("/assets/"))).name
            target = WEB_DIR / "assets" / name
            if target.exists() and target.is_file():
                self._serve_file(target, _content_type(target))
                return
            self.send_error(404, "Asset not found")
            return
        if path == "/api/state":
            # A snapshot only reads one persisted session. It should remain
            # available while another request is waiting on external sources.
            snapshot = self.engine.snapshot(session_id=session_id)
            self._send_json(snapshot)
            return
        if path == "/api/discovery/feed":
            topic = query.get("topic", [""])[0]
            with self.engine_lock:
                feed = self.engine.discovery_feed(session_id=session_id, topic=topic)
            self._send_json(feed)
            return
        if path == "/api/library/paper/file":
            paper_id = str(query.get("paper_id", [""])[0] or "").strip()
            with self.engine_lock:
                target = self.engine.cached_paper_pdf_path(paper_id, session_id=session_id)
            if target:
                self._serve_file(target, "application/pdf")
                return
            self.send_error(404, "PDF not found")
            return
        if path.startswith("/outputs/"):
            name = Path(unquote(path.removeprefix("/outputs/"))).name
            target = OUTPUT_DIR / name
            if target.exists() and target.is_file():
                self._serve_file(target, _content_type(target))
                return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/api/pdf/upload":
            self._handle_pdf_upload(query)
            return
        if path == "/api/library/paper/state":
            self._handle_paper_state_update()
            return
        if path == "/api/library/paper/download":
            self._handle_paper_download()
            return
        if path not in {"/api/chat/stream", "/api/session/clear"}:
            self.send_error(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            session_id = str(payload.get("session_id") or "default")
            if path == "/api/session/clear":
                with self.engine_lock:
                    self.engine.reset_session(session_id)
                self._send_json({"ok": True})
                return
            message = str(payload.get("message") or "")
            mode = str(payload.get("mode") or "auto")
            evidence_paper_ids = payload.get("evidence_paper_ids")
            if not isinstance(evidence_paper_ids, list):
                evidence_paper_ids = None
        except Exception:
            self.send_error(400, "Invalid JSON payload")
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            with self.engine_lock:
                for event in self.engine.handle_stream(
                    message,
                    mode=mode,
                    session_id=session_id,
                    evidence_paper_ids=evidence_paper_ids,
                ):
                    if not self._write_event(event):
                        return
        except Exception as exc:
            traceback.print_exc()
            error = {"type": "error", "message": f"执行失败：{exc}"}
            self._write_event(error)

    def _handle_paper_state_update(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            session_id = str(payload.get("session_id") or "default")
            paper_id = str(payload.get("paper_id") or "").strip()
            updates = payload.get("updates")
            if not paper_id or not isinstance(updates, dict):
                raise ValueError("paper_id 和 updates 不能为空。")
            with self.engine_lock:
                result = self.engine.update_paper_asset_state(paper_id, updates, session_id=session_id)
        except Exception as exc:
            self._send_json({"ok": False, "error": f"更新论文状态失败：{exc}"}, status=400)
            return
        status = 200 if result.get("ok") else 404
        self._send_json(result, status=status)

    def _handle_paper_download(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            session_id = str(payload.get("session_id") or "default")
            paper_id = str(payload.get("paper_id") or "").strip()
            if not paper_id:
                raise ValueError("paper_id 不能为空。")
            with self.engine_lock:
                result = self.engine.download_paper_pdf(paper_id, session_id=session_id)
        except Exception as exc:
            self._send_json({"ok": False, "error": f"下载论文 PDF 失败：{exc}"}, status=400)
            return
        self._send_json(result)

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write("[web] " + format % args + "\n")

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404, "Not found")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _handle_pdf_upload(self, query: dict[str, list[str]]) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise PdfExtractionError("未收到 PDF 文件。")
            if length > 26 * 1024 * 1024:
                raise PdfExtractionError("PDF 超过 25 MB，暂不支持上传。")
            content_type = self.headers.get("Content-Type", "").lower()
            if not content_type.startswith("application/pdf"):
                raise PdfExtractionError("上传内容必须是 PDF 文件。")
            session_id = str(query.get("session_id", ["default"])[0] or "default")
            paper_id = str(query.get("paper_id", [""])[0] or "").strip()
            filename = unquote(str(query.get("filename", ["uploaded-paper.pdf"])[0] or "uploaded-paper.pdf"))
            data = self.rfile.read(length)
            with self.engine_lock:
                document = self.engine.upload_pdf(
                    data,
                    filename=filename,
                    session_id=session_id,
                    paper_id=paper_id or None,
                )
        except PdfExtractionError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            return
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"ok": False, "error": f"上传失败：{exc}"}, status=500)
            return
        payload = {"ok": True, "document": document}
        linked_paper = document.get("linked_paper") if isinstance(document, dict) else None
        if isinstance(linked_paper, dict):
            payload["paper"] = linked_paper
        self._send_json(payload)

    def _send_json(self, payload: dict, *, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _write_event(self, event: dict) -> bool:
        try:
            self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False


def _content_type(path: Path) -> str:
    if path.suffix == ".md":
        return "text/markdown; charset=utf-8"
    if path.suffix == ".bib":
        return "text/plain; charset=utf-8"
    if path.suffix == ".json":
        return "application/json; charset=utf-8"
    if path.suffix == ".png":
        return "image/png"
    if path.suffix == ".jpg" or path.suffix == ".jpeg":
        return "image/jpeg"
    if path.suffix == ".webp":
        return "image/webp"
    if path.suffix == ".svg":
        return "image/svg+xml"
    return "application/octet-stream"


def run_web(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), ResearchAssistantHandler)
    print(f"Research assistant web UI running at http://{host}:{port}")
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the paper assistant web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    run_web(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
