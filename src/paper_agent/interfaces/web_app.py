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
    engine.store.interrupt_incomplete_jobs()

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
        if path == "/api/projects":
            self._send_json({"projects": self.engine.list_projects()})
            return
        if path.startswith("/api/projects/"):
            if self._handle_project_get(path, query):
                return
        if path.startswith("/api/jobs/"):
            if self._handle_job_get(path, query):
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
        if path == "/api/library/paper/find-open-pdf":
            self._handle_find_open_pdf()
            return
        if path == "/api/library/match-local-pdfs":
            self._handle_match_local_pdfs()
            return
        if path == "/api/library/open-folder":
            self._handle_open_library_folder()
            return
        if path == "/api/projects":
            self._handle_project_create()
            return
        if path.startswith("/api/projects/"):
            if self._handle_project_post(path):
                return
        if path == "/api/jobs":
            self._handle_job_create()
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
        self.send_header("X-Accel-Buffering", "no")
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

    def _handle_find_open_pdf(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            session_id = str(payload.get("session_id") or "default")
            paper_id = str(payload.get("paper_id") or "").strip()
            if not paper_id:
                raise ValueError("paper_id 不能为空。")
            with self.engine_lock:
                result = self.engine.find_open_pdf(paper_id, session_id=session_id)
        except Exception as exc:
            self._send_json({"ok": False, "error": f"查找开放 PDF 失败：{exc}"}, status=400)
            return
        self._send_json(result, status=200 if result.get("ok") else 404)

    def _handle_match_local_pdfs(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            session_id = str(payload.get("session_id") or "default")
            with self.engine_lock:
                result = self.engine.match_local_library_pdfs(session_id=session_id)
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"ok": False, "error": f"匹配本地 PDF 失败：{exc}"}, status=500)
            return
        self._send_json({"ok": True, **result})

    def _handle_open_library_folder(self) -> None:
        try:
            with self.engine_lock:
                result = self.engine.open_local_library_folder()
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"ok": False, "error": f"打开个人文献库失败：{exc}"}, status=500)
            return
        self._send_json(result, status=200 if result.get("ok") else 500)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象。")
        return payload

    def _handle_project_get(self, path: str, query: dict[str, list[str]]) -> bool:
        parts = [part for part in path.split("/") if part]
        if len(parts) < 4:
            return False
        project_id = parts[2]
        if len(parts) == 4 and parts[3] == "systematic-review":
            self._send_json(self.engine.systematic_review(session_id=project_id))
            return True
        if len(parts) == 4 and parts[3] == "drafts":
            self._send_json({"drafts": self.engine.list_drafts(session_id=project_id)})
            return True
        if len(parts) >= 5 and parts[3] == "drafts":
            draft_id = parts[4]
            if len(parts) == 6 and parts[5] == "versions":
                self._send_json({"versions": self.engine.draft_versions(draft_id, session_id=project_id)})
            else:
                draft = self.engine.get_draft(draft_id, session_id=project_id)
                self._send_json({"draft": draft} if draft else {"error": "未找到草稿。"}, status=200 if draft else 404)
            return True
        return False

    def _handle_project_post(self, path: str) -> bool:
        parts = [part for part in path.split("/") if part]
        if len(parts) < 3:
            return False
        project_id = parts[2]
        try:
            payload = self._read_json_body()
            if len(parts) == 3:
                project = self.engine.rename_project(project_id, str(payload.get("title") or ""))
                self._send_json({"project": project} if project else {"error": "未找到项目。"}, status=200 if project else 404)
                return True
            if len(parts) >= 4 and parts[3] == "systematic-review":
                if len(parts) == 4:
                    result = self.engine.update_systematic_review_protocol(payload, session_id=project_id)
                elif len(parts) == 5 and parts[4] == "screen":
                    result = self.engine.screen_review_paper(
                        str(payload.get("paper_id") or ""),
                        session_id=project_id,
                        stage=str(payload.get("stage") or "title_abstract"),
                        decision=str(payload.get("decision") or "pending"),
                        reason=str(payload.get("reason") or ""),
                    )
                elif len(parts) == 5 and parts[4] == "export":
                    result = self.engine.export_systematic_review(
                        session_id=project_id, format=str(payload.get("format") or "evidence_csv")
                    )
                else:
                    return False
                self._send_json(result, status=200 if result.get("ok") else 400)
                return True
            if len(parts) >= 5 and parts[3] == "drafts":
                draft_id = parts[4]
                if len(parts) == 6 and parts[5] == "export":
                    result = self.engine.export_draft(draft_id, session_id=project_id, format=str(payload.get("format") or "markdown"))
                    self._send_json(result, status=200 if result.get("ok") else 400)
                    return True
                if len(parts) == 6 and parts[5] == "revise":
                    result = self.engine.revise_draft(
                        draft_id,
                        session_id=project_id,
                        selected_text=str(payload.get("selected_text") or ""),
                        instruction=str(payload.get("instruction") or ""),
                    )
                    self._send_json(result, status=200 if result.get("ok") else 400)
                    return True
                if len(parts) == 6 and parts[5] == "restore":
                    result = self.engine.restore_draft_version(
                        draft_id,
                        session_id=project_id,
                        version=int(payload.get("version") or 0),
                    )
                    self._send_json(result, status=200 if result.get("ok") else 400)
                    return True
                draft = self.engine.update_draft(draft_id, payload, session_id=project_id, note=str(payload.get("note") or "手动编辑"))
                self._send_json({"draft": draft} if draft else {"error": "未找到草稿。"}, status=200 if draft else 404)
                return True
        except Exception as exc:
            self._send_json({"error": f"项目操作失败：{exc}"}, status=400)
            return True
        return False

    def _handle_project_create(self) -> None:
        try:
            project = self.engine.create_project(str(self._read_json_body().get("title") or "新研究项目"))
            self._send_json({"project": project}, status=201)
        except Exception as exc:
            self._send_json({"error": f"创建项目失败：{exc}"}, status=400)

    def _handle_job_create(self) -> None:
        try:
            payload = self._read_json_body()
            project_id = str(payload.get("project_id") or payload.get("session_id") or "default")
            message, mode = str(payload.get("message") or "").strip(), str(payload.get("mode") or "auto")
            evidence_ids = payload.get("evidence_paper_ids") if isinstance(payload.get("evidence_paper_ids"), list) else None
            if not message:
                raise ValueError("请输入请求内容。")
            self.engine._session_state(project_id)
            job = self.engine.store.create_job(project_id, message, mode, payload={"evidence_paper_ids": evidence_ids or []})
            threading.Thread(target=self._run_job, args=(project_id, job["id"], message, mode, evidence_ids), daemon=True).start()
            self._send_json({"job": job}, status=202)
        except Exception as exc:
            self._send_json({"error": f"创建任务失败：{exc}"}, status=400)

    def _run_job(self, project_id: str, job_id: str, message: str, mode: str, evidence_ids: list[str] | None) -> None:
        result: dict = {}
        try:
            with self.engine_lock:
                for event in self.engine.handle_stream(message, mode=mode, session_id=project_id, evidence_paper_ids=evidence_ids):
                    self.engine.store.append_job_event(job_id, event)
                    if event.get("type") == "answer":
                        result["answer"] = event.get("content", "")
                    if event.get("type") == "document":
                        result["document"] = event
                result["snapshot"] = self.engine.snapshot(session_id=project_id)
            self.engine.store.finish_job(job_id, result=result)
        except Exception as exc:
            traceback.print_exc()
            self.engine.store.append_job_event(job_id, {"type": "error", "message": f"执行失败：{exc}"})
            self.engine.store.finish_job(job_id, result=result, error=str(exc))

    def _handle_job_get(self, path: str, query: dict[str, list[str]]) -> bool:
        parts = [part for part in path.split("/") if part]
        if len(parts) < 3:
            return False
        job_id = parts[2]
        project_id = str(query.get("project_id", query.get("session_id", ["default"]))[0] or "default")
        if len(parts) == 3:
            job = self.engine.store.get_job(project_id, job_id)
            self._send_json({"job": job} if job else {"error": "未找到任务。"}, status=200 if job else 404)
            return True
        if len(parts) == 4 and parts[3] == "events":
            try:
                after_id = int(query.get("after_id", ["0"])[0] or 0)
            except ValueError:
                after_id = 0
            job = self.engine.store.get_job(project_id, job_id)
            events = self.engine.store.job_events(project_id, job_id, after_id=after_id) if job else []
            self._send_json({"job": job, "events": events} if job else {"error": "未找到任务。"}, status=200 if job else 404)
            return True
        return False

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
