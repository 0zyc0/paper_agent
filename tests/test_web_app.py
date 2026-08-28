import io
import json

from paper_agent.core.assistant_engine import ResearchAssistantEngine
from paper_agent.interfaces.web_app import OUTPUT_DIR, WEB_DIR, ResearchAssistantHandler


def test_web_paths_resolve_from_project_root():
    assert (WEB_DIR / "index.html").is_file()
    assert (WEB_DIR / "api.js").is_file()
    assert OUTPUT_DIR.name == "outputs"


def test_web_handler_serializes_shared_engine_access():
    assert ResearchAssistantHandler.engine_lock.acquire(blocking=False)
    ResearchAssistantHandler.engine_lock.release()


def test_web_project_and_draft_endpoints_are_real_not_future_contracts(tmp_path):
    previous_engine = ResearchAssistantHandler.engine
    engine = ResearchAssistantEngine(
        store_path=tmp_path / "papers.sqlite", output_dir=tmp_path / "outputs", upload_dir=tmp_path / "pdfs"
    )
    ResearchAssistantHandler.engine = engine
    handler = object.__new__(ResearchAssistantHandler)
    handler.engine = engine
    responses: list[tuple[int, dict]] = []
    handler._send_json = lambda payload, status=200: responses.append((status, payload))

    def set_body(payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        handler.headers = {"Content-Length": str(len(raw))}
        handler.rfile = io.BytesIO(raw)

    try:
        set_body({"title": "HTTP 验证项目"})
        handler._handle_project_create()
        assert responses[-1][0] == 201
        project = responses[-1][1]["project"]
        draft = engine.store.create_draft(project["id"], {"title": "引言", "content_markdown": "# 引言"})
        assert handler._handle_project_get(f"/api/projects/{project['id']}/drafts/{draft['id']}", {})
        fetched = responses[-1][1]
        assert fetched["draft"]["content_markdown"] == "# 引言"
        set_body({"content_markdown": "# 引言\n\n已编辑"})
        assert handler._handle_project_post(f"/api/projects/{project['id']}/drafts/{draft['id']}")
        assert responses[-1][1]["draft"]["version"] == 2
    finally:
        ResearchAssistantHandler.engine = previous_engine
