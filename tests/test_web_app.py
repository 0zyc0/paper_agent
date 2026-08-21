from paper_agent.interfaces.web_app import OUTPUT_DIR, WEB_DIR, ResearchAssistantHandler


def test_web_paths_resolve_from_project_root():
    assert (WEB_DIR / "index.html").is_file()
    assert (WEB_DIR / "api.js").is_file()
    assert OUTPUT_DIR.name == "outputs"


def test_web_handler_serializes_shared_engine_access():
    assert ResearchAssistantHandler.engine_lock.acquire(blocking=False)
    ResearchAssistantHandler.engine_lock.release()
