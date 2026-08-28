(() => {
  class ApiError extends Error {
    constructor(message, status = 0) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  }

  async function request(path, options = {}) {
    const response = await fetch(path, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new ApiError(payload.error || `请求失败 (${response.status})`, response.status);
    }
    return payload;
  }

  async function streamChat({ message, sessionId, evidencePaperIds, onEvent, signal }) {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        mode: "auto",
        session_id: sessionId,
        evidence_paper_ids: evidencePaperIds,
      }),
      signal,
    });
    if (!response.ok || !response.body) {
      throw new ApiError(`对话请求失败 (${response.status || "network"})`, response.status);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (line.trim()) onEvent(JSON.parse(line));
      }
    }
    if (buffer.trim()) onEvent(JSON.parse(buffer));
  }

  const jsonPost = (path, payload) => request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });

  window.paperApi = Object.freeze({
    ApiError,
    session: {
      snapshot: (sessionId) => request(`/api/state?session_id=${encodeURIComponent(sessionId)}`),
      clear: (sessionId) => request("/api/session/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      }),
    },
    chat: { stream: streamChat },
    jobs: {
      create: (payload) => jsonPost("/api/jobs", payload),
      get: (projectId, jobId) => request(`/api/jobs/${encodeURIComponent(jobId)}?${new URLSearchParams({ project_id: projectId }).toString()}`),
      events: (projectId, jobId, afterId = 0) => request(`/api/jobs/${encodeURIComponent(jobId)}/events?${new URLSearchParams({ project_id: projectId, after_id: String(afterId) }).toString()}`),
    },
    documents: {
      upload: (file, sessionId, paperId = "") => request(
        `/api/pdf/upload?${new URLSearchParams({ session_id: sessionId, filename: file.name, paper_id: paperId || "" }).toString()}`,
        { method: "POST", headers: { "Content-Type": "application/pdf" }, body: file },
      ),
    },
    projects: {
      list: () => request("/api/projects"),
      create: (payload) => jsonPost("/api/projects", payload),
      rename: (projectId, payload) => jsonPost(`/api/projects/${encodeURIComponent(projectId)}`, payload),
    },
    library: {
      updateState: (sessionId, paperId, updates) => request("/api/library/paper/state", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, paper_id: paperId, updates }),
      }),
      downloadPdf: (sessionId, paperId) => request("/api/library/paper/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, paper_id: paperId }),
      }),
      findOpenPdf: (sessionId, paperId) => request("/api/library/paper/find-open-pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, paper_id: paperId }),
      }),
      matchLocalPdfs: (sessionId) => request("/api/library/match-local-pdfs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      }),
      openFolder: () => request("/api/library/open-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      }),
    },
    writing: {
      drafts: (projectId) => request(`/api/projects/${encodeURIComponent(projectId)}/drafts`),
      draft: (projectId, draftId) => request(`/api/projects/${encodeURIComponent(projectId)}/drafts/${encodeURIComponent(draftId)}`),
      update: (projectId, draftId, payload) => jsonPost(`/api/projects/${encodeURIComponent(projectId)}/drafts/${encodeURIComponent(draftId)}`, payload),
      versions: (projectId, draftId) => request(`/api/projects/${encodeURIComponent(projectId)}/drafts/${encodeURIComponent(draftId)}/versions`),
      export: (projectId, draftId, format) => jsonPost(`/api/projects/${encodeURIComponent(projectId)}/drafts/${encodeURIComponent(draftId)}/export`, { format }),
    },
    discovery: {
      feed: (sessionId, topic) => request(`/api/discovery/feed?${new URLSearchParams({ session_id: sessionId || "default", topic: topic || "" }).toString()}`),
    },
  });
})();
