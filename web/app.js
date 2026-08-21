const $ = (selector) => document.querySelector(selector);

const elements = {
  status: $("#statusText"),
  nav: $("#mainNav"),
  sessionSelect: $("#sessionSelect"),
  newSession: $("#newSessionBtn"),
  clear: $("#clearBtn"),
  upload: $("#uploadPdfBtn"),
  pdfInput: $("#pdfInput"),
  uploadInfo: $("#uploadInfo"),
  chatForm: $("#chatForm"),
  input: $("#messageInput"),
  send: $("#sendBtn"),
  messages: $("#messages"),
  workspaceMeta: $("#workspaceMeta"),
  viewEyebrow: $("#viewEyebrow"),
  viewTitle: $("#viewTitle"),
  projectSignal: $("#projectSignal"),
  intent: $("#intentView"),
  debug: $("#debugView"),
  evidencePreview: $("#evidencePreview"),
  paperCount: $("#paperCount"),
  libraryPaperCount: $("#libraryPaperCount"),
  selectedPaperCount: $("#selectedPaperCount"),
  abstractCount: $("#abstractCount"),
  sourceFilter: $("#sourceFilter"),
  yearFilter: $("#yearFilter"),
  libraryTableBody: $("#libraryTableBody"),
  libraryEmpty: $("#libraryEmpty"),
  documentList: $("#documentList"),
  readerHeading: $("#readerHeading"),
  readerBody: $("#readerBody"),
  refreshDiscovery: $("#refreshDiscoveryBtn"),
  outline: $("#outlineView"),
  draftTitle: $("#draftTitle"),
  draftEvidence: $("#draftEvidence"),
  draftMeta: $("#draftMeta"),
  draftPreview: $("#draftPreview"),
  fileList: $("#fileList"),
  topicList: $("#topicList"),
  feedList: $("#feedList"),
  directionList: $("#directionList"),
};

const SESSIONS_KEY = "paperAssistant.sessions.v4";
const LEGACY_SESSIONS_KEY = "paperAssistant.sessions.v3";
const ACTIVE_SESSION_KEY = "paperAssistant.activeSession.v4";
const VIEW_KEY = "paperAssistant.activeView.v1";

const state = { sessions: [], activeSessionId: "", activeView: "workbench", busy: false, runningTasks: {} };
let pendingPaperUploadId = "";

const viewMeta = {
  workbench: { eyebrow: "项目工作台", title: "研究与写作" },
  library: { eyebrow: "文献库", title: "证据与筛选" },
  reader: { eyebrow: "论文阅读", title: "上传文献" },
  writing: { eyebrow: "写作", title: "章节与草稿" },
  discover: { eyebrow: "发现", title: "动态与方向" },
};

function makeSession() {
  return {
    id: `s_${Date.now()}_${Math.random().toString(16).slice(2)}`,
    title: "新研究项目",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [], papers: [], selectedPaperIds: [], files: [], intent: null,
    lastAction: "", lastGeneratedDocument: null, lastAnswer: "", debug: null,
    uploadedDocuments: [], readerDocumentId: "", taskState: "", taskLog: [], processVisible: false,
    discovery: null, discoveryLoading: false, researchTopics: [],
  };
}

function normalizeSession(session) {
  Object.assign(session, { ...makeSession(), ...session });
  session.messages = Array.isArray(session.messages) ? session.messages : [];
  session.papers = Array.isArray(session.papers) ? session.papers : [];
  session.files = Array.isArray(session.files) ? session.files : [];
  session.uploadedDocuments = Array.isArray(session.uploadedDocuments) ? session.uploadedDocuments : [];
  session.discovery = session.discovery && typeof session.discovery === "object" ? session.discovery : null;
  session.discoveryLoading = Boolean(session.discoveryLoading);
  session.researchTopics = Array.isArray(session.researchTopics) ? session.researchTopics : [];
  session.taskLog = Array.isArray(session.taskLog) ? session.taskLog : [];
  session.selectedPaperIds = Array.isArray(session.selectedPaperIds)
    ? session.selectedPaperIds
    : session.papers.map((paper) => paper.id).filter(Boolean);
  syncSelectedPaperIds(session);
  return session;
}

function syncSelectedPaperIds(session, preferredIds = null, options = {}) {
  const validIds = new Set((session.papers || []).map((paper) => paper.id).filter(Boolean));
  const sourceIds = Array.isArray(preferredIds) ? preferredIds : session.selectedPaperIds;
  session.selectedPaperIds = (sourceIds || []).filter((id) => validIds.has(id));
  if (options.selectAllIfEmpty && !session.selectedPaperIds.length && session.papers.length) {
    session.selectedPaperIds = session.papers.map((paper) => paper.id).filter(Boolean);
  }
}

function activeSession() {
  let session = sessionById(state.activeSessionId);
  if (!session) {
    session = makeSession();
    state.sessions.unshift(session);
    state.activeSessionId = session.id;
  }
  return normalizeSession(session);
}

function sessionById(sessionId) {
  return state.sessions.find((item) => item.id === sessionId) || null;
}

function saveState() {
  localStorage.setItem(SESSIONS_KEY, JSON.stringify(state.sessions));
  localStorage.setItem(ACTIVE_SESSION_KEY, state.activeSessionId);
  localStorage.setItem(VIEW_KEY, state.activeView);
}

function loadState() {
  try {
    state.sessions = JSON.parse(localStorage.getItem(SESSIONS_KEY) || localStorage.getItem(LEGACY_SESSIONS_KEY) || "[]");
  } catch {
    state.sessions = [];
  }
  if (!Array.isArray(state.sessions)) state.sessions = [];
  state.sessions = state.sessions.filter((session) => session && session.id).slice(0, 30).map(normalizeSession);
  state.activeSessionId = localStorage.getItem(ACTIVE_SESSION_KEY) || state.sessions[0]?.id || "";
  state.activeView = localStorage.getItem(VIEW_KEY) || "workbench";
  if (!state.sessions.length) {
    const session = makeSession();
    state.sessions.push(session);
    state.activeSessionId = session.id;
  }
  saveState();
}

function setStatus(text) {
  elements.status.textContent = text;
}

function setBusy(isBusy, sessionId = state.activeSessionId) {
  if (isBusy) state.runningTasks[sessionId] = true;
  else delete state.runningTasks[sessionId];
  state.busy = Object.keys(state.runningTasks).length > 0;
  elements.send.disabled = state.busy;
  elements.input.disabled = state.busy;
  elements.upload.disabled = state.busy;
  elements.clear.disabled = state.busy;
  elements.newSession.disabled = false;
  elements.sessionSelect.disabled = false;
}

function syncActiveTaskUi() {
  const hasActiveTask = Boolean(state.runningTasks[state.activeSessionId]);
  const hasAnyTask = Object.keys(state.runningTasks).length > 0;
  state.busy = hasAnyTask;
  elements.send.disabled = hasAnyTask;
  elements.input.disabled = hasAnyTask;
  elements.upload.disabled = hasAnyTask;
  elements.clear.disabled = hasAnyTask;
  elements.newSession.disabled = false;
  elements.sessionSelect.disabled = false;
  const session = activeSession();
  if (session.taskState) setStatus(session.taskState);
  else if (hasActiveTask) setStatus("当前项目正在处理中...");
  else if (hasAnyTask) setStatus("另一个项目正在处理中...");
  else setStatus("就绪");
}

function setView(view) {
  if (!viewMeta[view]) return;
  state.activeView = view;
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  document.querySelectorAll(".view").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${view}View`);
  });
  elements.viewEyebrow.textContent = viewMeta[view].eyebrow;
  elements.viewTitle.textContent = viewMeta[view].title;
  saveState();
  render();
  if (view === "discover") fetchDiscovery(false);
}

function render() {
  const session = activeSession();
  renderSessionSelect();
  renderWorkspaceMeta(session);
  renderProjectSignal(session);
  renderMessages(session);
  renderIntent(session.intent);
  renderDebug(session.debug);
  renderEvidencePreview(session);
  renderLibrary(session);
  renderReader(session);
  renderWriting(session);
  renderDiscover(session);
  renderUploadInfo(session);
  const meta = viewMeta[state.activeView] || viewMeta.workbench;
  elements.viewEyebrow.textContent = meta.eyebrow;
  elements.viewTitle.textContent = meta.title;
}

function renderSessionSelect() {
  elements.sessionSelect.innerHTML = state.sessions.map((session) => {
    const selected = session.id === state.activeSessionId ? " selected" : "";
    return `<option value="${escapeAttr(session.id)}"${selected}>${escapeHtml(session.title)}</option>`;
  }).join("");
}

function renderWorkspaceMeta(session) {
  const topic = session.intent?.normalized_topic || "未设置研究主题";
  elements.workspaceMeta.textContent = `${topic} · ${session.papers.length} 篇论文 · ${session.uploadedDocuments.length} 份 PDF`;
}

function renderProjectSignal(session) {
  const signals = [
    ["研究主题", session.intent?.normalized_topic || "待识别"],
    ["证据论文", `${session.selectedPaperIds.length} / ${session.papers.length}`],
    ["上传文献", `${session.uploadedDocuments.length} 份`],
    ["写作草稿", `${session.files.length} 个文件`],
  ];
  elements.projectSignal.innerHTML = signals.map(([term, value]) => (
    `<div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(value)}</dd></div>`
  )).join("");
}

function renderMessages(session) {
  elements.messages.innerHTML = "";
  if (!session.messages.length && !shouldShowProcess(session)) {
    elements.messages.innerHTML = `<div class="empty-chat"><p>从一个研究问题开始。</p></div>`;
    return;
  }
  session.messages.forEach((message) => renderMessage(message.role, message.content, false));
  if (shouldShowProcess(session)) renderProcessMessage(session, false);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function shouldShowProcess(session) {
  const steps = Array.isArray(session.taskLog) ? session.taskLog.slice(-8) : [];
  const running = Boolean(state.runningTasks[session.id]);
  return running && session.processVisible && steps.length > 0;
}

function renderProcessMessage(session, append = true) {
  const steps = Array.isArray(session.taskLog) ? session.taskLog.slice(-8) : [];
  if (!steps.length) return;
  const current = steps[steps.length - 1];
  const item = document.createElement("article");
  item.className = "message assistant process-message";
  item.innerHTML = `<div class="bubble process-bubble">
    <div class="process-head">
      <div><p class="eyebrow">执行过程</p><strong>${escapeHtml(current?.text || "正在处理")}</strong></div>
      <span class="process-state running">进行中</span>
    </div>
    <ol class="process-steps">
      ${steps.map((step, index) => {
        const isLast = index === steps.length - 1;
        const stateClass = isLast ? "active" : "done";
        return `<li class="${stateClass}"><span>${escapeHtml(step.label || step.type || "步骤")}</span><p>${escapeHtml(step.text || "")}</p></li>`;
      }).join("")}
    </ol>
  </div>`;
  elements.messages.appendChild(item);
  if (append) elements.messages.scrollTop = elements.messages.scrollHeight;
}

function renderMessage(role, content, append = true) {
  const item = document.createElement("article");
  item.className = `message ${role}`;
  item.innerHTML = `<div class="bubble">${renderRichText(content)}</div>`;
  elements.messages.appendChild(item);
  if (append) elements.messages.scrollTop = elements.messages.scrollHeight;
  typesetMath(item);
}

function appendMessage(role, content, sessionId = state.activeSessionId) {
  const session = sessionById(sessionId);
  if (!session) return;
  session.messages.push({ role, content });
  session.messages = session.messages.slice(-180);
  session.updatedAt = Date.now();
  saveState();
  if (sessionId === state.activeSessionId) {
    if (state.activeView === "workbench") renderMessages(session);
    renderProjectSignal(session);
  }
}

function appendStatus(content, sessionId = state.activeSessionId) {
  const session = sessionById(sessionId);
  if (!session) return;
  addTaskStep(session, "status", "进度", content);
  if (sessionId === state.activeSessionId) renderMessages(session);
}

function addTaskStep(session, type, label, text) {
  const normalized = String(text || "").trim();
  if (!normalized) return;
  const previous = session.taskLog[session.taskLog.length - 1];
  if (previous && previous.text === normalized && previous.label === label) return;
  session.taskLog.push({ type, label, text: normalized, at: Date.now() });
  session.taskLog = session.taskLog.slice(-16);
  session.updatedAt = Date.now();
  saveState();
}

function renderIntent(intent) {
  if (!intent) {
    elements.intent.textContent = "等待研究请求";
    elements.intent.classList.add("muted");
    return;
  }
  elements.intent.classList.remove("muted");
  const rows = [
    ["方向", intent.normalized_topic || "-"],
    ["领域", intent.cs_area || "-"],
    ["范围", [(intent.target_venues || []).join(", "), (intent.target_venue_ranks || []).join(", ")].filter(Boolean).join(" · ") || "未限定"],
    ["年份", intent.recent_years ? `近 ${intent.recent_years} 年` : "未限定"],
  ];
  elements.intent.innerHTML = `<dl>${rows.map(([term, value]) => `<dt>${escapeHtml(term)}</dt><dd>${escapeHtml(value)}</dd>`).join("")}</dl>`;
}

function renderDebug(debug) {
  if (!debug) {
    elements.debug.textContent = "尚未执行任务";
    elements.debug.classList.add("muted");
    return;
  }
  elements.debug.classList.remove("muted");
  const kimi = debug.kimi || {};
  const search = debug.search || {};
  const calls = kimi.calls || [];
  const plan = (debug.tool_plan || []).join(" → ") || "等待计划";
  elements.debug.innerHTML = `
    <div class="run-summary"><span class="run-state ${kimi.success_count ? "ready" : ""}">${kimi.called ? `Kimi ${kimi.success_count || 0} 次` : "Kimi 未调用"}</span><span>${escapeHtml(plan)}</span></div>
    <dl><dt>动作</dt><dd>${escapeHtml(actionLabel(debug.action || ""))}</dd><dt>检索</dt><dd>${search.used_existing_evidence ? "复用当前证据" : `缓存 ${search.cached_hits || 0} 篇`}</dd><dt>调用</dt><dd>${escapeHtml(calls.map((call) => call.label).join(" / ") || "无")}</dd></dl>
  `;
}

function renderEvidencePreview(session) {
  const selected = selectedPapers(session).slice(0, 5);
  elements.paperCount.textContent = String(session.papers.length);
  if (!selected.length) {
    elements.evidencePreview.textContent = "检索到的论文会成为可写作证据。";
    elements.evidencePreview.classList.add("muted");
    return;
  }
  elements.evidencePreview.classList.remove("muted");
  elements.evidencePreview.innerHTML = selected.map((paper) => `
    <button class="evidence-item" type="button" data-paper-id="${escapeAttr(paper.id || "")}" title="在文献库中查看">
      <strong>${escapeHtml(paper.title)}</strong><span>${escapeHtml([paper.year, paper.venue || paper.source].filter(Boolean).join(" · "))}</span>
    </button>
  `).join("");
}

function renderLibrary(session) {
  const papers = filteredPapers(session);
  const sources = [...new Set(session.papers.map((paper) => paper.source).filter(Boolean))].sort();
  const years = [...new Set(session.papers.map((paper) => paper.year).filter(Boolean))].sort((left, right) => right - left);
  preserveOptions(elements.sourceFilter, "全部来源", "all", sources);
  preserveOptions(elements.yearFilter, "全部年份", "all", years.map(String));
  elements.libraryPaperCount.textContent = String(session.papers.length);
  elements.selectedPaperCount.textContent = String(session.selectedPaperIds.length);
  elements.abstractCount.textContent = String(session.papers.filter((paper) => paper.abstract).length);
  elements.libraryEmpty.hidden = papers.length > 0;
  elements.libraryTableBody.innerHTML = papers.map((paper) => {
    const included = session.selectedPaperIds.includes(paper.id) ? " checked" : "";
    const source = paper.source || "unknown";
    const evidence = paperEvidenceState(paper);
    const href = safeHref(paper.source_url || paper.pdf_url || "");
    return `<tr>
      <td><input class="paper-toggle" type="checkbox" data-paper-id="${escapeAttr(paper.id || "")}"${included} aria-label="纳入 ${escapeAttr(paper.title)}" /></td>
      <td><div class="paper-title">${href ? `<a href="${href}" target="_blank" rel="noreferrer">${escapeHtml(paper.title)}</a>` : escapeHtml(paper.title)}</div><p>${escapeHtml((paper.authors || []).slice(0, 3).join(", ") || "作者待补全")}</p></td>
      <td><span class="source-tag">${escapeHtml(source)}</span><small>${escapeHtml(paper.venue || "venue 待补全")}</small></td>
      <td>${escapeHtml(String(paper.year || "-"))}</td>
      <td>${renderPaperFileAction(paper, evidence, session)}</td>
    </tr>`;
  }).join("");
}

function renderPaperFileAction(paper, evidence, session) {
  const localUrl = paper.local_pdf_url
    ? safeHref(`${paper.local_pdf_url}&session_id=${encodeURIComponent(session.id)}`)
    : "";
  const localPath = paper.local_pdf_display_path || paper.local_pdf_path || "";
  const canDownload = Boolean(paper.pdf_url);
  const canUpload = Boolean(paper.id && !localUrl);
  let actions = `<span class="file-action-muted">无开放 PDF</span>`;
  let path = "";
  if (localUrl) {
    actions = `<a class="tiny-button" href="${localUrl}" target="_blank" rel="noreferrer">打开 PDF</a>`;
    path = `<span class="local-path" title="${escapeAttr(localPath)}">${escapeHtml(localPath)}</span>`;
  } else if (canDownload) {
    actions = `<button class="tiny-button paper-download" type="button" data-paper-id="${escapeAttr(paper.id || "")}">下载 PDF</button>`;
  }
  if (canUpload) {
    actions += `<button class="tiny-button paper-upload" type="button" data-paper-id="${escapeAttr(paper.id || "")}" title="把你本地的 PDF 绑定到这篇论文">上传原文</button>`;
  }
  const tip = paper.fulltext_tip ? `<small class="fulltext-tip">${escapeHtml(paper.fulltext_tip)}</small>` : "";
  return `<div class="paper-file-cell"><span class="evidence-status ${escapeAttr(evidence.className)}">${escapeHtml(evidence.label)}</span><div class="paper-file-actions">${actions}</div>${path}${tip}</div>`;
}

function paperEvidenceState(paper) {
  if (paper.fulltext_status === "extracted") return { label: "原文已缓存", className: "verified" };
  if (paper.local_pdf_path) return { label: "PDF 已缓存", className: "verified" };
  if (paper.fulltext_status === "failed") return { label: "下载失败", className: "failed" };
  if (paper.abstract) return { label: "摘要可用", className: "verified" };
  if (paper.pdf_url) return { label: "可取原文", className: "pending" };
  return { label: "仅元数据", className: "pending" };
}

function renderReader(session) {
  const documents = session.uploadedDocuments;
  if (!documents.length) {
    elements.documentList.innerHTML = `<div class="empty-state">还没有上传论文。</div>`;
    elements.readerHeading.innerHTML = `<p class="eyebrow">阅读器</p><h3>等待上传</h3>`;
    elements.readerBody.textContent = "上传 PDF 后，全文解析、章节阅读和页码证据会显示在这里。";
    elements.readerBody.classList.add("muted");
    return;
  }
  const selectedId = session.readerDocumentId || documents[documents.length - 1].id;
  session.readerDocumentId = selectedId;
  const documentInfo = documents.find((item) => item.id === selectedId) || documents[documents.length - 1];
  elements.documentList.innerHTML = documents.map((document) => `
    <button class="document-item ${document.id === documentInfo.id ? "active" : ""}" type="button" data-document-id="${escapeAttr(document.id)}">
      <strong>${escapeHtml(document.name)}</strong><span>${document.page_count} 页 · ${Number(document.char_count || 0).toLocaleString()} 字符</span>
    </button>
  `).join("");
  elements.readerHeading.innerHTML = `<p class="eyebrow">当前文献</p><h3>${escapeHtml(documentInfo.name)}</h3><p>${documentInfo.page_count} 页 · 已提取 ${Number(documentInfo.char_count || 0).toLocaleString()} 字符</p>`;
  const answer = session.lastAnswer || "可以直接提问全文解读、方法、实验、局限或指定页码。";
  elements.readerBody.classList.remove("muted");
  elements.readerBody.innerHTML = renderRichText(answer);
  typesetMath(elements.readerBody);
}

function renderWriting(session) {
  const topic = session.intent?.normalized_topic || "当前研究主题";
  const draft = session.lastGeneratedDocument;
  const lines = [
    `${topic} 的研究背景、问题和范围`,
    `基于 ${session.selectedPaperIds.length} 篇证据论文组织研究脉络`,
    "按用户要求生成章节、综述、总结或 BibTeX",
  ];
  const outline = Array.isArray(draft?.outline) && draft.outline.length ? draft.outline : lines;
  elements.outline.innerHTML = outline.map((line) => `<li>${escapeHtml(line)}</li>`).join("");
  elements.draftEvidence.textContent = `${session.selectedPaperIds.length} 篇证据`;
  if (!draft) {
    elements.draftTitle.textContent = "尚未生成草稿";
    elements.draftMeta.textContent = "生成后将显示文档类型、引用覆盖和证据强度。";
    elements.draftMeta.classList.add("muted");
    elements.draftPreview.textContent = "生成内容会以可追溯的草稿形式出现在这里。";
    elements.draftPreview.classList.add("muted");
  } else {
    elements.draftTitle.textContent = draft.title || "最新写作草稿";
    elements.draftMeta.classList.remove("muted");
    elements.draftMeta.innerHTML = renderDraftMeta(draft);
    elements.draftPreview.classList.remove("muted");
    elements.draftPreview.innerHTML = renderRichText(draft.preview_markdown || draft.preview || "草稿已生成，请从下方文件打开完整内容。");
    typesetMath(elements.draftPreview);
  }
  if (!session.files.length) {
    elements.fileList.textContent = "暂无生成文件";
    elements.fileList.classList.add("muted");
  } else {
    elements.fileList.classList.remove("muted");
    elements.fileList.innerHTML = session.files.map((file) => {
      const href = safeHref(file.url || "");
      if (!href) return `<span class="file-link">${escapeHtml(file.name)}</span>`;
      const preview = file.kind === "markdown"
        ? `<button class="file-preview-button" type="button" data-preview-file="${escapeAttr(file.url || "")}">预览</button>`
        : "";
      return `<span class="file-entry"><a class="file-link" href="${href}" target="_blank" rel="noreferrer">${escapeHtml(file.name)}</a>${preview}</span>`;
    }).join("");
  }
}

function renderDraftMeta(draft) {
  const quality = draft.quality_report || {};
  const evidence = draft.evidence_report || {};
  const labels = [];
  if (draft.writing_kind) labels.push(`<span class="draft-chip">${escapeHtml(writingKindLabel(draft.writing_kind))}</span>`);
  if (quality.word_count) labels.push(`<span class="draft-chip">${Number(quality.word_count).toLocaleString()} 字</span>`);
  if (quality.citation_uses !== undefined) labels.push(`<span class="draft-chip">${Number(quality.citation_uses)} 处引用</span>`);
  const fulltext = quality.fulltext_paper_count ?? evidence.local_fulltext_count;
  if (fulltext !== undefined) labels.push(`<span class="draft-chip verified">${Number(fulltext)} 篇全文证据</span>`);
  const warnings = Array.isArray(quality.warnings) ? quality.warnings : [];
  const warningHtml = warnings.length ? `<p class="draft-warning">${escapeHtml(warnings.join(" "))}</p>` : "";
  const truncated = draft.preview_truncated ? `<p class="draft-note">当前显示的是长文档前半部分，可点击下方“预览”加载完整文件。</p>` : "";
  return `<div class="draft-chips">${labels.join("")}</div>${warningHtml}${truncated}`;
}

function writingKindLabel(kind) {
  return {
    related_work: "相关工作",
    survey: "研究综述",
    introduction: "引言",
    method_section: "方法章节",
    experiment_section: "实验章节",
    summary: "文献总结",
    outline: "论文大纲",
    bibliography: "BibTeX 导出",
    uploaded_pdf_section: "基于 PDF 写作",
    report: "研究报告",
  }[kind] || "写作草稿";
}

function renderDiscover(session) {
  const topic = session.intent?.normalized_topic || "尚未设置主题";
  const discovery = session.discovery || null;
  if (elements.refreshDiscovery) elements.refreshDiscovery.disabled = session.discoveryLoading;
  const profile = discovery?.profile || {};
  const topics = discovery?.topics || profile.topics || session.researchTopics || [];
  const groupedSources = summarizeDiscoverySources(discovery?.sources || []);
  const sourceSummary = groupedSources.length
    ? groupedSources.map((source) => `${source.source}: ${source.ok}/${source.total}`).join(" · ")
    : "OpenAlex / arXiv / DBLP / RSSHub 可用时会更新";
  elements.topicList.innerHTML = `
    <div class="discovery-profile">
      <span>${session.discoveryLoading ? "正在刷新" : "研究画像"}</span>
      <strong>${escapeHtml(discovery?.topic || topic)}</strong>
      <p>基于 ${Number(profile.paper_count || session.papers.length || 0)} 篇当前证据和 ${topics.length || 1} 个历史检索方向推荐。</p>
    </div>
    <div class="topic-chip-list">
      ${(topics.length ? topics : [topic]).slice(0, 6).map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
    </div>
    <div class="source-grid">
      ${groupedSources.map((source) => `<div><strong>${escapeHtml(source.source)}</strong><span>${source.ok}/${source.total} 成功 · ${source.count} 条</span></div>`).join("") || `<div><strong>来源状态</strong><span>${escapeHtml(sourceSummary)}</span></div>`}
    </div>
  `;
  if (session.discoveryLoading && !discovery) {
    elements.feedList.innerHTML = `<div class="empty-state">正在从 OpenAlex、arXiv、DBLP 和配置的技术动态源获取最新内容。</div>`;
    elements.directionList.innerHTML = `<div class="empty-state">发现结果返回后会生成研究机会。</div>`;
    return;
  }
  const papers = discovery?.papers || [];
  const paperGroups = discovery?.paper_groups || {};
  const techItems = discovery?.tech_items || [];
  const searchLinks = discovery?.search_links || [];
  const groupedPaperHtml = Object.entries(paperGroups).map(([groupTopic, groupPapers]) => `
    <div class="discovery-category">
      <div class="category-title"><span>方向</span><strong>${escapeHtml(groupTopic)}</strong></div>
      ${(groupPapers || []).slice(0, 4).map((paper) => renderDiscoveryPaper(paper)).join("") || `<div class="empty-state">这个方向暂未发现新论文。</div>`}
    </div>
  `).join("");
  const paperHtml = groupedPaperHtml || (papers.length ? papers.slice(0, 8).map((paper) => {
    return renderDiscoveryPaper(paper);
  }).join("") : `<div class="empty-state">还没有发现增量论文。先在工作台设置研究主题，或点击刷新发现。</div>`);
  const techHtml = techItems.slice(0, 6).map((item) => {
    const href = safeHref(item.url || "");
    return `<div class="feed-item tech-feed">
      <strong>${href ? `<a href="${href}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a>` : escapeHtml(item.title)}</strong>
      <span>${escapeHtml([item.source, item.kind === "search" ? "搜索入口" : item.published_at].filter(Boolean).join(" · "))}</span>
    </div>`;
  }).join("");
  const linkHtml = searchLinks.map((item) => {
    const href = safeHref(item.url || "");
    return `<a class="platform-link" href="${href}" target="_blank" rel="noreferrer"><strong>${escapeHtml(item.source)}</strong><span>${escapeHtml(item.title)}</span></a>`;
  }).join("");
  elements.feedList.innerHTML = `
    <div class="discovery-block"><div class="category-title"><span>论文动态</span><strong>按历史方向分类</strong></div>${paperHtml}</div>
    <div class="discovery-block"><div class="category-title"><span>技术社区</span><strong>文章动态</strong></div>${techHtml || `<div class="empty-state">配置 RSSHub 或 RSS 源后显示公众号/CSDN/知乎文章。</div>`}</div>
    <div class="discovery-block"><div class="category-title"><span>平台入口</span><strong>继续检索</strong></div><div class="platform-grid">${linkHtml}</div></div>
  `;
  const trends = discovery?.trends || [];
  const directions = discovery?.directions || [];
  elements.directionList.innerHTML = [
    ...trends.map((trend) => `<div class="direction-item"><span>${escapeHtml(trend.label || "趋势")}</span><p><strong>${escapeHtml(trend.title || "")}</strong><br>${escapeHtml(trend.summary || "")}</p></div>`),
    ...directions.map((direction) => `<div class="direction-item"><span>${escapeHtml(direction.label || "方向")}</span><p>${escapeHtml(direction.text || "")}</p></div>`),
  ].join("") || `<div class="empty-state">完成一次发现刷新后生成可验证方向。</div>`;
}

function renderDiscoveryPaper(paper) {
  const href = safeHref(paper.url || "");
  return `<div class="feed-item paper-feed">
    <strong>${href ? `<a href="${href}" target="_blank" rel="noreferrer">${escapeHtml(paper.title)}</a>` : escapeHtml(paper.title)}</strong>
    <span>${escapeHtml([paper.year || paper.published_at || "-", paper.venue || paper.source].filter(Boolean).join(" · "))}</span>
  </div>`;
}

function summarizeDiscoverySources(sources) {
  const grouped = {};
  (sources || []).forEach((item) => {
    const key = item.source || "来源";
    if (!grouped[key]) grouped[key] = { source: key, ok: 0, total: 0, count: 0 };
    grouped[key].total += 1;
    if (item.status === "ok") grouped[key].ok += 1;
    grouped[key].count += Number(item.count || 0);
  });
  return Object.values(grouped);
}

function renderUploadInfo(session) {
  const documents = session.uploadedDocuments;
  if (!documents.length) {
    elements.uploadInfo.textContent = "未上传 PDF";
    return;
  }
  elements.uploadInfo.textContent = `已读 ${documents.length} 份 PDF`;
  elements.uploadInfo.title = documents.map((document) => `${document.name} · ${document.page_count} 页`).join("\n");
}

function selectedPapers(session) {
  const wanted = new Set(session.selectedPaperIds);
  return session.papers.filter((paper) => wanted.has(paper.id));
}

function filteredPapers(session) {
  const source = elements.sourceFilter.value || "all";
  const year = elements.yearFilter.value || "all";
  return session.papers.filter((paper) => (
    (source === "all" || paper.source === source) && (year === "all" || String(paper.year) === year)
  ));
}

function preserveOptions(select, allLabel, allValue, values) {
  const previous = select.value || allValue;
  select.innerHTML = [`<option value="${allValue}">${allLabel}</option>`, ...values.map((value) => `<option value="${escapeAttr(value)}">${escapeHtml(value)}</option>`)].join("");
  select.value = values.includes(previous) || previous === allValue ? previous : allValue;
}

async function sendMessage(message) {
  const session = activeSession();
  const sessionId = session.id;
  session.taskState = "正在理解请求...";
  session.taskLog = [];
  session.processVisible = true;
  addTaskStep(session, "planning", "规划", session.taskState);
  setBusy(true, sessionId);
  if (sessionId === state.activeSessionId) {
    setStatus(session.taskState);
    renderMessages(session);
  }
  let receivedOutput = false;
  let receivedError = false;
  try {
    await window.paperApi.chat.stream({
      message,
      sessionId,
      evidencePaperIds: session.selectedPaperIds,
      onEvent: (event) => {
        if (event.type === "answer" && String(event.content || "").trim()) receivedOutput = true;
        if (event.type === "document") receivedOutput = true;
        if (event.type === "papers" && Array.isArray(event.papers) && event.papers.length > 0) receivedOutput = true;
        if (event.type === "error") receivedError = true;
        handleEvent(event, sessionId);
      },
    });
    session.taskState = "";
    if (receivedError) {
      session.processVisible = false;
      addTaskStep(session, "error", "失败", "任务执行失败，请查看错误信息后重试。");
      if (sessionId === state.activeSessionId) setStatus("执行失败");
    } else if (receivedOutput) {
      session.processVisible = false;
      addTaskStep(session, "done", "完成", "任务已完成，结果已写入当前项目。");
      if (sessionId === state.activeSessionId) setStatus("就绪");
    } else {
      session.processVisible = false;
      addTaskStep(session, "warning", "未完成", "没有收到可展示结果，已停止本次任务。");
      if (sessionId === state.activeSessionId) setStatus("未完成");
    }
  } catch (error) {
    appendMessage("assistant error", `请求失败：${error.message || error}`, sessionId);
    session.taskState = "请求失败";
    session.processVisible = false;
    addTaskStep(session, "error", "失败", error.message || String(error));
    if (sessionId === state.activeSessionId) setStatus("请求失败");
  } finally {
    setBusy(false, sessionId);
    if (sessionId === state.activeSessionId) render();
  }
}

async function uploadPdf(file, paperId = "") {
  const sessionId = state.activeSessionId;
  const session = activeSession();
  const paper = paperId ? session.papers.find((item) => item.id === paperId) : null;
  session.taskState = paper
    ? `正在把 ${file.name} 绑定到《${paper.title}》...`
    : `正在读取 ${file.name}...`;
  session.taskLog = [];
  session.processVisible = true;
  addTaskStep(session, "upload", "读取 PDF", session.taskState);
  setBusy(true, sessionId);
  setStatus(session.taskState);
  renderMessages(session);
  try {
    const payload = await window.paperApi.documents.upload(file, sessionId, paperId);
    const target = sessionById(sessionId);
    if (!target) return;
    target.uploadedDocuments = [...target.uploadedDocuments.filter((document) => document.id !== payload.document.id), payload.document];
    target.readerDocumentId = payload.document.id;
    if (payload.paper) replaceSessionPaper(target, payload.paper);
    target.taskState = "";
    target.processVisible = false;
    const savedPath = payload.document.local_pdf_display_path || payload.document.local_pdf_path || "个人文献库";
    addTaskStep(target, "done", "完成", `已读取 ${payload.document.name}，共 ${payload.document.page_count} 页。`);
    const linkedTitle = payload.document.linked_paper_title;
    const attachLine = linkedTitle ? `\n\n已补全论文：**${linkedTitle}**。之后写作会优先使用这份本地全文。` : "";
    appendMessage("assistant", `已读取 PDF：**${payload.document.name}**（${payload.document.page_count} 页，约 ${payload.document.char_count.toLocaleString()} 个字符）。${attachLine}\n\n保存位置：\`${savedPath}\``, sessionId);
    if (sessionId === state.activeSessionId) setStatus("PDF 已就绪");
  } catch (error) {
    const target = sessionById(sessionId);
    if (target) target.taskState = "PDF 上传失败";
    if (target) target.processVisible = false;
    if (target) addTaskStep(target, "error", "失败", error.message || String(error));
    appendMessage("assistant error", `PDF 上传失败：${error.message || error}`, sessionId);
    if (sessionId === state.activeSessionId) setStatus("PDF 上传失败");
  } finally {
    setBusy(false, sessionId);
    if (sessionId === state.activeSessionId) render();
  }
}

async function downloadPaperPdf(paperId) {
  const sessionId = state.activeSessionId;
  const session = activeSession();
  if (!paperId || state.busy) return;
  const paper = session.papers.find((item) => item.id === paperId);
  const title = paper?.title || "论文";
  session.taskState = `正在下载《${title}》到个人文献库...`;
  addTaskStep(session, "download", "文献库", session.taskState);
  setBusy(true, sessionId);
  setStatus("正在下载 PDF");
  render();
  try {
    const payload = await window.paperApi.library.downloadPdf(sessionId, paperId);
    const target = sessionById(sessionId);
    if (!target) return;
    if (payload.paper) replaceSessionPaper(target, payload.paper);
    target.taskState = "";
    target.processVisible = false;
    if (payload.ok) {
      addTaskStep(target, "done", "文献库", payload.message || "PDF 已保存到个人文献库。");
      const savedPath = payload.paper?.local_pdf_display_path || payload.paper?.local_pdf_path || "data/paper_files/pdf/";
      appendMessage("assistant", `${payload.message || "PDF 已保存到个人文献库。"}\n\n保存位置：\`${savedPath}\`\n\n你可以在文献库中点击 **打开 PDF** 查看本地原文。`, sessionId);
      if (sessionId === state.activeSessionId) setStatus("PDF 已入库");
    } else {
      addTaskStep(target, "warning", "文献库", payload.message || "PDF 未能自动下载，请手动上传原文。");
      appendMessage("assistant error", payload.message || "PDF 未能自动下载。请在文献库中点击“上传原文”补全。", sessionId);
      if (sessionId === state.activeSessionId) setStatus("等待手动上传");
    }
  } catch (error) {
    const target = sessionById(sessionId);
    if (target) {
      target.taskState = "";
      target.processVisible = false;
      addTaskStep(target, "error", "文献库", error.message || String(error));
    }
    appendMessage("assistant error", `PDF 下载失败：${error.message || error}`, sessionId);
    if (sessionId === state.activeSessionId) setStatus("就绪");
  } finally {
    setBusy(false, sessionId);
    if (sessionId === state.activeSessionId) render();
  }
}

function replaceSessionPaper(session, updatedPaper) {
  const index = session.papers.findIndex((paper) => paper.id === updatedPaper.id);
  if (index >= 0) session.papers.splice(index, 1, updatedPaper);
  else session.papers.unshift(updatedPaper);
  if (!session.selectedPaperIds.includes(updatedPaper.id)) session.selectedPaperIds.push(updatedPaper.id);
  saveState();
}

function handleEvent(event, sessionId) {
  const session = sessionById(sessionId);
  if (!session) return;
  if (event.type === "status") {
    session.taskState = event.message || "正在处理中...";
    if (sessionId === state.activeSessionId) {
      setStatus(session.taskState);
      appendStatus(session.taskState, sessionId);
    }
    saveState();
    return;
  }
  if (event.type === "action") {
    session.lastAction = event.action?.action || "";
    const tools = event.action?.tools || [];
    const toolText = tools.length ? `，工具链：${tools.join(" → ")}` : "";
    addTaskStep(session, "action", "决策", `判断任务为「${actionLabel(session.lastAction)}」${toolText}`);
    saveState();
    if (sessionId === state.activeSessionId) renderMessages(session);
    return;
  }
  if (event.type === "intent") {
    session.intent = event.intent;
    const topic = event.intent?.normalized_topic || "未识别主题";
    if (event.intent?.normalized_topic) {
      session.researchTopics = [
        event.intent.normalized_topic,
        ...(session.researchTopics || []).filter((item) => item.toLowerCase() !== event.intent.normalized_topic.toLowerCase()),
      ].slice(0, 12);
      session.discovery = null;
    }
    const scope = [
      (event.intent?.target_venues || []).join(", "),
      (event.intent?.target_venue_ranks || []).join(", "),
      event.intent?.recent_years ? `近 ${event.intent.recent_years} 年` : "",
    ].filter(Boolean).join(" · ") || "未限定来源范围";
    addTaskStep(session, "intent", "识别", `${topic}；${scope}`);
  } else if (event.type === "papers") {
    session.papers = event.papers || [];
    session.selectedPaperIds = session.papers.map((paper) => paper.id).filter(Boolean);
    const paperMessage = session.papers.length
      ? `已得到 ${session.papers.length} 篇候选论文，默认全部纳入证据池。`
      : "当前约束下没有检索到候选论文。";
    addTaskStep(session, "papers", "检索", paperMessage);
  } else if (event.type === "document") {
    session.processVisible = false;
    session.files = [...(event.files || []), ...session.files];
    session.lastGeneratedDocument = event;
    addTaskStep(session, "document", "写作", `已生成 ${event.title || "写作草稿"} 和 ${event.files?.length || 0} 个文件。`);
  } else if (event.type === "answer") {
    session.processVisible = false;
    session.lastAnswer = event.content;
    appendMessage("assistant", event.content, sessionId);
  } else if (event.type === "debug") {
    session.debug = event.debug || null;
    const calls = event.debug?.kimi?.success_count || 0;
    if (calls) addTaskStep(session, "debug", "模型", `Kimi 调用成功 ${calls} 次。`);
  } else if (event.type === "error") {
    session.processVisible = false;
    addTaskStep(session, "error", "失败", event.message || "执行失败");
    appendMessage("assistant error", event.message, sessionId);
  }
  session.updatedAt = Date.now();
  saveState();
  if (sessionId === state.activeSessionId) render();
}

async function fetchState(sessionId = state.activeSessionId) {
  try {
    const snapshot = await window.paperApi.session.snapshot(sessionId);
    const session = sessionById(sessionId);
    if (!session) return;
    session.papers = snapshot.papers || session.papers;
    session.intent = snapshot.intent || session.intent;
    session.files = snapshot.generated_files || session.files;
    session.lastGeneratedDocument = snapshot.last_generated_document || session.lastGeneratedDocument;
    session.debug = snapshot.debug || session.debug;
    session.uploadedDocuments = snapshot.uploaded_documents || session.uploadedDocuments;
    session.researchTopics = snapshot.research_topics || session.researchTopics;
    const hasEvidenceIds = Object.prototype.hasOwnProperty.call(snapshot, "evidence_paper_ids");
    syncSelectedPaperIds(session, hasEvidenceIds ? snapshot.evidence_paper_ids : null, { selectAllIfEmpty: !hasEvidenceIds });
    saveState();
    if (sessionId === state.activeSessionId) render();
  } catch {
    if (sessionId === state.activeSessionId) setStatus("后端状态暂时不可用");
  }
}

async function fetchDiscovery(force = false) {
  const session = activeSession();
  const topicKey = (session.researchTopics || []).join("|") || session.intent?.normalized_topic || session.papers[0]?.title || "";
  if (!topicKey && !force) {
    renderDiscover(session);
    return;
  }
  if (!force && session.discovery?.profile_key === topicKey && session.discovery?.updated_at) {
    return;
  }
  session.discoveryLoading = true;
  saveState();
  renderDiscover(session);
  try {
    const payload = await window.paperApi.discovery.feed(session.id, "");
    payload.profile_key = topicKey;
    const target = sessionById(session.id);
    if (!target) return;
    target.discovery = payload;
    target.discoveryLoading = false;
    target.updatedAt = Date.now();
    saveState();
    if (target.id === state.activeSessionId) renderDiscover(target);
  } catch (error) {
    const target = sessionById(session.id);
    if (target) {
      target.discoveryLoading = false;
      target.discovery = {
        topic: topicKey || "computer science",
        updated_at: new Date().toISOString(),
        sources: [{ source: "发现服务", status: "error", count: 0, error: error.message || String(error) }],
        papers: [],
        tech_items: [],
        search_links: [],
        trends: [{ label: "失败", title: "发现刷新失败", summary: error.message || String(error) }],
        directions: [],
      };
      saveState();
      if (target.id === state.activeSessionId) renderDiscover(target);
    }
  }
}

function createSession() {
  const session = makeSession();
  state.sessions.unshift(session);
  state.activeSessionId = session.id;
  saveState();
  setView("workbench");
  syncActiveTaskUi();
  setStatus("已新建项目");
}

function switchSession(sessionId) {
  state.activeSessionId = sessionId;
  saveState();
  render();
  syncActiveTaskUi();
  fetchState(sessionId);
}

function renameUntitledSession(message) {
  const session = activeSession();
  if (session.title !== "新研究项目") return;
  session.title = message.replace(/\s+/g, " ").slice(0, 24) || "新研究项目";
  saveState();
}

function actionLabel(action) {
  return { chat: "自由对话", search: "检索论文", answer: "依据证据回答", document: "生成文档" }[action] || "未识别";
}

function renderRichText(value) {
  const lines = String(value || "").replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let paragraph = [];
  let listType = "";
  let listItems = [];
  let codeLines = null;

  const flushParagraph = () => {
    if (!paragraph.length) return;
    blocks.push(`<p>${paragraph.map(renderInline).join("<br>")}</p>`);
    paragraph = [];
  };
  const flushList = () => {
    if (!listItems.length) return;
    blocks.push(`<${listType}>${listItems.map((item) => `<li>${renderInline(item)}</li>`).join("")}</${listType}>`);
    listType = "";
    listItems = [];
  };
  const flushCode = () => {
    if (codeLines === null) return;
    blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
    codeLines = null;
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (line.trim().startsWith("```")) {
      if (codeLines !== null) flushCode();
      else {
        flushParagraph();
        flushList();
        codeLines = [];
      }
      continue;
    }
    if (codeLines !== null) {
      codeLines.push(line);
      continue;
    }
    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|?\s*:?-{3,}/.test(lines[index + 1] || "")) {
      flushParagraph();
      flushList();
      const headers = splitTableRow(line);
      index += 1;
      const rows = [];
      while (/^\s*\|.*\|\s*$/.test(lines[index + 1] || "")) rows.push(splitTableRow(lines[++index]));
      blocks.push(`<div class="markdown-table-wrap"><table class="markdown-table"><thead><tr>${headers.map((cell) => `<th>${renderInline(cell)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${headers.map((_, cell) => `<td>${renderInline(row[cell] || "")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = Math.min(heading[1].length + 1, 5);
      blocks.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      continue;
    }
    if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) {
      flushParagraph();
      flushList();
      blocks.push("<hr>");
      continue;
    }
    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unordered || ordered) {
      flushParagraph();
      const nextType = unordered ? "ul" : "ol";
      if (listType && listType !== nextType) flushList();
      listType = nextType;
      listItems.push((unordered || ordered)[1]);
      continue;
    }
    if (line.startsWith(">")) {
      flushParagraph();
      flushList();
      blocks.push(`<blockquote>${renderInline(line.replace(/^>\s?/, ""))}</blockquote>`);
      continue;
    }
    paragraph.push(line);
  }
  flushParagraph();
  flushList();
  flushCode();
  return blocks.join("") || "<p></p>";
}

function splitTableRow(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
}

function renderInline(value) {
  let html = escapeHtml(String(value || ""));
  html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*\n][\s\S]*?[^*\n])\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>");
  html = html.replace(/\\cite\{([^}]+)\}/g, (_, keys) => `<span class="citation-chip">${escapeHtml(keys.split(",").map((key) => key.trim()).join(", "))}</span>`);
  html = html.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, url) => {
    const href = safeHref(url);
    return href ? `<a href="${href}" target="_blank" rel="noreferrer">${label}</a>` : label;
  });
  return html;
}

function typesetMath(root) {
  if (window.MathJax?.typesetPromise) window.MathJax.typesetPromise([root]).catch(() => {});
}

function safeHref(value) {
  try {
    const url = new URL(value, window.location.origin);
    return ["http:", "https:"].includes(url.protocol) ? escapeAttr(url.href) : "";
  } catch {
    return "";
  }
}

function escapeHtml(value) {
  return String(value || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/'/g, "&#39;");
}

elements.nav.addEventListener("click", (event) => {
  const button = event.target.closest("[data-view]");
  if (button) setView(button.dataset.view);
});
elements.newSession.addEventListener("click", createSession);
elements.sessionSelect.addEventListener("change", () => switchSession(elements.sessionSelect.value));
elements.clear.addEventListener("click", async () => {
  const session = activeSession();
  Object.assign(session, normalizeSession(makeSession()), { id: session.id, title: session.title, createdAt: session.createdAt });
  saveState();
  render();
  await window.paperApi.session.clear(state.activeSessionId).catch(() => {});
  setStatus("当前会话已清空");
});
elements.upload.addEventListener("click", () => elements.pdfInput.click());
if (elements.refreshDiscovery) elements.refreshDiscovery.addEventListener("click", () => fetchDiscovery(true));
elements.pdfInput.addEventListener("change", () => {
  const [file] = elements.pdfInput.files || [];
  elements.pdfInput.value = "";
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    appendMessage("assistant error", "请选择 PDF 文件。");
    return;
  }
  const paperId = pendingPaperUploadId;
  pendingPaperUploadId = "";
  uploadPdf(file, paperId);
});
elements.chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = elements.input.value.trim();
  if (!message || state.busy) return;
  elements.input.value = "";
  appendMessage("user", message);
  renameUntitledSession(message);
  sendMessage(message);
});
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) elements.chatForm.requestSubmit();
});
elements.sourceFilter.addEventListener("change", () => renderLibrary(activeSession()));
elements.yearFilter.addEventListener("change", () => renderLibrary(activeSession()));
elements.libraryTableBody.addEventListener("change", (event) => {
  const input = event.target.closest(".paper-toggle");
  if (!input) return;
  const session = activeSession();
  const ids = new Set(session.selectedPaperIds);
  if (input.checked) ids.add(input.dataset.paperId); else ids.delete(input.dataset.paperId);
  session.selectedPaperIds = [...ids];
  saveState();
  render();
});
elements.libraryTableBody.addEventListener("click", (event) => {
  const downloadButton = event.target.closest(".paper-download");
  if (downloadButton) {
    downloadPaperPdf(downloadButton.dataset.paperId);
    return;
  }
  const uploadButton = event.target.closest(".paper-upload");
  if (uploadButton) {
    pendingPaperUploadId = uploadButton.dataset.paperId || "";
    elements.pdfInput.click();
  }
});
elements.evidencePreview.addEventListener("click", (event) => {
  if (event.target.closest(".evidence-item")) setView("library");
});
elements.documentList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-document-id]");
  if (!button) return;
  activeSession().readerDocumentId = button.dataset.documentId;
  saveState();
  renderReader(activeSession());
});
elements.fileList.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-preview-file]");
  if (!button) return;
  const fileUrl = button.dataset.previewFile || "";
  if (!fileUrl.startsWith("/outputs/")) return;
  button.disabled = true;
  button.textContent = "加载中";
  try {
    const response = await fetch(fileUrl);
    if (!response.ok) throw new Error(`文件无法读取 (${response.status})`);
    const content = await response.text();
    const session = activeSession();
    if (session.lastGeneratedDocument) {
      session.lastGeneratedDocument.preview_markdown = content;
      session.lastGeneratedDocument.preview_truncated = false;
      saveState();
      renderWriting(session);
    }
  } catch (error) {
    appendMessage("assistant error", `文件预览失败：${error.message || error}`, state.activeSessionId);
  } finally {
    button.disabled = false;
    if (button.isConnected) button.textContent = "预览";
  }
});
document.querySelectorAll(".prompt-action").forEach((button) => button.addEventListener("click", () => {
  const prompt = button.dataset.prompt;
  setView("workbench");
  elements.input.value = prompt;
  elements.input.focus();
}));

loadState();
setView(state.activeView);
syncActiveTaskUi();
render();
fetchState();
