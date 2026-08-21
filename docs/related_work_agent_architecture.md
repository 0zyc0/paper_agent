# 科研写作与论文调研 Agent 初步架构

## 1. 目标

构建一个论文助手 Agent，第一阶段聚焦三件事：

1. 根据用户输入的研究主题、关键词、草稿或上传论文，检索对应领域的历史经典论文与最新论文。
2. 基于真实可验证的论文记录，按用户需求生成综述、论文章节、Related Work、论文总结或科研方向建议。
3. 为生成内容中的引用、证据和来源绑定可信论文记录。

核心原则：

- 不让大模型凭记忆编论文。
- 所有论文必须来自检索工具或用户上传文献。
- 所有引用必须绑定 `paper_id`、DOI、arXiv ID 或可信来源 URL。
- 本地不保存全量论文全文，只保存元数据、摘要索引和必要的全文缓存。

## 2. MVP 功能范围

第一版建议只做这条主链路：

```text
用户输入研究主题 / 上传论文 / 粘贴论文草稿
→ 提取研究问题、关键词、任务、方法、数据集
→ 检索论文候选池
→ 去重、排序、聚类
→ 选择核心论文
→ 对核心论文按需获取摘要或全文
→ 生成写作大纲
→ 生成目标章节 / 综述 / Related Work 草稿
→ 插入真实引用
→ 导出 Word / Markdown / BibTeX
```

第一版暂不做过重功能：

- 不建设全量全文论文库。
- 不做复杂社交化推荐。
- 不做完整论文管理器。
- 不做所有出版社全文抓取。

## 3. 前端架构

推荐做成一个科研写作工作台，而不是单纯聊天页。

### 3.1 页面结构

```text
/project/:id
  左侧：项目与文献库
  中间：写作工作台
  右侧：Agent 面板

/discover
  最新论文检索与订阅

/papers/:paper_id
  论文详情、摘要、引用信息、可用全文状态
```

### 3.2 科研写作工作台

核心区域：

- 研究主题输入框
- 关键词与研究范围编辑
- 检索来源选择
- 候选论文列表
- 论文聚类视图
- 写作大纲
- 正文编辑器
- 引用侧栏

候选论文卡片字段：

```text
标题
作者
年份
venue / source
摘要
DOI / arXiv ID / URL
引用数
推荐理由
是否已验证
是否已加入写作
```

写作编辑器要求：

- 每段文字显示对应引用。
- 点击引用可查看论文来源。
- 对无法验证的引用显示警告。
- 支持导出 Markdown、Word、BibTeX。

## 4. 后端架构

建议先采用单体后端 + 异步任务队列，后续再拆微服务。

```text
API Server
  用户与项目 API
  论文检索 API
  Related Work 生成 API
  引用验证 API
  导出 API

Worker
  论文检索任务
  元数据补全任务
  embedding 生成任务
  PDF 解析任务
  Related Work 草稿生成任务

Storage
  PostgreSQL
  pgvector / Qdrant
  S3 / MinIO / 本地对象存储
```

推荐技术栈：

```text
前端：Next.js / React
后端：FastAPI
任务队列：Celery / RQ / Dramatiq
数据库：PostgreSQL
向量检索：pgvector 起步，后续可换 Qdrant
对象存储：本地文件系统起步，后续换 S3 / MinIO
PDF 解析：PyMuPDF + GROBID 可选
```

## 5. 数据库设计

### 5.1 papers

保存论文元数据，不保存全文正文。

```sql
CREATE TABLE papers (
  id UUID PRIMARY KEY,
  title TEXT NOT NULL,
  abstract TEXT,
  year INT,
  published_at DATE,
  venue TEXT,
  source TEXT NOT NULL,
  source_url TEXT,
  doi TEXT,
  arxiv_id TEXT,
  citation_count INT,
  reference_count INT,
  is_verified BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  updated_at TIMESTAMP NOT NULL DEFAULT now()
);
```

### 5.2 paper_authors

```sql
CREATE TABLE paper_authors (
  id UUID PRIMARY KEY,
  paper_id UUID NOT NULL REFERENCES papers(id),
  name TEXT NOT NULL,
  author_order INT,
  external_author_id TEXT
);
```

### 5.3 paper_embeddings

只对标题、摘要、结构化总结生成 embedding。

```sql
CREATE TABLE paper_embeddings (
  id UUID PRIMARY KEY,
  paper_id UUID NOT NULL REFERENCES papers(id),
  embedding_type TEXT NOT NULL,
  content TEXT NOT NULL,
  vector VECTOR,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);
```

### 5.4 paper_summaries

保存轻量结构化知识，避免每次重新读全文。

```sql
CREATE TABLE paper_summaries (
  id UUID PRIMARY KEY,
  paper_id UUID NOT NULL REFERENCES papers(id),
  problem TEXT,
  method TEXT,
  contribution TEXT,
  dataset TEXT,
  result TEXT,
  limitation TEXT,
  generated_from TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);
```

### 5.5 paper_fulltext_cache

只缓存用户上传、收藏、实际引用或短期写作需要的论文全文。

```sql
CREATE TABLE paper_fulltext_cache (
  id UUID PRIMARY KEY,
  paper_id UUID NOT NULL REFERENCES papers(id),
  pdf_path TEXT,
  parsed_text_path TEXT,
  cache_status TEXT NOT NULL,
  last_accessed_at TIMESTAMP,
  expires_at TIMESTAMP,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);
```

### 5.6 paper_chunks

只给已缓存全文的论文建立 chunk。

```sql
CREATE TABLE paper_chunks (
  id UUID PRIMARY KEY,
  paper_id UUID NOT NULL REFERENCES papers(id),
  section_title TEXT,
  page_number INT,
  chunk_index INT NOT NULL,
  chunk_text TEXT NOT NULL,
  vector VECTOR,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);
```

### 5.7 writing_projects

```sql
CREATE TABLE writing_projects (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL,
  title TEXT NOT NULL,
  topic TEXT,
  research_question TEXT,
  target_venue TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  updated_at TIMESTAMP NOT NULL DEFAULT now()
);
```

### 5.8 project_papers

记录某个写作项目使用了哪些论文。

```sql
CREATE TABLE project_papers (
  id UUID PRIMARY KEY,
  project_id UUID NOT NULL REFERENCES writing_projects(id),
  paper_id UUID NOT NULL REFERENCES papers(id),
  role TEXT NOT NULL,
  relevance_score FLOAT,
  selected BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);
```

`role` 可选值：

```text
classic
recent
baseline
method
dataset
survey
contrast
```

### 5.9 citations

Related Work 中每条引用都要绑定真实论文。

```sql
CREATE TABLE citations (
  id UUID PRIMARY KEY,
  project_id UUID NOT NULL REFERENCES writing_projects(id),
  paper_id UUID NOT NULL REFERENCES papers(id),
  citation_key TEXT NOT NULL,
  bibtex TEXT,
  citation_style TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);
```

### 5.10 related_work_drafts

```sql
CREATE TABLE related_work_drafts (
  id UUID PRIMARY KEY,
  project_id UUID NOT NULL REFERENCES writing_projects(id),
  version INT NOT NULL,
  outline JSONB,
  content_markdown TEXT,
  claim_map JSONB,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);
```

`claim_map` 示例：

```json
[
  {
    "claim": "Retrieval-augmented methods improve factual grounding in generation tasks.",
    "paragraph_index": 1,
    "paper_ids": ["paper_uuid_1", "paper_uuid_2"],
    "evidence_status": "verified"
  }
]
```

## 6. 外部论文来源

第一版建议优先接入开放、稳定、适合自动化检索的来源：

```text
arXiv
Semantic Scholar
OpenAlex
Crossref
PubMed, 如果做医学方向
DBLP, 如果做计算机方向
OpenReview, 如果关注 ICLR / NeurIPS / ICML 投稿与评审
Papers with Code, 如果关注代码和 benchmark
```

检索策略：

```text
经典论文：按 citation_count、survey、历史高相关性排序
最新论文：按 published_at、source、领域关键词排序
代表论文：按引用数、venue、同被引关系排序
可复现论文：优先 Papers with Code / GitHub 链接存在
```

## 7. Tools 设计

Agent 不直接编论文信息，只能调用工具。

### 7.1 检索工具

```python
search_papers(
    query: str,
    fields: list[str],
    from_year: int | None,
    to_year: int | None,
    sources: list[str],
    limit: int
) -> list[PaperCandidate]
```

用途：

- 搜索历史论文。
- 搜索最新论文。
- 搜索某个方法、任务、数据集相关论文。

### 7.2 元数据补全工具

```python
enrich_paper_metadata(
    title: str | None,
    doi: str | None,
    arxiv_id: str | None,
    source_url: str | None
) -> Paper
```

用途：

- 补 DOI。
- 补作者。
- 补 venue。
- 补引用数。
- 补摘要。

### 7.3 论文验证工具

```python
verify_paper(
    paper_id: str
) -> VerificationResult
```

校验内容：

- DOI 是否存在。
- arXiv ID 是否存在。
- 标题和作者是否匹配。
- 来源 URL 是否可访问。

### 7.4 相关论文扩展工具

```python
find_related_papers(
    seed_paper_ids: list[str],
    relation_types: list[str],
    limit: int
) -> list[PaperCandidate]
```

`relation_types`：

```text
references
citations
same_author
semantic_similarity
same_dataset
same_method
```

### 7.5 全文按需获取工具

```python
fetch_fulltext_if_needed(
    paper_id: str,
    reason: str,
    cache_ttl_days: int
) -> FulltextStatus
```

只有这些情况才拉全文：

- 用户打开论文。
- 用户要求精读。
- 论文要被 Related Work 正式引用。
- 论文摘要不足以支持某个具体 claim。

### 7.6 Related Work 生成工具

```python
generate_related_work_outline(
    topic: str,
    selected_paper_ids: list[str]
) -> RelatedWorkOutline
```

```python
draft_related_work(
    outline: RelatedWorkOutline,
    selected_paper_ids: list[str],
    citation_style: str
) -> RelatedWorkDraft
```

### 7.7 引用导出工具

```python
generate_bibtex(
    paper_ids: list[str]
) -> dict[str, str]
```

```python
export_docx(
    project_id: str,
    draft_id: str,
    citation_style: str
) -> FileExport
```

## 8. Agent 与 Skills 设计

### 8.1 主 Agent

负责理解用户意图、拆分任务、调度子 Agent。

```text
输入：研究主题、草稿、上传论文、用户偏好
输出：检索计划、论文候选池、Related Work 草稿
```

### 8.2 Searcher Agent

负责论文检索。

规则：

- 只能基于工具返回的结果输出论文。
- 搜索结果必须写入 `papers` 表。
- 不允许补全不存在的标题、作者、年份。

### 8.3 Verifier Agent

负责校验论文是否真实。

规则：

- Related Work 中被引用的论文必须通过验证。
- 无法验证的论文只能放入候选区，不能进入正式引用。

### 8.4 Curator Agent

负责筛选和聚类。

聚类维度：

```text
研究问题
方法路线
任务场景
数据集
时间线
经典 / 最新 / survey / baseline
```

### 8.5 Writer Agent

负责写 Related Work。

规则：

- 每个关键 claim 必须关联至少一个 `paper_id`。
- 引用只从 `citations` 表中选择。
- 不确定时生成 TODO，而不是编造引用。

### 8.6 Reviewer Agent

负责检查草稿质量。

检查项：

- 是否遗漏经典论文。
- 是否包含最新论文。
- 引用是否支持句子 claim。
- 段落逻辑是否从经典到最新、从大类到细分。
- 是否过度声称。

## 9. Related Work 生成流程

### 9.1 输入阶段

用户可以提供任意一种输入：

```text
研究主题：例如 "retrieval-augmented generation for medical question answering"
关键词：RAG, medical QA, evidence retrieval
上传论文：自己的方法论文或参考论文
草稿：已有 introduction / method
目标会议：ACL / EMNLP / NeurIPS / Nature 子刊等
```

### 9.2 查询扩展

Agent 先提取：

```text
任务：medical question answering
方法：retrieval-augmented generation
领域：medical NLP
关键词：evidence retrieval, factual grounding, clinical QA
排除项：非医学、非问答、非 RAG
时间范围：经典论文不限，最新论文近 2-3 年
```

再生成多组检索 query：

```text
"retrieval augmented generation medical question answering"
"clinical question answering evidence retrieval"
"medical large language model retrieval augmented generation"
"RAG factual grounding medical NLP"
```

### 9.3 检索与入库

```text
调用 search_papers
→ 合并多来源结果
→ 根据 DOI / arXiv ID / 标题归一化去重
→ 保存 papers
→ 生成 title / abstract embedding
→ 标记来源和检索时间
```

### 9.4 排序

综合考虑：

```text
语义相关性
发表时间
引用数
venue 质量
是否 survey
是否有代码
是否被多篇核心论文引用
是否和用户草稿相似
```

### 9.5 聚类

输出类似：

```text
1. Classical retrieval-based QA
2. Neural retrievers and dense passage retrieval
3. Retrieval-augmented generation
4. Medical QA and clinical evidence grounding
5. Recent LLM-based medical RAG systems
```

### 9.6 精读候选论文

对每个 cluster 选：

```text
2-3 篇经典论文
2-3 篇最新论文
1 篇 survey 或 benchmark
1 篇与用户方法最接近的论文
```

如果摘要不足以支持 claim，则拉取全文缓存。

### 9.7 写作

生成大纲：

```text
Paragraph 1: Broad background and classical methods
Paragraph 2: Retrieval-augmented generation methods
Paragraph 3: Domain-specific medical QA systems
Paragraph 4: Gap and positioning of user's work
```

生成正文时，每个段落附带 claim map：

```json
{
  "paragraph": 2,
  "claims": [
    {
      "text": "Recent retrieval-augmented methods combine parametric and non-parametric knowledge.",
      "paper_ids": ["..."],
      "evidence_status": "verified"
    }
  ]
}
```

## 10. 防幻觉机制

必须作为系统硬约束，而不是提示词建议。

### 10.1 论文身份约束

```text
正式引用的论文必须存在于 papers 表。
papers 表中的论文必须来自外部来源或用户上传。
没有 DOI / arXiv ID 的论文，必须至少有 source_url 和检索时间。
```

### 10.2 写作约束

```text
Writer Agent 不能直接生成 citation。
Writer Agent 只能请求 citation service 插入引用。
每个引用位置必须绑定 paper_id。
```

### 10.3 Claim 约束

```text
每个重要 claim 都要有 evidence_papers。
如果证据不足，输出 [需要补充引用]。
如果工具没有找到论文，明确说没有找到，而不是猜。
```

### 10.4 引用一致性检查

在导出前运行：

```text
检查 citation_key 是否都存在。
检查 paper_id 是否都验证通过。
检查引用论文是否真的属于该段主题。
检查 BibTeX 是否完整。
```

## 11. 缓存与空间控制

不要存所有全文。

```text
全量论文库：元数据 + 摘要 + embedding
用户项目论文：元数据 + 摘要 + 结构化总结
正式引用论文：按需缓存全文
用户上传论文：保留全文
临时检索论文：7-30 天后清理全文缓存
长期不用论文：删除 chunks 和 PDF，只留元数据
```

缓存策略：

```text
uploaded: 永久保存，除非用户删除
selected_for_project: 保存 90 天
cited_in_export: 长期保存摘要和引用，全文可冷归档
temporary_candidate: 保存 7-30 天
```

## 12. API 初稿

### 12.1 创建写作项目

```http
POST /api/projects
```

```json
{
  "title": "Medical RAG Related Work",
  "topic": "retrieval-augmented generation for medical question answering",
  "target_venue": "ACL"
}
```

### 12.2 检索论文

```http
POST /api/projects/{project_id}/paper-search
```

```json
{
  "query": "retrieval-augmented generation for medical question answering",
  "include_classic": true,
  "include_recent": true,
  "recent_years": 3,
  "sources": ["arxiv", "semantic_scholar", "openalex"],
  "limit": 80
}
```

### 12.3 选择论文

```http
POST /api/projects/{project_id}/papers/select
```

```json
{
  "paper_ids": ["..."],
  "roles": {
    "paper_id_1": "classic",
    "paper_id_2": "recent"
  }
}
```

### 12.4 生成 Related Work 大纲

```http
POST /api/projects/{project_id}/related-work/outline
```

### 12.5 生成 Related Work 草稿

```http
POST /api/projects/{project_id}/related-work/draft
```

```json
{
  "citation_style": "apa",
  "language": "en",
  "length": "medium"
}
```

### 12.6 导出

```http
POST /api/projects/{project_id}/exports/docx
```

```json
{
  "draft_id": "...",
  "citation_style": "apa"
}
```

## 13. 第一阶段当前实现状态

截至 2026-07-08，当前仓库已经从“架构草案”推进到一个可运行的轻量 MVP。实现路线与上文目标架构略有差异：现在先采用 Python 标准库 HTTP 服务 + SQLite + 本地静态前端，暂不引入 FastAPI、Next.js、PostgreSQL、Celery 和 pgvector。这个选择适合早期验证链路，但还不是正式产品形态。

### 13.1 已完成

| 模块 | 当前状态 | 代码位置 |
| --- | --- | --- |
| 小型 Web 前端 | 已实现项目目标、来源选择、阶段清单、聊天、候选论文、已选证据、写作结构、生成文件、多对话切换 | `web/index.html`, `web/app.js`, `web/styles.css` |
| Kimi API 接入 | 已实现 Kimi K2.6 调用、流式 SSE 读取、temperature 兼容 | `src/paper_agent/llm.py`, `src/paper_agent/http_client.py` |
| 用户行为 intent | 已实现 `chat/search/answer/document` 四类行为识别，Kimi 优先、规则兜底 | `src/paper_agent/intent.py` |
| 研究检索 intent | 已实现研究方向、年份、会议/期刊、CCF A/B、SCI Q1-Q3、分源 query 解析 | `src/paper_agent/intent.py` |
| 多来源检索 | 已接入 arXiv、DBLP、Semantic Scholar、Google Scholar/SerpAPI | `src/paper_agent/search.py` |
| CCF/Sci 过滤 | 已实现基于配置文件的 CCF A/B 与 SCI Q1-Q3 过滤 | `src/paper_agent/venues.py`, `config/venues.json` |
| 本地缓存库 | 已从 JSON 升级为 SQLite，保存元数据、摘要、来源和 query 记录 | `src/paper_agent/store.py`, `data/papers.sqlite` |
| 缓存复用 | 检索前先查本地缓存，再补查外部来源 | `src/paper_agent/assistant_engine.py` |
| 去重与排序 | 已按 DOI、arXiv ID、URL、标题去重，并做轻量相关性排序 | `src/paper_agent/search.py`, `src/paper_agent/rank.py` |
| 写作草稿生成 | 已支持 Related Work/综述式草稿生成和 Kimi 生成，引用只来自当前论文池 | `src/paper_agent/related_work.py`, `src/paper_agent/kimi_writer.py` |
| BibTeX / Markdown / claim map | 已支持生成 `.md`、`.bib`、claim map JSON | `src/paper_agent/assistant_engine.py`, `outputs/` |
| Markdown / LaTeX 展示 | 前端已支持基础 Markdown 渲染和 MathJax LaTeX | `web/app.js`, `web/index.html` |
| 自动测试 | 已有 intent、venue、Related Work、SQLite 缓存等测试 | `tests/` |

### 13.2 与原第一阶段目标相比尚未完成

下面这些功能仍属于第一阶段应该补齐的内容，按优先级排序。

| 优先级 | 未完成功能 | 为什么重要 | 建议实现 |
| --- | --- | --- | --- |
| P0 | 上传论文 / 粘贴草稿解析 | 原目标支持“上传论文带读、根据用户论文写 Related Work”，现在还只能处理文本请求和外部检索 | 增加 PDF 上传接口、文本抽取、上传论文入库，并把用户论文作为检索 seed |
| P0 | 元数据补全与摘要补全 | DBLP 常没有摘要，Related Work 质量会受限 | 用 DOI、arXiv ID、Semantic Scholar、Crossref/OpenAlex 补全摘要、venue、citation_count |
| P0 | 引用一致性检查 | 现在有 claim map，但导出前没有独立 checker 阻止无效引用 | 新增 citation checker：检查 citation_key、paper_id、BibTeX、source_url、引用是否来自当前论文池 |
| P0 | 正式的论文选择机制 | 当前默认用排序前 N 篇生成，用户不能明确选择“纳入写作”的论文 | 前端增加“加入写作/排除/设为经典/设为最新/设为核心相关” |
| P1 | Related Work 大纲单独生成 | 现在直接生成正文，没有显式 outline 审阅阶段 | 增加 outline generation 和用户确认后再写正文 |
| P1 | 聚类与主题分组 | 现在只有排序，没有将论文分成方法、任务、数据集、应用方向 | 先用关键词/embedding-lite 聚类，后续接 pgvector 或本地向量库 |
| P1 | 经典论文检索 | 当前更偏近年检索，经典论文依赖排序和引用数，不稳定 | 单独生成 classic queries，放宽年份，按 citation_count 和 survey/benchmark 标记挑选 |
| P1 | Word 导出 | 原目标包含 Word，目前只导出 Markdown、BibTeX、JSON | 增加 `.docx` 导出，保留 citation_key 和参考文献 |
| P1 | 项目级数据库模型 | 现在是 session + 全局 paper cache，没有 `writing_projects/project_papers/drafts` | 在 SQLite 中补 `projects`、`project_papers`、`drafts`、`messages` |
| P1 | 对话持久化到后端 | 前端 localStorage 保存消息，服务重启后后端 session 状态丢失 | 把 messages、session state、selected papers 写入 SQLite |
| P2 | 全文按需缓存 | 当前不下载 PDF，也不做全文 chunk，因此复杂 claim 仍只能基于摘要 | 对 arXiv/open PDF 做按需下载、文本抽取、chunk 缓存和过期清理 |
| P2 | 向量检索 | 当前缓存复用是 token overlap，不能很好处理同义表达 | 先接本地 embedding 或 SQLite FTS5，后续迁移 pgvector/Qdrant |
| P2 | 数据源权威性维护 | CCF/SCI 列表是 seed list，不是权威同步 | 增加 CCF 目录、JCR/中科院分区导入脚本与更新时间字段 |
| P2 | 检索可观测性 | 用户看不到每个来源命中多少、失败原因、过滤原因 | 在前端展示 source stats、query plan、过滤统计、错误信息 |

### 13.3 一阶段建议补齐顺序

为了先提升 Related Work 质量，而不是过早做重平台化，建议按下面顺序继续：

1. **论文选择与引用检查**：前端支持选择核心论文，后端新增 citation checker。
2. **摘要/元数据补全**：对 DBLP 结果用 DOI、arXiv、Semantic Scholar、OpenAlex/Crossref 补摘要。
3. **上传论文解析**：支持上传 PDF 或粘贴论文文本，提取题目、摘要、关键词、研究贡献，作为检索 seed。
4. **Related Work 大纲阶段**：先生成 outline 和分组，再生成正文。
5. **项目级持久化**：补 `projects`、`project_papers`、`drafts`、`messages`，替代纯前端 localStorage。
6. **Word 导出**：生成 `.docx`，并带 BibTeX / claim map。
7. **全文按需缓存**：仅对用户选中的核心论文拉取全文，做 chunk 与证据句抽取。
8. **向量检索或 FTS5**：提高相似问题复用和论文聚类质量。

## 14. 第一阶段验收标准与当前差距

| 验收项 | 当前状态 | 说明 |
| --- | --- | --- |
| 用户输入研究方向，系统返回真实论文列表 | 基本完成 | 依赖 arXiv、DBLP、Semantic Scholar、Google Scholar/SerpAPI |
| 每篇论文有来源、作者、年份、摘要和链接 | 部分完成 | arXiv/Semantic Scholar 多数有摘要，DBLP 通常缺摘要，需要补全 |
| 系统区分经典论文和最新论文 | 部分完成 | 生成器内部有 classic/recent split，但检索阶段还没有稳定 classic pipeline |
| Related Work 引用可回溯到 paper record | 基本完成 | 引用来自当前论文池，但缺少导出前独立一致性检查 |
| 找不到证据时不编造论文 | 基本完成 | Kimi prompt 和生成器有约束，但还需要 citation checker 做硬约束 |
| 用户可以导出 Related Work 和 BibTeX | 基本完成 | 已有 Markdown、BibTeX、claim map；Word 未完成 |
| 用户可以上传论文带读 | 未完成 | 尚无上传、PDF 解析、用户论文入库 |
| 用户可以选择哪些论文进入写作 | 未完成 | 目前自动取排序结果前 N 篇 |
| 相似问题可复用历史检索 | 初步完成 | SQLite 缓存已实现，仍需 FTS/embedding 提升召回 |
| LaTeX/Markdown 在界面可读 | 基本完成 | 已有基础 Markdown 和 MathJax，仍不是完整编辑器 |

当前一阶段最关键的剩余工作不是继续扩大检索源，而是补齐“摘要/全文证据质量、用户选择、引用检查、项目持久化”。这几项完成后，系统才更接近可用于正式论文写作的 Related Work Agent。
