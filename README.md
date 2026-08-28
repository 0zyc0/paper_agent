# 科研写作与论文调研 Agent MVP

这是一个初步可运行的论文助手核心代码。它目前专注一条主链路，并逐步扩展到综述、论文章节、论文总结和科研方向建议：

```text
输入研究主题
→ 检索最新与相关论文
→ 去重、排序、保存本地缓存
→ 按需求生成写作草稿
→ 生成 BibTeX
```

第一版刻意不把所有论文全文存进本地，只保存论文元数据、摘要、来源链接和检索 query 记录。Web 端默认使用 `data/papers.sqlite` 作为本地缓存库；生成内容中的引用只来自检索到的真实论文记录。

文献库正在按“长期科研资产”方向演进：数据库保存元数据、摘要、用户阅读状态、重要性、标签、排除原因和章节使用记录；PDF 原文与全文 chunk 保存在本地文件夹并由数据库记录路径。详细验收口径见 `docs/literature_library_evaluation.md`。

产品目标、功能边界、Agent/Tool/Skill 设计、数据模型、接口、阶段优先级和发布验收标准见 [docs/product_requirements.md](docs/product_requirements.md)。

## 项目结构

核心实现现在按职责拆分，避免把 Agent 编排、外部 API 和技能提示词堆在同一目录：

```text
src/paper_agent/
├── core/          # Agent 编排、意图模型、任务准备、PDF 证据、领域模型、排序与文档生成
├── tools/         # 外部 API 适配器与注册式工具契约
├── skills/        # 可版本化的领域工作流定义（SKILL.md）与 Skill 选择器
├── storage/       # SQLite / JSON 本地缓存
├── interfaces/    # CLI 与 Web 服务入口
├── local_config.py
└── __init__.py
```

依赖方向为：`interfaces -> core -> tools / skills / storage`。新代码应放入对应目录，包根目录只保留配置与包定义。

### Agent 架构

系统采用“语义理解、能力选择、工具执行”三层架构。Kimi 只输出任务语义；程序依据会话状态编译工具链，因此模型不能把“基于当前证据池写 Introduction”擅自改成外部检索。

```text
Web / CLI
  -> AgentRequestContext（会话、证据、PDF、已生成文档）
  -> PaperAgentOrchestrator
       -> IntentAnalyzer（Kimi 结构化语义）
       -> PaperToolRegistry（工具前置条件与动态裁剪）
       -> PaperSkillCatalog（选择领域 SKILL.md）
  -> ResearchAssistantEngine（执行已验证的工具链、持久化状态）
  -> SQLite / 本地 PDF / 外部论文来源
```

- Tool：`paper_search`、`pdf_read`、`paper_fulltext_read`、`evidence_answer`、`write_document`、`document_inspect`、`free_chat`。工具契约和可用条件统一定义在 `src/paper_agent/tools/registry.py`。
- Skill：论文检索、证据问答、PDF 精读、研究发现，以及按写作目标拆分的研究报告、综述、Related Work、引言、方法设计、实验设计、总结、大纲和 BibTeX 导出。每个 Skill 以独立 `SKILL.md` 保存工作流规则，位于 `src/paper_agent/skills/definitions/`。
- Context：`data/sessions.json` 保存会话短期状态；SQLite 保存长期文献资产；PDF 正文保存在个人文献库文件夹。

### 写作 Skill 与证据边界

`write_document` 不再把所有请求套进 Related Work 模板。路由始终选择统一的 `academic-writing` Skill：先读取 `resources/writing-routing.md` 确认交付物，再加载对应 resource 和 `resources/markdown-contract.md`：

```text
研究报告       -> academic-writing/resources/report.md
综述           -> academic-writing/resources/survey.md
相关工作       -> academic-writing/resources/related-work.md
引言           -> academic-writing/resources/introduction.md
方法设计草案   -> academic-writing/resources/method.md
实验设计       -> academic-writing/resources/experiment.md
文献总结       -> academic-writing/resources/summary.md
大纲 / BibTeX  -> academic-writing/resources/outline.md / bibliography.md
```

写作时优先使用本地 PDF 提取的相关片段；没有全文的论文只能用于高层定位。方法和实验 Skill 会明确保留待作者补充的信息，不能凭已有文献虚构“本文方法”或实验结果。模型草稿会检查标题结构、引用是否来自证据池、引用覆盖和重复段落；未通过时才使用同样按写作目标组织的离线草稿，并在质量报告中标记为兜底结果。

未来若要接入 LangGraph、MCP 或其他模型，只需让新的运行时消费 `PaperToolRegistry` 和 `PaperSkillCatalog`，不需要重写检索、全文阅读或写作服务。

## 快速运行

启动小型前端：

```bash
python run.py web --port 8765
```

如果上传或下载论文后提示 `服务端未安装 pypdf`，说明当前 `python` 环境没有安装 PDF 解析依赖。可以改用已安装依赖的 Anaconda Python 启动：

```bash
/opt/anaconda3/bin/python run.py web --port 8765
```

或者在当前 Python 环境安装项目依赖：

```bash
python -m pip install -e .
```

打开：

```text
http://127.0.0.1:8765
```

停止或重启服务：

```bash
# 在正在运行服务的终端里停止
Ctrl+C

# 重新启动
python run.py web --port 8765
```

如果提示端口已被占用，先找到占用进程，再结束它：

```bash
lsof -nP -iTCP:8765 -sTCP:LISTEN
kill <PID>
python run.py web --port 8765
```

前端支持：

```text
实时聊天问答
自由聊天模式：直接调用 Kimi，不触发论文检索
项目目标、写作任务、目标章节和检索来源选择
自动识别用户想调研的方向、年份和会议/期刊
按要求检索 arXiv / DBLP / Semantic Scholar / Google Scholar
候选论文选入证据池
按阶段清单组织写作流程
根据当前检索结果回答问题
上传 PDF 后进行带页码引用的论文问答
生成综述 / 章节 / Related Work / BibTeX / claim map
```

```bash
python run.py related-work \
  --query "retrieval augmented generation for medical question answering" \
  --limit 20 \
  --language en
```

输出文件：

```text
outputs/related_work.md
outputs/references.bib
data/papers.json
```

Web 端检索缓存：

```text
data/papers.sqlite
```

用户上传的 PDF 和点击“下载 PDF”保存的开放论文原文会统一进入个人文献库文件夹：

```text
data/paper_files/pdf/
```

数据库只保存论文元数据、摘要、阅读状态和本地 PDF 路径，PDF 原文不会塞进 SQLite。这样 DBeaver 里可以查看索引状态，本地文件夹负责长期保存原文。

Web 端项目会话会自动持久化，服务重启后会恢复当前项目的文献池、已选论文、上传 PDF 解析记录、生成文件记录和最近上下文：

```text
data/sessions.json
outputs/
```

如果只想重置 Web 会话，可以先停止服务，再删除这些本地文件：

```bash
rm -f data/sessions.json
rm -rf outputs
```

如果要彻底清空个人文献库，再额外删除：

```bash
rm -rf data/paper_files/pdf
rm -rf data/paper_texts
```

## 计算机领域自动调研

如果用户只是自然语言描述方向，可以使用 `investigate`。系统会先判断研究方向、年份范围和目标会议/期刊，再检索 arXiv、DBLP、Semantic Scholar，并过滤配置内 CCF A/B 会议和 SCI Q1-Q3 期刊。

```bash
python run.py investigate \
  --request "我想调研大模型智能体在软件漏洞检测中的最新论文" \
  --limit 20 \
  --language zh
```

指定近几年和会议/期刊也可以直接写在请求里：

```bash
python run.py investigate \
  --request "我想查近五年 ACL 上 RAG 相关论文" \
  --sources dblp \
  --language zh
```

也可以显式传参：

```bash
python run.py investigate \
  --request "RAG evaluation" \
  --venue ACL \
  --recent-years 5 \
  --sources dblp \
  --language zh
```

输出文件：

```text
outputs/investigation_related_work.md
outputs/investigation_references.bib
outputs/investigation_claim_map.json
outputs/intent.json
```

## 使用 Kimi 作为 Agent LLM

配置 API key 后，`investigate` 会使用 Kimi 做研究意图识别和 Related Work 起草。Web Agent 的意图路由完全由 Kimi 的结构化 JSON 决定；本地代码只校验工具是否可用，不会用关键词或正则表达式把模型的决定改写成另一种任务。

如果 Kimi 未配置、超时或返回无效 JSON，系统会安全保留 `free_chat` 入口并提示路由不可用，**不会**猜测研究方向后自动检索。这避免了在模型不可用时把普通对话误判为论文调研。

方式一：直接在代码里填写：

```python
# src/paper_agent/local_config.py
KIMI_API_KEY = "your_key"
KIMI_MODEL = "kimi-k2.6"
KIMI_API_BASE = "https://api.moonshot.cn/v1"
SEMANTIC_SCHOLAR_API_KEY = "your_semantic_scholar_key"
SERPAPI_API_KEY = "your_serpapi_key"
OPENALEX_API_KEY = "your_openalex_key"
OPENALEX_MAILTO = "your_email@example.com"
RSSHUB_BASE_URL = ""
DISCOVERY_RSS_FEEDS = [
  # {"name": "Example Tech RSS", "url": "https://example.com/search/rss?q={query}"},
]
```

方式二：使用环境变量：

```bash
export KIMI_API_KEY="your_key"
export KIMI_MODEL="kimi-k2.6"
export KIMI_API_BASE="https://api.moonshot.cn/v1"
export SEMANTIC_SCHOLAR_API_KEY="your_semantic_scholar_key"
export SERPAPI_API_KEY="your_serpapi_key"
export OPENALEX_API_KEY="your_openalex_key"
export OPENALEX_MAILTO="your_email@example.com"
export RSSHUB_BASE_URL="https://your-rsshub.example.com"
```

OpenAlex 的免费 API key 可以在 `https://openalex.org/settings/api` 申请。系统会把
`OPENALEX_API_KEY` 作为请求参数 `api_key` 发送；`OPENALEX_MAILTO` 用于 OpenAlex
polite pool 和联系邮箱标识。环境变量优先级高于 `src/paper_agent/local_config.py`。

发现模块默认使用 OpenAlex、arXiv、DBLP 获取近期论文，并提供微信公众号、CSDN、知乎、
GitHub 的主题搜索入口。公众号、CSDN、知乎没有稳定的官方免费检索 API；如果需要直接把
这些平台的内容流接入发现页，可以配置 `RSSHUB_BASE_URL` 或 `DISCOVERY_RSS_FEEDS`。
RSSHub 路由是否可用取决于你部署的实例和平台反爬状态，因此这部分只作为技术动态补充，
不作为论文引用证据。

意图识别是通用科研领域抽取，不是针对某个固定方向。Kimi 可用时会由 LLM 从用户原文中抽取真实细分领域，例如：

```text
动态推荐系统 -> dynamic recommender systems
目标检测 -> object detection
开放词汇目标检测 -> open-vocabulary object detection
去偏推荐系统 -> debiasing recommender systems
医学图像分割 -> medical image segmentation
```

意图识别采用“类别 -> 子任务/交付物 -> 工具链”的两阶段系统提示词。模型先识别用户是在自由问答、检索、证据问答、PDF 阅读、写作、文件检查还是发现科研进展，再输出可执行工具链。核心 JSON 字段包括：

```json
{
  "category": "document_writing",
  "subtask": "current_evidence_survey",
  "deliverable": "survey",
  "evidence_scope": "current_evidence",
  "tool_plan": ["write_document"],
  "needs_fresh_literature": false,
  "normalized_topic": "debiasing recommender systems",
  "cs_area": "AI",
  "keywords": ["debiasing", "recommender", "systems"],
  "queries": ["debiasing recommender systems"],
  "source_queries": {
    "dblp": ["debiasing recommender systems", "debiasing recommendation"],
    "arxiv": ["bias mitigation recommender systems"],
    "semantic_scholar": ["debiasing recommender systems"],
    "google_scholar": ["\"debiasing recommender systems\""]
  },
  "source_urls": {
    "google_scholar": ["https://scholar.google.com/scholar?q=..."]
  },
  "target_venues": [],
  "recent_years": null,
  "from_year": null,
  "to_year": null
}
```

其中 `free_chat` 是永久保留的自由问答工具：例如 `hello`、研究思路讨论、非证据依赖的普通交流，模型会返回 `category: "chat"` 与 `tool_plan: ["free_chat"]`。当前已有论文或 PDF 不会自动把这类对话改成检索或证据问答。

`--no-kimi` 可用于离线检索或在已有证据上生成确定性写作草稿，但不会启用自动意图识别和自动工具路由：

```bash
python run.py investigate \
  --request "我想调研RAG在代码生成中的应用" \
  --no-kimi
```

## 目标发表源配置

目标范围在 [config/venues.json](/Users/shadow/Documents/Mypro/config/venues.json) 中维护：

```text
arXiv：默认保留计算机相关分类
CCF A/B：主要通过 DBLP 会议元数据和会议白名单过滤
SCI Q1-Q3：通过期刊白名单过滤
```

当前配置是 MVP seed list，不应当视为权威完整目录。正式产品建议接入官方 CCF 目录和 JCR / 中科院分区等稳定数据源，定期同步。DBLP 通常提供标题、作者、年份、venue、DOI/链接等元数据，不一定提供摘要；需要更细的 Related Work 时，可以再用 DOI/链接去补全文或摘要。

## 只检索论文

```bash
python run.py search \
  --query "large language model agents tool use" \
  --limit 30
```

## 数据来源

默认使用：

- arXiv
- DBLP
- Semantic Scholar
- Google Scholar

Semantic Scholar 如果配置了 API key，会自动使用：

```bash
export SEMANTIC_SCHOLAR_API_KEY="your_key"
```

没有 key 也可以使用公开接口，但可能限流。

Google Scholar 没有稳定官方免费 API。系统会始终生成可打开的 Google Scholar 查询链接；如果要自动抓取 Scholar 结果，请配置 SerpAPI：

```bash
export SERPAPI_API_KEY="your_key"
```

## 当前边界

- 这是 MVP，不是完整论文管理系统。
- 目前不下载全文 PDF，只使用标题、摘要、作者、年份、来源链接生成 Related Work 初稿。
- Web 端已有 SQLite 元数据/摘要缓存，但还没有项目级论文选择、全文证据抽取和引用一致性检查。
- 如果要写正式论文，建议下一步加入全文按需缓存和 claim-level evidence 检查。
- CCF 和 SCI 分区需要维护配置或接入权威数据源；本仓库里的列表只是初始示例。
