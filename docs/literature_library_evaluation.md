# 文献库资产化评估标准

本文档用于验收“文献库变成真正的项目资产”这一阶段目标。系统不应只展示一次性检索结果，而应能长期保存、合并、补全、标记和复用论文资产。

## 1. 数据结构完整性

每篇论文入库后必须至少支持以下字段：

- 基础元数据：`title`、`authors`、`year`、`venue`、`source`、`source_url`。
- 标识符：`doi`、`arxiv_id`，没有时允许为空，但不能影响标题级去重。
- 内容摘要：`abstract`、`abstract_status`。
- 影响力与领域：`citation_count`、`reference_count`、`fields_of_study`、`venue_rank`、`venue_reason`。
- 本地资产：`pdf_url`、`local_pdf_path`、`local_text_path`、`fulltext_status`、`fulltext_sha256`、`fulltext_downloaded_at`、`fulltext_error`。
- 用户状态：`reading_status`、`importance`、`user_tags`、`excluded`、`exclusion_reason`、`user_notes`、`used_in_sections`、`relevance_score`。
- 生命周期：`added_at`、`updated_at`、`retrieved_at`、`last_seen_at`。

验收标准：

- 新建数据库包含全部字段。
- 旧数据库自动迁移后也包含全部字段。
- `Paper.to_dict()` 和 API payload 能输出用户状态和本地资产字段。
- 缺失摘要的论文必须标记为 `abstract_status=none`，有摘要时为 `available`。

## 2. 去重与合并

入库时必须按多层规则识别同一篇论文：

- 第一层：`doi` 完全归一化匹配。
- 第二层：`arxiv_id` 去版本号后匹配。
- 第三层：标题归一化匹配，忽略大小写、标点、空格和连字符差异。
- 第四层：标题 + 年份 + 前 3 位作者匹配，用于处理不同来源 URL 造成的重复。

合并规则：

- 保留质量更高的摘要；更长且非空的摘要可以覆盖较短摘要。
- 合并来源，不能只保留最后一次来源。
- 保留最高引用数。
- 保留 DOI、arXiv、PDF、本地全文路径等补全信息。
- 用户手动状态不得被普通检索刷新覆盖，包括已读、核心、标签、排除、备注和用于章节。

评估指标：

- 构造同一论文的 DBLP、OpenAlex、arXiv、Google Scholar 四源样本，入库后应合并为 1 条。
- 去重精确率目标：人工抽样 100 组疑似重复，误合并不超过 2 组。
- 去重召回率目标：人工构造 50 组同文不同源样本，至少合并 48 组。

## 3. 元数据与摘要覆盖

文献库应优先保存摘要和可复用元数据，避免每次问答重新依赖外部检索。

验收标准：

- 外部来源返回摘要时必须入库。
- DBLP 等仅返回元数据的来源，应通过可用的补全流程尝试补摘要。
- `stats()` 返回 `paper_count`、`abstract_count`、`fulltext_count`、`query_count` 以及用户状态统计。
- 检索缓存命中时能直接返回摘要，而不是只返回标题和链接。

建议指标：

- 有摘要比例：普通计算机方向检索结果中，目标不低于 70%。
- 核心写作证据中，有摘要比例目标不低于 90%。
- 元数据完整率：标题、作者、年份、来源四项完整率不低于 95%。

## 4. PDF 与全文资产管理

数据库只存元数据、摘要和本地路径；PDF 原文和抽取后的全文 chunk 存本地文件夹。

验收标准：

- PDF 保存到 `data/paper_files/` 或对应配置目录。
- 抽取后的全文 chunk 保存到 `data/paper_texts/` 或对应配置目录。
- 数据库记录 `local_pdf_path`、`local_text_path` 和 `fulltext_status`。
- 无全文时不能伪装为已阅读全文，应标记 `none` 或 `failed` 并记录失败原因。
- 需要全文问答时，Agent 通过本地路径读取全文，不把全文直接塞进数据库。

建议指标：

- 已缓存 PDF 的论文，`local_pdf_path` 文件存在率应为 100%。
- 已抽取全文的论文，`local_text_path` 文件存在率应为 100%。
- 25 MB 以内普通 PDF 抽取成功率目标不低于 90%。

## 5. 用户状态与项目资产行为

文献库必须支持长期科研管理，而不是临时列表。

状态定义：

- `reading_status`：`unread`、`to_read`、`reading`、`read`。
- `importance`：`low`、`normal`、`high`、`core`。
- `excluded`：用户明确排除，不应进入默认缓存检索、证据池和写作。
- `used_in_sections`：记录论文被用于哪些章节，例如 `related_work`、`method_survey`。

验收标准：

- 用户标记状态后，重启服务仍保留。
- 同一论文再次从外部来源检索入库时，用户状态仍保留。
- 被排除论文仍保存在库中，但默认不进入写作证据池。
- API 能更新状态，并返回更新后的论文 payload。

## 6. 查询与复用

文献入库必须记录检索来源和历史 query，方便追踪为什么进入库中。

验收标准：

- `paper_queries` 记录 `paper_id`、`query`、`topic`、`source`、`searched_at`。
- 合并重复论文后，历史 query 必须指向合并后的 canonical paper。
- 本地缓存检索应基于标题、摘要、venue 召回，并排除 `excluded=true` 的论文。

建议指标：

- 复查某篇论文时，能找到它来自哪些 query 和 topic。
- 对同一主题重复检索，本地缓存能优先返回已有高相关论文。

## 7. 性能与空间

目标是“长期可用”而不是“把所有东西塞进数据库”。

验收标准：

- 1 万篇论文元数据查询在本地 SQLite 下应可接受，普通列表加载目标小于 1 秒。
- 数据库不保存 PDF 二进制和大段全文正文。
- PDF 和全文 chunk 使用文件路径引用。
- 去重合并对 1 万篇以内文献库应能在可交互时间内完成。

建议指标：

- 1 万篇论文元数据数据库体积通常应保持在百 MB 以内，具体取决于摘要长度。
- 全文文件夹可独立清理或备份，不影响元数据表可读性。

## 8. 测试门槛

每次修改文献库模块后，至少运行：

```bash
python -m compileall src tests
/opt/anaconda3/bin/python -m pytest tests/test_store.py tests/test_assistant_engine.py tests/test_web_app.py -q
```

必须覆盖：

- 新库建表和旧库迁移。
- 摘要、PDF 路径、全文状态字段持久化。
- DOI、arXiv、标题、作者年份维度去重。
- 重复入库后保留用户状态。
- 被排除论文不进入默认缓存命中和证据池。
- Web API 能更新论文资产状态。

## 9. 当前阶段完成定义

本阶段视为完成，需要满足：

- 数据模型和 SQLite 表支持上述资产字段。
- 入库去重不再只靠单一来源 URL。
- 检索刷新不会抹掉用户状态。
- 本地数据库保存摘要和元数据，本地文件夹保存 PDF/全文路径。
- 后端 API 可更新论文阅读、重要性、标签、排除和章节使用状态。
- 有明确测试证明这些行为。
- 本文档作为后续迭代的评估清单持续维护。
