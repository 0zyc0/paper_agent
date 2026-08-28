# 学术写作路由

这是唯一的学术写作 Skill。它只服务于用户**明确要求生成、撰写、改写、导出或扩写文稿**的场景；“调研、检索、了解进展、找论文”不是写作请求，必须停在论文检索或证据问答。

运行时先读取 `resources/writing-routing.md` 确认唯一交付物，再注入对应资源：`report.md`、`survey.md`、`related-work.md`、`introduction.md`、`method.md`、`experiment.md`、`summary.md`、`outline.md` 或 `bibliography.md`。不能因为已有论文就猜测用户需要大纲、Related Work 或报告。

所有分支共同遵守 `resources/markdown-contract.md`。先以用户原始请求确定写作目标，再使用当前证据写出完整正文；引用只能来自当前证据池。优先使用本地全文片段，题名或摘要只能用于高层定位。生成后提供 Markdown、BibTeX 和证据映射供用户核对。
