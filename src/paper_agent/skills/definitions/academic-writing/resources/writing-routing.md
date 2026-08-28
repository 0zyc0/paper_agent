# 写作场景路由

只根据用户明确提出的主要交付物选择一个资源；不要从研究主题、已有论文或“调研”一词猜测文稿类型。若用户没有明确要求写作，不进入本 Skill。若已明确写作但交付物不清楚，选择 `general`。

- `report`：研究报告、调研报告、基于当前文献库形成研究判断。
- `survey`：survey、综述、文献回顾、按路线总结研究进展。
- `related-work`：Related Work、相关工作、论文中先前工作章节。
- `introduction`：Introduction、引言、研究背景与问题定位。
- `method`：Method、方法章节、模型或方案设计。
- `experiment`：Experiment、实验章节、评测设计、实验方案。
- `summary`：总结、概述、压缩现有文献发现。
- `outline`：大纲、章节结构、论文框架。
- `bibliography`：BibTeX、`.bib`、引用导出。
- `general`：其他明确写作但无法归入上述类型的请求。

路由只决定写作结构，不决定是否检索论文。是否需要新检索由上层意图和工具计划决定；“基于当前文献库”必须复用现有证据。每个非 `outline`、非 `bibliography` 分支都必须写出完整段落，不能只返回章节标题或待填占位符。
