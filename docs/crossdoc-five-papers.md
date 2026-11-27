# 跨文献任务：比较五篇纵向常模论文

> 2.2 跨文献复杂任务的验收任务。真实任务，用 Planner-Executor 跑通。
> 本地验证：把下面的问题直接问 Agent（后端跑起来后），检查输出的对比表格。

## 任务问题

比较五篇纵向常模论文（Bethlehem 2022、Di Biase 2023、Ge 2024、Seidel 2026、Rutherford 2022）的数据集、建模变量、模型、结果和局限性。

## 五篇论文（都在开发集里）

| 论文 | 文件 | 数据集 | 模型要点 |
|---|---|---|---|
| Bethlehem 2022 | 2022-bethlehem-brain-charts.pdf | 123,984 次 MRI 扫描，101,457 人，受孕后 115 天到 100 岁 | 脑图谱，centile 分数，GAMLSS |
| Di Biase 2023 | 2023-dibiase-mapping-brain-charts.pdf | UK Biobank（47–80 岁，约 2752–2832 人）+ ABCD（9–11 岁，约 6537–7480 人） | 横断面 vs 纵向对比；横断面低估纵向变化，皮层厚度最多低估 44%（60 岁时） |
| Ge 2024（CentileBrain） | 2024-ge-centilebrain.pdf | 37,407 名健康人，87 个数据集，3–90 岁 | 比较 8 种算法，MFPR 最优；样本量超 3000 后性能趋于平稳 |
| Seidel 2026 | 2026-seidel-baseline-cortical-thickness.pdf | ABCD 研究，6000 多名青少年 | B-Norm（基线厚度 + 基线年龄 + 随访年龄）vs C-Norm（只用年龄）；B-Norm 预测更准、对青春期更敏感 |
| Rutherford 2022 | 2022-rutherford-brain-growth-aging-precision.pdf | 58,836 人，82 个站点，2–100 岁 | 脑生长常模建模 |

注意：开发集里有两篇 Rutherford 2022，本任务用脑生长那篇（2022-rutherford-brain-growth-aging-precision.pdf），不是精神病学框架那篇（2022-rutherford-normative-modeling-framework-psychiatry.pdf）。

## 比较维度（5 个）

数据集、建模变量、模型、结果、局限性。

## 期望的执行路径

Router（“比较”“五篇”触发 requires_planning）→ Planner 按维度拆成 5 个子任务（数据集、建模变量、模型、结果、局限性各一个，不按论文拆——这是 Planner prompt 的既定规则）→ Executor 每个子任务分别检索（local_search）→ Answerer 汇总成对比表格 → Verifier 逐项核查引用。

## 验收标准

- 对比表格每行都有引用，引用真实存在（论文 + 页码在检索结果里）。
- 五篇论文、五个维度都要覆盖到，漏掉任何一篇/一个维度都不算跑通。
- 有 1 个维度（或 1 篇）证据不足时，输出 partial（已核实的部分正常给，没证据的部分标出来），而不是全拒答。
- 引用编造（引了不在检索结果里的页码）要被核查拦住、重答，编造 0 次。

## 已知背景

- 上一轮 40 道评测里 Q18 跨文献题内容答对了，但只引了 Pomponio、没引到 Fortin 原文。这就是跨文献引用的短板：单次检索只召回一篇论文的片段，答案就只能引那一篇。本任务用 Planner 按子任务分别检索，正是要对着这个问题。
- 核查缺口（2.2 修掉）：已核实的子任务证据如果根本没被草稿引用，原核查会直接报 verified——答案悄悄漏了一个维度/一篇论文，系统却说通过。修法是核查加覆盖检查：已核实的子任务至少要被引用一次，否则打回回答员重答；重试用完则 partial。
