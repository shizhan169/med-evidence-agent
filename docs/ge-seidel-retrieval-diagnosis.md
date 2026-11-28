# Ge 2024 和 Seidel 2026 五维对比表拒答诊断

**结论。** B 方案已经把两篇原文送进候选集合。问题出在按篇过滤后的检索词仍然带着五篇作者和年份。五个维度的 BM25 输入完全一样，两篇的向量检索首位也全是参考文献。RRF 将双路同时命中的文末条目排到前面，每篇取两条后，回答员在两篇上共看到 20 条次参考文献，合并去重后仅 7 个不同片段。拒答符合实际证据。

**只调配额无效。** 每篇的 RRF 前四在全部十个「论文 × 维度」组合上都是参考文献。原配额 2 没有触发字符截断。调到 3 后五个维度各只剩一条 Seidel，调到 4 后五个维度都丢掉了 Ge 和 Seidel。优先修正按篇过滤后的内容查询。

## 复现依据

- 本地提交 `1b4876b6ea958d019f4925ca6e1080aecd169b20`。LangSmith 中与原问题完全匹配的 B 方案执行记录为 `cb76f59f-88eb-4789-a677-e8e9033773b8`，开始于 2026-11-28 21:54（北京时间）。同题较早一轮每维 10 条，本报告只用每维 12 条的 B 方案记录。
- 使用 `medevidence` 环境、本地 Chroma 索引的临时副本（5,085 个片段）、本地 BGE-M3 和实际 BM25。重排器按当前配置关闭。逐字取出该轮五次 `local_search` 的参数，直接调用 `retrieve(**params)`。五次返回的文件、页码、全文及顺序和历史工具输出全部相同，共 60/60 条。
- 同一次回放记录向量和 BM25 的每篇前 15 条候选。配额对照使用相同候选、原 `_rrf_fuse`、六个文件原顺序和原 `_assemble_context`。英文查询对照另行调用真实 `_retrieve_per_paper`。诊断止于送往回答员的证据，未重新生成整张答案。
- 下文页码均从 PDF 首页按 1 计数。Ge 的 PDF p.1 对应期刊 e211。原文可在 [Ge 2024](../data/papers/2024-ge-centilebrain.pdf) 和 [Seidel 2026](../data/papers/2026-seidel-baseline-cortical-thickness.pdf) 核对。

## 五次工具调用对应的真实检索词

- **数据集**　Bethlehem 2022、Di Biase 2023、Ge 2024、Seidel 2026 和 Rutherford 2022 各使用了什么数据集、样本、人群和纵向设计？
- **建模变量**　Bethlehem 2022、Di Biase 2023、Ge 2024、Seidel 2026 和 Rutherford 2022 在建模中分别纳入哪些变量、协变量、暴露或结局定义？
- **模型**　Bethlehem 2022, Di Biase 2023, Ge 2024, Seidel 2026, Rutherford 2022 各自采用的统计模型或纵向常模构建方法
- **结果**　Bethlehem 2022、Di Biase 2023、Ge 2024、Seidel 2026 和 Rutherford 2022 的主要结果、常模发现或关键估计是什么？
- **局限性**　Bethlehem 2022、Di Biase 2023、Ge 2024、Seidel 2026 和 Rutherford 2022 分别报告了哪些局限性、偏倚或外部效度限制？

## 实际返回的片段

Ge 的 source 为 `2024-ge-centilebrain.pdf`，Seidel 的 source 为 `2026-seidel-baseline-cortical-thickness.pdf`。名次从 1 开始。V 和 B 分别是该片段在篇内向量和 BM25 的名次。末列直接截取工具返回的正文前 100 个字符，保留 PDF 抽取时的断字。相同页码可以对应不同片段。

| 维度 | 来源 | PDF 页 | RRF | V | B | 内容摘要 | 前 100 字符 |
|---|---|---:|---:|---:|---:|---|---|
| 数据集 | Ge | 11 | 1 | 1 | 2 | Bayer、Haas、Kia 等引文 | `Mol Psychiatry 2021; 26: 3512–23.10 Bayer JMM, Dinga R, Kia SM, et al. Accommodating site variation ` |
| 数据集 | Ge | 11 | 2 | 4 | 4 | Rutherford 脑生长研究等引文 | `PLoS One 2022; 17: e0278776. 13 Rutherford S, Fraza C, Dinga R, et al. Charting brain growth and agi` |
| 数据集 | Seidel | 16 | 1 | 4 | 3 | Verdi 等关于阿尔茨海默病的引文 | `Brain. 2021;144(10):2946–53. 26. Verdi S, Kia SM, Yong KXX, Tosun D, Schott JM, Marquand AF et al. R` |
| 数据集 | Seidel | 15 | 2 | 8 | 1 | Rutherford、Di Biase、Frangou 等引文 | `Rutherford S, Barkema P , Tso IF, Sripada C, Beckmann CF, Ruhe HG, et al. Evidence for embracing nor` |
| 建模变量 | Ge | 11 | 1 | 1 | 2 | Bayer、Haas、Kia 等引文 | `Mol Psychiatry 2021; 26: 3512–23.10 Bayer JMM, Dinga R, Kia SM, et al. Accommodating site variation ` |
| 建模变量 | Ge | 11 | 2 | 6 | 4 | Rutherford 脑生长研究等引文 | `PLoS One 2022; 17: e0278776. 13 Rutherford S, Fraza C, Dinga R, et al. Charting brain growth and agi` |
| 建模变量 | Seidel | 16 | 1 | 2 | 3 | Verdi 等关于阿尔茨海默病的引文 | `Brain. 2021;144(10):2946–53. 26. Verdi S, Kia SM, Yong KXX, Tosun D, Schott JM, Marquand AF et al. R` |
| 建模变量 | Seidel | 16 | 2 | 6 | 5 | Marquand 模型评价及青春期研究引文 | `Nat Neurosci. 2019;22(10):1617–23. 55. Marquand A, Rutherford S, Dinga R. Fairly evaluating the perf` |
| 模型 | Ge | 11 | 1 | 1 | 2 | Bayer、Haas、Kia 等引文 | `Mol Psychiatry 2021; 26: 3512–23.10 Bayer JMM, Dinga R, Kia SM, et al. Accommodating site variation ` |
| 模型 | Ge | 11 | 2 | 7 | 1 | ADHD、自闭症和精神分裂症常模研究引文 | `Individual differences v. the average patient: mapping the heterogeneity in ADHD using normative mod` |
| 模型 | Seidel | 15 | 1 | 6 | 1 | Rutherford、Di Biase、Frangou 等引文 | `Rutherford S, Barkema P , Tso IF, Sripada C, Beckmann CF, Ruhe HG, et al. Evidence for embracing nor` |
| 模型 | Seidel | 16 | 2 | 4 | 3 | Verdi 等关于阿尔茨海默病的引文 | `Brain. 2021;144(10):2946–53. 26. Verdi S, Kia SM, Yong KXX, Tosun D, Schott JM, Marquand AF et al. R` |
| 结果 | Ge | 11 | 1 | 1 | 2 | Bayer、Haas、Kia 等引文 | `Mol Psychiatry 2021; 26: 3512–23.10 Bayer JMM, Dinga R, Kia SM, et al. Accommodating site variation ` |
| 结果 | Ge | 11 | 2 | 2 | 4 | Rutherford 脑生长研究等引文 | `PLoS One 2022; 17: e0278776. 13 Rutherford S, Fraza C, Dinga R, et al. Charting brain growth and agi` |
| 结果 | Seidel | 15 | 1 | 6 | 1 | Rutherford、Di Biase、Frangou 等引文 | `Rutherford S, Barkema P , Tso IF, Sripada C, Beckmann CF, Ruhe HG, et al. Evidence for embracing nor` |
| 结果 | Seidel | 16 | 2 | 5 | 3 | Verdi 等关于阿尔茨海默病的引文 | `Brain. 2021;144(10):2946–53. 26. Verdi S, Kia SM, Yong KXX, Tosun D, Schott JM, Marquand AF et al. R` |
| 局限性 | Ge | 11 | 1 | 1 | 2 | Bayer、Haas、Kia 等引文 | `Mol Psychiatry 2021; 26: 3512–23.10 Bayer JMM, Dinga R, Kia SM, et al. Accommodating site variation ` |
| 局限性 | Ge | 11 | 2 | 2 | 4 | Rutherford 脑生长研究等引文 | `PLoS One 2022; 17: e0278776. 13 Rutherford S, Fraza C, Dinga R, et al. Charting brain growth and agi` |
| 局限性 | Seidel | 15 | 1 | 1 | 8 | Marquand、Rutherford、Bethlehem 等引文 | `Marquand AF, Rezek I, Buitelaar J, Beckmann CF. Understanding heterogene - ity in clinical cohorts u` |
| 局限性 | Seidel | 15 | 2 | 8 | 1 | Rutherford、Di Biase、Frangou 等引文 | `Rutherford S, Barkema P , Tso IF, Sripada C, Beckmann CF, Ruhe HG, et al. Evidence for embracing nor` |

答案只引用了 Seidel p.15，实际参与回答的还有 p.16。五个维度合并后，Ge 有 3 个不同片段，Seidel 有 4 个。执行器按文件、页码和全文去重，再把合并结果交给回答员。

## PDF 原文中的证据

| 维度 | Ge 2024（CentileBrain） | Seidel 2026 |
|---|---|---|
| 数据集 | **p.1、6**，87 个数据集，20 个国家，共 37,407 名健康参与者，3–90 岁。**p.3**，独立纵向验证样本 SLIM（118 人）、QTAB（259 人）。主常模以横断面数据建立，另用重复扫描验证稳定性。 | **p.1–3、5**，ABCD 5.1。基线和两年随访各 6,179 人，用其中 5,374 人训练，另留有全部三次测量的 805 人独立测试，包含四年随访。 |
| 建模变量 | **p.1–3、5–6**，按性别和脑区建模，结局为皮层下体积、皮层厚度或表面积。优化模型用年龄的非线性项和相应全脑指标的线性项。扫描仪厂商、FreeSurfer 版本、Euler 数等属于评估过的候选协变量。 | **p.2–3、5**，预测每个脑区的皮层厚度，左右半球各 180 区，按性别建模。C-Norm 用年龄和站点，B-Norm 用基线厚度、基线年龄、随访年龄和站点。青春期发育评分用于验证与偏离分数的关联。 |
| 模型 | **p.2–3**，八种算法为 OLSR、BLR、GAMLSS、LMS、GPR、HBR、WBLR、MFPR。**p.6–7** 说明为何选择多变量分数多项式回归（MFPR）。 | **p.5** 给出两种模型的公式，均使用带似然变换的贝叶斯线性回归（PCNtoolkit）。C-Norm 和 B-Norm 的主要区别在输入变量。 |
| 结果 | **p.6–7**，优化后的 MFPR、WBLR、GPR 准确性相近，MFPR 兼顾准确性和计算效率。约 3,000 人后性能趋于平台。**p.8**，两套独立队列约两年间隔的区域 Z 分数具有稳定性。 | **p.2、6、8、12**，两年和四年测试中，B-Norm 在几乎所有脑区的拟合更好，解释方差更高。**p.10–14**，女性部分脑区或脑叶的偏离与青春期阶段有关，作者对效应强度及协变量调整后的稳健性持保留意见。 |
| 局限性 | **p.10**，年轻及中年成人和长期随访不足，特定祖源的泛化待检验，未穷尽可能解释脑形态差异的变量。 | **p.14，Limitations**，年龄范围和 ABCD 队列限制外推，许多男性参与者尚未完成青春期。加入 BMI 和社会经济状况后关联不再显著，家庭相关性与样本量下降的影响也难以区分。 |

## 问题发生在哪一步

1. **查询进入两路排名时已经失去重点。** 执行器走 `app/tools.py:96–100`，调用 `retrieve(query, top_k=top_k)` 时不传 `llm_call`。向量端用整句中文问题加五组作者年份。BM25 在 `app/rag.py:92` 提取英语词和数字，五次都得到同一组 `2022, 2023, 2024, 2026, bethlehem, biase, di, ge, rutherford, seidel`。维度词全部消失，已用作文件过滤的作者和年份却继续参与打分。两篇参考文献里这些姓名、年份集中出现。
2. **两路都把引文排在前面。** 五次 Ge 的向量首位全是同一条 p.11，BM25 首位也全是 p.11。五次 Seidel 的向量首位分别为 p.16、15、16、16、15 的引文，BM25 首位固定为 p.15 的 Rutherford 等引文。单独去掉 BM25 仍会在这两个文件上首选引文。
3. **RRF 进一步奖励两路共有的条目。** 例如 Ge 模型维度，p.2 列出八种算法的正文在向量排第 2，未进 BM25 前 15，融合降至第 6。相反，p.11 的 Bayer 等引文在两路排第 1 和第 2，融合第 1。Seidel 局限性维度，p.14 讨论 BMI、SES 的正文向量第 3，未进 BM25 前 15，融合后第 9。依据相同文本的人名匹配被计入了两路分数。
4. **前四之外确实有可用证据，但单看页码会高估召回。** 下表列出原始池中核过片段全文的例子。数字为 RRF 名次，并非只要放宽到该名次就足以填完整单元格。

| 维度 | Ge 篇内证据举例 | Seidel 篇内证据举例 |
|---|---|---|
| 数据集 | p.1 的 87 个数据集与入组条件第 15；第 6 虽也在 p.1，却是作者机构。 | p.5 的分组流程第 10，p.1 明列训练 5,374 人及测试 805 人的片段第 19。 |
| 建模变量 | p.2 的全脑测量候选项第 7，p.3 详细协变量第 17。 | p.5 第 11 是用于比较模型性能的 ANOVA 因变量，并未给出预测皮层厚度的输入公式。 |
| 模型 | p.2 八算法流程图说明第 6。 | 候选里虽有 p.5（第 15），内容是数据合并与质控；说明带似然变换 BLR 的方法段未进两路前 15 的并集。 |
| 结果 | p.7 各年龄组误差稳定性的结果第 14。 | p.8 脑区解释方差的结果第 18。 |
| 局限性 | p.10 横断面建模与长期随访限制第 11，含祖源泛化与遗漏协变量的明确局限段第 20。 | p.14 BMI/SES 敏感性第 9，ABCD 人群外推限制第 19。 |

## 配额 2、3、4 的对照

每格为预算后的「总片段数；Ge / Seidel 条数；正文字符数」。12000 的单位是字符，每次工具调用各算一次。下表的正文字符数也包括参考文献片段中的文字。

| 维度 | 每篇 2（入选 12 条） | 每篇 3（入选 18 条） | 每篇 4（入选 24 条） |
|---|---|---|---|
| 数据集 | 12；2 / 2；8,553 | 16；3 / 1；11,593 | 16；0 / 0；11,754 |
| 建模变量 | 12；2 / 2；8,629 | 16；3 / 1；11,304 | 16；0 / 0；11,490 |
| 模型 | 12；2 / 2；8,374 | 16；3 / 1；11,465 | 16；0 / 0；11,453 |
| 结果 | 12；2 / 2；9,023 | 16；3 / 1；11,559 | 16；0 / 0；11,747 |
| 局限性 | 12；2 / 2；9,003 | 16；3 / 1；11,765 | 15；0 / 0；11,634 |

六个文件按文件名排序，Ge 和 Seidel 恰在第 5、6 位。每篇内部按 RRF 排好后整篇依次拼接；`app/rag.py:473` 在遇到第一个装不下的片段时直接停止。本轮 k=2 时五次全部装下，预算与原始拒答无关。k=3 时组装前已有 12,878–13,255 字符，k=4 时为 17,263–18,138 字符。即使取消预算，这两篇新增的第 3、4 条也仍是参考文献。

## 同配额下，更换篇内检索词的实测

保留原始点名得到的六个文件、两路各 15 条、RRF、每篇 2 条和 12,000 字符预算。唯一变化是两路都使用不含作者和年份的英文维度问句。下列五条是本地实际跑过的查询，每次都完整保留 12/12 条。括号内为两篇各自的 RRF 第 1、2 名所在页。

| 维度 | 实测篇内查询 | Ge 页码及证据 | Seidel 页码及证据 |
|---|---|---|---|
| 数据集 | What was the total study population? Total number of participants or healthy individuals included in the study, number of datasets, age range and sex distribution. | （6, 1）总样本 37,407 人、87 个数据集及纳入条件 | （14, 1）ABCD 队列、6,000 多名青少年；首条讲人群外推 |
| 建模变量 | What target measure did the models predict from the input features? Which predictors and covariates were included during model training and prediction? | （6, 2）全脑测量等候选协变量、按性别分层与站点处理 | （5, 6）BLR 公式、基线厚度与两次年龄等输入 |
| 模型 | Which statistical algorithms and regression models were trained and compared for normative modelling? | （2, 8）八种算法的比较设计；第二条是附加临床预测分析 | （5, 5）分组和检验设计、两类常模的 BLR 方法 |
| 结果 | What were the main findings and results for the predictive performance and accuracy of the models? | （7, 6）优化模型性能比较、选择 MFPR 的原因 | （13, 6）C-Norm 与 B-Norm 的拟合差异和比较指标 |
| 局限性 | What limitations did the authors report about the study sample, generalisability, and longitudinal follow-up? | （10, 10）长期随访、年龄覆盖、祖源与协变量限制 | （14, 14）ABCD 外推、BMI/SES、家庭相关性与样本量限制 |

**查询措辞的边界。** 英文措辞本身也会改变前两条。用包含 `training and testing, longitudinal follow-up` 的数据集查询，Seidel 得到 p.5、1（含 5,374 / 805），Ge 得到 p.2、7 的切分和年龄分组，缺总样本数。改问参与者、年龄和随访访视，Ge 又转到 p.3 的两个独立验证队列。上表直接问总研究人群时，Ge 命中 p.6；同一问法下 Seidel p.1 只提供粗略总数。简单问 `outcome measures, explanatory variables, predictors and covariates`，Seidel 命中 p.5 的性能评价和 ANOVA 变量。上表改问预测目标和训练输入才命中公式。进入正文和完整回答一个单元格应分别核验。

## 具体修复建议（待确认）

首选改动点是 [app/rag.py 的多篇分支](../app/rag.py)，以提交 `1b4876b` 的第 617–620 行为准。原查询仍用来取得 `named`。在 `if len(named) >= 2` 分支内，对能明确识别为单一比较维度的查询，另构造只描述内容的 `content_query`，然后把当前第 620 行调用替换为下列形式。

```python
fused = _retrieve_per_paper(
    collection, content_query, _query_tokens(content_query), named, top_k
)
```

五个维度可先用上表实测的英文模板，按原问中的「数据集/样本/人群」「变量/协变量/结局」「统计模型/构建方法」「主要结果/发现/估计」「局限/偏倚/外部效度」等语义选择。在识别时只处理明确的单维通用比较，保留原问中有意义的额外限定；有多个意图或无法可靠识别时，走现有 `vector_query, tokens`。文件名和年份继续承担文件过滤职责。该位置同时覆盖向量与 BM25，无需增加 LLM 或工具调用、改执行器、调大配额或调整 8 步预算。

验收时重放本报告五个原始工具参数，逐条核验前两名是否含本研究的事实，以及预算后两个文件是否都有证据。特别核查 Ge 的主样本与独立验证队列、Seidel 的训练/测试人数、预测协变量与模型评价指标。若要确认最终五篇对比表的完整性，需在实施后另跑整表并核验单元格引用。本次已验证到检索和上下文组装层。
