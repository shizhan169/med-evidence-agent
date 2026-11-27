# 语料分集：30 篇 → 开发集 25 篇 + 冻结测试集 5 篇

分集日期：2026-11-27。依据：框架 v2（30 篇 = 25 开发 + 5 冻结测试）与实施计划 2.4。

## 分集原则（面试能讲出来的口径）

1. **已有 28 道题覆盖的 7 篇必须留在开发集**。现有题库只覆盖这 7 篇：Bethlehem 2022、Di Biase 2023、Fujita 2023、Ge 2024、Seidel 2026、Hedman 2012、Coupé 2017。它们已经被 10 轮评测调过，放到测试集就是泄漏——报的就不是“没见过的题”的成绩。
2. **测试集 5 篇全部选自“还没有任何评测题”的论文**。这样测试测的是泛化，不是背题。
3. **测试集按主题分层，不随机抽**。5 篇各代表一个方向：AD 纵向进展、婴幼儿早期发育、站点差异 / harmonization、精神疾病异质性、可泛化的神经退行检测。开发集 25 篇里每个方向都还有其他论文兜底，不会因为抽走一篇就缺一个方向。
4. **4 篇新论文全进测试集**。它们是真正没见过的（没写过题、没调过参），放测试集最能证明系统不是只会对着旧库调出来的。代价是开发集没有新论文可学——可以接受，开发集已有 26 篇，够调了。

## 冻结测试集（5 篇）

| 文件名 | 方向 | 为什么选它 |
|---|---|---|
| 2024-verdi-personalizing-progressive-changes-alzheimers.pdf | AD 纵向进展 | 新论文；同 Subramaniapillai 一脉的 tOC 方法，AD + MCI 转归，海马变化最快 |
| 2021-dimitrova-phenotyping-preterm-brain.pdf | 婴幼儿早期发育 | 新论文；早产儿个体偏离常模，dHCP 两个队列，补上早期发育这个年龄端 |
| 2022-bayer-site-variation-normative-models.pdf | 站点差异 / harmonization | 新论文；分层贝叶斯 vs ComBat，方法类唯一进测试集的代表 |
| 2018-wolfers-heterogeneous-phenotype-schizophrenia-bipolar.pdf | 精神疾病异质性 | 新论文；常模临床应用的经典实证，个体重叠率仅约 2% |
| 2025-parker-generalizable-mri-normative-modelling.pdf | 可泛化的神经退行检测 | 旧库中无题的论文；56,173 人常模，NACC / ADNI / FTD，覆盖 AD 与额颞叶变性 |

## 开发集（25 篇）

其余全部 25 篇，见 `dev.txt`。包括上述 7 篇有题论文，以及 Fraza 2025（帕金森）、Poos 2022（FTD）、Brouwer 2022（遗传）、Ho 2026（多队列 harmonisation）、Cirstian 2025（自闭症白质）等，方向不缺。

## 冻结规则

- 测试集的 10 道题（每篇 2 道）已于 2026-11-27 冻结，文件 `eval/datasets/test_questions_frozen.md`（T01–T10，用户已对照本地 PDF 逐题核对无误）。开发过程中不跑、不看、不调，只在阶段验收时跑一次，跑完报成绩，不根据结果再改模型 / prompt / 阈值；要改就改完再重新冻结一版（注明第二版）。
- R03（APOE 相关应拒答）仍然成立：新 1 号换成 Verdi 2024 后，库里仍然没有 APOE / 基因相关的论文，APOE 问题继续拒答，不用改题。

## 现状核对（2026-11-27，如实记录）

- 用户本地已下载 4 篇新论文；本机（VM）`data/papers/` 里目前只有 26 个 PDF，未同步新下的 4 篇。
- 入库前先把 4 篇按上表文件名放进本地 `data/papers/`，再删 `data/chroma`、重跑 `python -m app.ingest --papers data/papers`，核对片段数在 4456 基础上增加。
- PAPERS.md 已同步更新：1 号、4 号替换为 Verdi 2024、Dimitrova 2021，2 号 Bayer 2022、3 号 Wolfers 2018 状态改为已下载；Macleod 2025、Subramaniapillai 2026 不再计入 30 篇。
