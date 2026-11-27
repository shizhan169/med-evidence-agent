# 文献库清单（30 篇，纵向常模方向）

更新时间：2026-11-27。文件名格式：年份-作者-标题（2026-11-25 统一改名）。状态说明：

- **已入库**：26 篇已在库中（5 篇 MVP 种子 + 12 篇扩库下载 + 9 篇用户补齐）。
- **已下载（用户本地，待同步入库）**：4 篇（2026-11-27 用户下载：Verdi 2024、Dimitrova 2021、Bayer 2022、Wolfers 2018）。本机 `data/papers/` 目前仍只有 26 个 PDF，入库前需先把这 4 篇放进来。
- **待补**：0 篇。原待补的 Subramaniapillai 2026（medRxiv 被 Cloudflare 封 IP）和 Macleod 2025（仅会议摘要、无全文）已按用户要求替换，不再计入 30 篇。

## 分集（2026-11-27，见 eval/datasets/SPLIT.md）

- **开发集 25 篇**：除下表标注“测试”的 5 篇外全部。包括已有 28 道题覆盖的 7 篇（Bethlehem、Di Biase、Fujita、Ge、Seidel、Hedman、Coupé），它们必须留在开发集，否则测试泄漏。
- **冻结测试集 5 篇**：Verdi 2024、Dimitrova 2021、Bayer 2022（站点差异）、Wolfers 2018、Parker 2025。全部是目前没有任何评测题的论文，4 篇是新下的。测试题（每篇 2 道，共 10 道）写完封存，只跑一次验收。

| # | 文件名 | 文献 | 状态 | 备注 |
|---|---|---|---|---|
| 1 | 2022-bethlehem-brain-charts.pdf | Bethlehem et al. 2022, Brain charts for the human lifespan, Nature | 已入库 |  |
| 2 | 2023-dibiase-mapping-brain-charts.pdf | Di Biase et al. 2023, Mapping human brain charts cross-sectionally and longitudinally, PNAS | 已入库 |  |
| 3 | 2023-fujita-brain-volume-aging.pdf | Fujita et al. 2023, Characterization of Brain Volume Changes in Aging Individuals With Normal Cognition Using Serial Magnetic Resonance Imaging, JAMA Netw Open | 已入库 |  |
| 4 | 2024-ge-centilebrain.pdf | Ge et al. 2024, Normative modelling of brain morphometry across the lifespan with CentileBrain: algorithm benchmarking and model optimisation | 已入库 |  |
| 5 | 2026-seidel-baseline-cortical-thickness.pdf | Seidel et al. 2026, Predicting developmental norms from baseline cortical thickness in longitudinal studies, Biol Sex Differ | 已入库 |  |
| 6 | 2026-bayer-charting-velocity-brain-growth.pdf | Bayer et al. 2026, Charting the velocity of brain growth and development, arXiv:2601.07591 | 已入库 | arXiv 预印本，64 页 |
| 7 | 2025-kafadar-conditional-longitudinal-brain-growth.pdf | Kafadar et al. 2025, Conditional-longitudinal brain growth charts detect MRI changes with birth weight and psychopathology, bioRxiv | 已入库 | bioRxiv 预印本，21 页 |
| 8 | 2021-dimitrova-phenotyping-preterm-brain.pdf | Dimitrova et al. 2021, Phenotyping the Preterm Brain: Characterizing Individual Deviations From Normative Volumetric Development in Two Large Infant Cohorts, Cereb Cortex | 已下载（用户本地） | 替换 Macleod 2025；测试集；开放获取 CC BY，PMC8258435 |
| 9 | 2025-rehak-buckova-normative-models-longitudinal-changes.pdf | Rehák Bučková et al. 2025, Using normative models pre-trained on cross-sectional data to evaluate intra-individual longitudinal changes in neuroimaging data, eLife | 已入库 | 28 页 |
| 10 | 2025-fraza-neurodegenerative-progression-parkinsons.pdf | Fraza et al. 2025, Understanding individual neurodegenerative progression in Parkinson's disease through normative modelling, Sci Rep | 已入库 | 18 页 |
| 11 | 2024-gaiser-cortical-thickness-trajectories-children.pdf | Gaiser et al. 2024, Estimating cortical thickness trajectories in children across different scanners using transfer learning from normative models, Hum Brain Mapp | 已入库 | 用户补齐 |
| 12 | 2024-verdi-personalizing-progressive-changes-alzheimers.pdf | Verdi et al. 2024, Personalizing progressive changes to brain structure in Alzheimer's disease using normative modeling, Alzheimers Dement | 已下载（用户本地） | 替换 Subramaniapillai 2026；测试集；开放获取，PMC11633367 |
| 13 | 2022-poos-longitudinal-brain-atrophy-ftd.pdf | Poos et al. 2022, Longitudinal Brain Atrophy Rates in Presymptomatic Carriers of Genetic Frontotemporal Dementia, Neurology | 已入库 | 用户补齐 |
| 14 | 2022-brouwer-genetic-variants-longitudinal-brain-structure.pdf | Brouwer et al. 2022, Genetic variants associated with longitudinal changes in brain structure across the lifespan, Nat Neurosci | 已入库 | 用户补齐 |
| 15 | 2022-bayer-site-variation-normative-models.pdf | Bayer et al. 2022, Accommodating site variation in neuroimaging data using normative and hierarchical Bayesian models, NeuroImage | 已下载（用户本地） | 测试集；开放获取，PMC7614761 / ScienceDirect |
| 16 | 2026-ho-unified-multi-cohort-harmonisation.pdf | Ho et al. 2026, Unified Multi-Cohort Harmonisation and Normative Modelling of Neuroimaging Data via Hierarchical GAMLSS, bioRxiv | 已入库 | bioRxiv 预印本，65 页 |
| 17 | 2025-parker-generalizable-mri-normative-modelling.pdf | Parker et al. 2025, Generalizable MRI normative modelling to detect age-inappropriate neurodegeneration, Alzheimers Res Ther | 已入库 | 测试集；21 页 |
| 18 | 2025-cirstian-white-matter-development-autism.pdf | Cirstian et al. 2025, Linking White-Matter Development to Clinical Variation in Autism: Longitudinal Normative Modelling of FA in the EU-AIMS LEAP Cohort, bioRxiv | 已入库 | bioRxiv 预印本，24 页 |
| 19 | 2016-marquand-heterogeneity-normative-models.pdf | Marquand et al. 2016, Understanding Heterogeneity in Clinical Cohorts Using Normative Models: Beyond Case-Control Studies, Biol Psychiatry | 已入库 | 用户补齐 |
| 20 | 2022-rutherford-brain-growth-aging-precision.pdf | Rutherford et al. 2022, Charting brain growth and aging at high spatial precision, eLife | 已入库 | 28 页 |
| 21 | 2022-rutherford-normative-modeling-framework-psychiatry.pdf | Rutherford et al. 2022, The normative modeling framework for computational psychiatry, Nat Protoc | 已入库 | bioRxiv 预印本版（2021.08.08.455583），44 页 |
| 22 | 2018-wolfers-heterogeneous-phenotype-schizophrenia-bipolar.pdf | Wolfers et al. 2018, Mapping the heterogeneous phenotype of schizophrenia and bipolar disorder using normative models, JAMA Psychiatry | 已下载（用户本地） | 测试集；开放获取，PMC6248110 |
| 23 | 2020-pomponio-harmonization-mri-datasets.pdf | Pomponio et al. 2020, Harmonization of large MRI datasets for the analysis of brain imaging patterns throughout the lifespan, NeuroImage | 已入库 | 用户补齐 |
| 24 | 2018-fortin-harmonization-cortical-thickness.pdf | Fortin et al. 2018, Harmonization of cortical thickness measurements across scanners and sites, NeuroImage | 已入库 | bioRxiv 预印本版（10.1101/148502），28 页 |
| 25 | 2023-segal-regional-circuit-network-heterogeneity.pdf | Segal et al. 2023, Regional, circuit and network heterogeneity of brain abnormalities in psychiatric disorders, Nat Neurosci | 已入库 | 35 页 |
| 26 | 2022-kia-closing-lifecycle-normative-modeling.pdf | Kia et al. 2022, Closing the life-cycle of normative modeling using federated hierarchical Bayesian regression, PLoS ONE | 已入库 | 29 页 |
| 27 | 2017-coupe-unified-analysis-brain-maturation-aging.pdf | Coupé et al. 2017, Towards a unified analysis of brain maturation and aging across the entire lifespan: A MRI analysis, Hum Brain Mapp | 已入库 | 用户补齐 |
| 28 | 2021-romero-unified-analysis-cerebellum-maturation-aging.pdf | Romero et al. 2021, Toward a unified analysis of cerebellum maturation and aging across the entire lifespan: A MRI analysis, Hum Brain Mapp | 已入库 | 用户补齐 |
| 29 | 2012-hedman-human-brain-changes-review-56-longitudinal-mri.pdf | Hedman et al. 2012, Human brain changes across the life span: A review of 56 longitudinal magnetic resonance imaging studies, Hum Brain Mapp | 已入库 | 用户补齐 |
| 30 | 2008-shaw-neurodevelopmental-trajectories-cerebral-cortex.pdf | Shaw et al. 2008, Neurodevelopmental Trajectories of the Human Cerebral Cortex, J Neurosci | 已入库 | 用户补齐 |

## 统计

- 已入库：26 篇；已下载（用户本地，待同步入库）：4 篇；合计 30 篇；待补：0 篇。
- 分集：开发集 25 篇，冻结测试集 5 篇（Verdi 2024、Dimitrova 2021、Bayer 2022 站点差异、Wolfers 2018、Parker 2025），详见 eval/datasets/SPLIT.md。
- 2026-11-25 文件名统一改为年份-作者-标题格式；改名后本地需删除 data/chroma 重新入库（旧片段用的旧文件名），评测题库中的 5 篇 MVP 文件名已同步更新。
- 拦截原因汇总（同一篇可能同时遇到两种拦截）：PMC 全文下载被 proof-of-work 机器人验证拦截（11 篇）；Wiley / Elsevier / JAMA / Neurology / J Neurosci 出版社站直接返回 403（9 篇）；medRxiv 返回 403（1 篇）；Macleod 2025 仅有会议摘要、无公开全文（1 篇）。未付费、未登录任何账号。2026-11-27 补充：medRxiv Subramaniapillai PDF 在用户 VPN IP 上遇到 Cloudflare 直接封 IP，已改用 Verdi 2024 替换。

## 入库后

重新跑 `python -m app.ingest --papers data/papers` 即可入库（改名后第一次需先删 data/chroma 再跑）。注意：Subramaniapillai 2026 已替换为 Verdi 2024，库里仍然没有 APOE / 基因相关的论文，R03（APOE 相关应拒答）继续成立，不用改题。
