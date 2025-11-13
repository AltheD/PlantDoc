---
title: PlantDoc 作业执行指南
author: 课程小组
date: 2025-11-13
---

# PlantDoc 作业执行指南

> 适用对象：首次接触计算机视觉与深度学习的两人小组  
> 作业截止：2025 年 11 月 30 日  
> 参考文献：Mall, S. S. K., & Arora, A. (2020). *PlantDoc: A Dataset for Visual Plant Disease Detection*. In Proceedings of the 11th ACM India Computing Convention (pp. 249–258). https://doi.org/10.1145/3371158.3371196

---

## 1. 项目背景与目标

- **核心任务**：基于 PlantDoc 数据集完成植物病害图像分类任务，覆盖数据准备、特征工程、模型训练、性能分析与报告撰写的端到端流程。  
- **数据集特点（来自参考论文）**：  
  - 共 2,598 张田间实拍图像，包含 13 种作物、17 类病害，共 27 个「作物-病害/健康」类别。  
  - 数据采集自网络真实场景，光照、背景和叶片姿态存在较大变化，贴合实际部署环境。  
  - 标注以整张叶片为单位，适合分类与检测任务。论文中给出了基于 Faster R-CNN 与 SSD 的识别基线，展示了数据集的挑战性。  
- **考核重点**：实验报告占 85%，模型精度占 15%。需要对传统方法与深度学习方法进行系统对比、消融分析和结果解读。

---

## 2. 角色分工（两人小组）

| 角色 | 负责人 | 主要职责 |
| --- | --- | --- |
| 传统机器学习负责人（成员 B） | 组员 2 | 数据预处理支持、特征提取（SIFT/HOG/SURF 等）、分类器训练（SVM/LogReg/MLP/KNN）、消融实验（特征维度、分类器选择、正则化）、可视化（特征示例、决策边界）、报告撰写对应章节 |
| 深度学习负责人（成员 A，当前你） | 你 | 数据准备主导、数据增强策略、深度学习模型（CNN、预训练模型、Vision Transformer）训练与调优、LLM 相关探索（提示式分类或视觉语言模型）、消融实验（学习率/优化器/数据增强/模型深度）、可解释性分析（Grad-CAM 等）、报告撰写对应章节 |
| 公共协作 | 双方 | 数据集维护、日志同步、实验记录、结果汇总表、报告整体排版、提交物整理与审核 |

---

## 3. 时间规划（建议安排）

| 时间段 | 关键里程碑 | 负责人 |
| --- | --- | --- |
| 第 1 天 | 环境搭建、数据集下载与检查、Git/云盘项目结构初始化 | 双方 |
| 第 2 天 | 数据探索（EDA）、类别分布与样例可视化、任务定义与实验清单确认 | 双方 |
| 第 3-4 天 | 传统方法建立基线（HOG+SVM），记录首轮结果 | 成员 B |
| 第 3-5 天 | 深度学习基线（预训练 ResNet50 迁移学习），记录首轮结果 | 成员 A |
| 第 5-6 天 | 扩展实验：数据增强对比、不同骨干网络、超参数搜索 | 成员 A 主导，成员 B 协助记录 |
| 第 6 天 | 传统方法扩展（SURF+SVM、HOG+LogReg、PCA 消融等） | 成员 B |
| 第 7 天 | Vision Transformer、轻量 Transformer（DeiT/TinyViT）试验 | 成员 A |
| 第 8 天 | LLM/视觉语言探索（如 CLIP/BLIP 特征 + 线性分类器、LLM 提示分类） | 成员 A |
| 第 9 天 | 失败样例分析、Grad-CAM 可视化、混淆矩阵绘制 | 双方 |
| 第 10 天 | 报告撰写初稿（结构 + 图表 + 结果汇总表） | 双方 |
| 第 11-12 天 | 报告润色、引用检查、提交物整理（预测文件、代码仓库、README） | 双方 |
| 截止前 1 天 | 最终检查，发送邮件提交 | 双方 |

> 若时间紧张，可合并或压缩部分实验，但务必保留至少：传统基线、深度学习基线、两组以上消融实验、失败样例分析、混淆矩阵。

---

## 4. 项目目录与版本管理建议

```
PlantDoc/
├── data/
│   ├── raw/               # 原始数据（保持只读）
│   ├── processed/         # 统一尺寸、划分后的数据
│   └── splits/            # 训练/验证/测试划分索引
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 10_traditional_baseline.ipynb
│   ├── 20_dl_resnet50.ipynb
│   ├── 21_dl_augment_ablation.ipynb
│   ├── 22_dl_vit.ipynb
│   └── 30_failure_analysis.ipynb
├── src/
│   ├── data/              # 数据处理脚本
│   ├── models/            # 模型定义
│   ├── training/          # 训练循环与配置
│   └── eval/              # 评估与可视化
├── outputs/
│   ├── logs/
│   ├── checkpoints/
│   ├── predictions/
│   └── figures/
├── report/
│   ├── draft.docx (或 LaTeX)
│   └── tables_figures/
├── requirements.txt
├── README.md
└── Guide.md (本文件)
```

- 使用 Git 或云盘版本控制，确保关键阶段打上 tag。  
- 每次实验保留配置（YAML/JSON）、日志与指标（CSV/JSON），便于追踪与复现实验。

---

## 5. 环境与工具准备

### 5.1 共同准备
- **Python 版本**：建议 3.10/3.11。确保安装 `virtualenv` 或使用 Conda。  
- **核心库**：`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `opencv-python`, `albumentations`, `seaborn`.  
- **深度学习栈（成员 A）**：`torch`, `torchvision`, `timm`, `transformers`, `pytorch-lightning` 或 `lightning`, `grad-cam`.  
- **传统方法栈（成员 B）**：`scikit-image`, `opencv-contrib-python`, `xgboost`（可选）。  
- **可选工具**：`wandb`/`tensorboard`（实验跟踪）、`hydra`（配置管理）、`rich`（日志输出）。

### 5.2 数据集下载
1. 参考课程群共享或 PlantDoc 官方仓库。常用来源：  
   - 官方 GitHub（含图像与标注）：https://github.com/AI-Lab-Research/PlantDoc-Dataset  
2. 下载后校验：  
   - 确认类别目录齐全（27 个子目录）。  
   - 随机查看若干图像，确认数据无损坏。  
3. 整理到 `data/raw/PlantDoc`，使用脚本统一命名和划分数据集（推荐留出 15% 作为测试集，不参训）。

### 5.3 计算资源
- 若本地 GPU 不足，优先使用 Google Colab Pro / Kaggle Notebook。  
- 建议在 Colab 中维护 `notebooks/`，定期下载并同步到本地仓库。  
- 大模型或 Transformer 实验可先在较小分辨率 (e.g. 224×224) 与较小 batch size 上调试。

---

## 6. 数据探索与预处理流程

1. **EDA**（共享 notebook `00_eda.ipynb`）：  
   - 统计每类样本数量，绘制条形图，找出不平衡类别。  
   - 可视化随机样本，观察光照、背景、病斑形态。  
   - 若存在严重不平衡，记录在报告中并在训练阶段使用加权损失或过采样策略。
2. **数据划分**：  
   - 建议 `train:val:test = 70:15:15` 按叶片图像级别随机划分。  
   - 保存划分索引到 `data/splits/plantdoc_split_seed42.json` 以保证双方实验一致。  
3. **图像预处理**：  
   - 统一尺寸（推荐 256×256，再中心裁剪/缩放到 224×224）。  
   - 归一化（`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`）。  
4. **数据增强策略**（深度学习主线）：  
   - 基础：随机水平翻转、随机旋转 (±20°)、随机裁剪与缩放。  
   - 进阶：ColorJitter、Cutout/Mixup/CutMix（用于消融）。  
5. **传统特征准备**：  
   - 转灰度、尺寸归一（如 256×256）。  
   - 对比：原图 vs. 去噪/直方图均衡后的影响。

---

## 7. 深度学习路线（成员 A）

### 7.1 基线模型
- **ResNet50 迁移学习**  
  - 预训练权重：ImageNet。  
  - 冻结前几层、只训练全连接层 → 记录结果。  
  - 解冻全部层、使用较小学习率微调 → 记录结果。  
  - 优化器：AdamW，学习率 1e-4 起步；Scheduler：CosineAnnealing 或 StepLR。  
  - 批次大小：16（视显存调整）。  
  - 训练轮数：10~20（早停监控验证集 F1）。  
  - 输出：训练曲线、最优模型权重、验证/测试集指标。

### 7.2 扩展实验清单
1. **模型骨干对比**  
   - EfficientNet-B0, ResNet18（轻量基线）  
   - ConvNeXt-Tiny 或 MobileNetV3（轻量化）。  
   - 记录参数量、训练时间、精度差异。
2. **Vision Transformer (ViT/DeiT)**  
   - 使用 `timm` 加载 `vit_base_patch16_224` 或 `deit_small_patch16_224`.  
   - 尝试混合精度训练（AMP）降低资源压力。  
   - 对比 Transformer 与 CNN 的收敛速度与性能。  
3. **数据增强消融**  
   - 无增强 vs. 基础增强 vs. 强增强 (Mixup/CutMix)。  
   - 记录验证集准确率与过拟合情况。  
4. **超参数敏感性**  
   - 学习率：`1e-3`, `5e-4`, `1e-4`。  
   - 优化器：SGD(momentum=0.9) vs. AdamW。  
   - 批次大小：8 vs. 16。  
5. **LLM/视觉语言探索**  
   - CLIP 特征提取 + 线性分类器：  
     - 使用 OpenAI CLIP ViT-B/32 提取图像特征。  
     - 将类别名称/描述通过文本编码，比较零样本 vs. 线性微调。  
   - BLIP/Florence 等模型生成描述，再由 LLM 做推断（可选，报告中说明尝试与局限）。  
   - 若资源有限，可只做 CLIP 零样本分类并分析差异。

### 7.3 可解释性与可视化
- **Grad-CAM**：对最佳 CNN 模型生成可视化热力图，展示模型关注区域。  
- **Transformer Attention Map**：展示 ViT 的注意力分布（可使用 `timm` 或 `einops` 辅助）。  
- **错误案例分析**：  
  - 从混淆矩阵中选出误判率最高的 2-3 对类别。  
  - 展示原图、预测结果、Grad-CAM，讨论错误原因（如病斑相似、背景干扰、叶片姿态）。

### 7.4 日志与记录
- 每次实验记录：配置、训练时间、最优指标、备注。建议使用 CSV 或 Notion 表格，列如：`exp_id`, `model`, `aug`, `lr`, `optimizer`, `val_acc`, `test_acc`, `notes`.  
- 将关键实验导出的曲线与混淆矩阵保存至 `outputs/figures/`，命名如 `exp12_resnet50_lr1e-4_confusion.png`。

---

## 8. 传统方法路线（成员 B）

### 8.1 特征提取
- **HOG (Histogram of Oriented Gradients)**：使用 `skimage.feature.hog`，调整 cell size 与 orientations，比较不同参数组合。  
- **SIFT / SURF**（需要 `opencv-contrib-python`）：  
  - 提取关键点与描述子。  
  - 使用 Bag-of-Visual-Words（BOVW）或 VLAD 聚合成固定长度特征。  
- **颜色直方图 + 纹理特征（LBP/Gabor）**：作为补充对比。

### 8.2 分类器
- Logistic Regression（带 L2 正则）  
- SVM（RBF 与 Linear 核）  
- 随机森林 / XGBoost（可选）  
- MLP（1-2 层全连接网络作为传统方法延伸）

### 8.3 消融实验
- **特征组合**：HOG 单独 vs. HOG+颜色 vs. HOG+BOVW。  
- **降维策略**：PCA 保留 95% 方差 vs. 不降维。  
- **分类器对比**：同一特征下不同分类器的准确率、训练时间、预测速度。  
- **样本不平衡处理**：类别权重、SMOTE 过采样等方法影响。

### 8.4 输出产物
- 指标表（准确率/精确率/召回率/F1）。  
- 传统方法混淆矩阵与错误示例。  
- 特征可视化（如 HOG 可视图、BOVW 聚类图）。

---

## 9. 消融实验与结果整合

### 9.1 建议至少完成以下实验组合
| 实验编号 | 描述 | 负责人 | 指标重点 |
| --- | --- | --- | --- |
| E1 | HOG + SVM（Baseline） | 成员 B | 准确率、F1、混淆矩阵 |
| E2 | ResNet50 迁移学习（无增强） | 成员 A | 准确率、Loss 曲线 |
| E3 | ResNet50 迁移学习（含增强） | 成员 A | 泛化对比 |
| E4 | ViT/DeiT 模型 | 成员 A | 与 CNN 对比 |
| E5 | CLIP 零样本 vs. 线性微调 | 成员 A | 创新性讨论 |
| E6 | 数据增强消融 | 成员 A | 泛化能力 |
| E7 | 传统特征组合消融 | 成员 B | 特征工程影响 |

### 9.2 结果汇总建议
- 制作统一表格（放入报告）：  
  - 行：各实验编号  
  - 列：Top-1 准确率、Macro-F1、训练时长、推理速度、备注  
- 绘制图表：  
  - 学习率敏感性曲线  
  - 不同模型准确率柱状图  
  - 数据增强对性能影响的折线/柱状图  
  - 混淆矩阵（最佳模型）  
- 失败样例：至少 6 对（真实标签 vs. 预测标签 + 分析文字）。

---

## 10. 实验报告撰写框架（5-10 页）

1. **摘要（可选）**：简述任务、方法、主要结果。  
2. **引言**：背景、PlantDoc 数据集意义、参考论文贡献（引用原文）。  
3. **相关工作**：概述传统与深度学习植物病害识别研究（可列 2-3 篇参考文献）。  
4. **数据集与预处理**：数据源、类别统计、划分策略、增强方法。  
5. **方法**：  
   - 传统方法：特征提取、分类器与参数。  
   - 深度学习：模型结构、训练策略、超参数。  
   - LLM/视觉语言探索：方法与限制。  
6. **实验设置**：硬件、软件版本、训练细节、评价指标。  
7. **实验结果与分析**：  
   - 主结果表格  
   - 子章节讨论消融实验、模型对比  
   - 混淆矩阵、失败样例、Grad-CAM 等可视化  
8. **讨论**：优势、局限、未来改进方向（例如数据扩充、半监督、部署考虑）。  
9. **结论**：总结主要发现与贡献。  
10. **参考文献**：遵循课程要求（例如 ACM/IEEE 引用格式），务必包含 PlantDoc 原始论文。

> **提示**：报告中所有图表须有编号与标题；正文中引用图表（如“见图 2”），并说明观察到的现象与原因。

---

## 11. 提交物清单

- [ ] 实验报告（PDF，5-10 页）  
- [ ] 最佳模型在测试集上的预测结果文件（CSV/JSON，注明类别标签映射）  
- [ ] 源代码仓库链接（含 README、运行说明、依赖列表）  
- [ ] 关键实验日志、曲线与可视化图片（可附在报告或提供补充材料）  
- [ ] 邮件发送至 `guangyu.ryan@yahoo.com`，主题格式建议：“PlantDoc 作业提交 - 第 X 小组 - 成员姓名”  
- [ ] 邮件正文列出附件/链接、运行环境说明及模型精度摘要

---

## 12. 风险与备选方案

- **算力不足**：  
  - 先在较小输入尺寸 (128×128) 上验证流程；  
  - 使用轻量模型（MobileNetV3、EfficientNet-Lite）；  
  - 可在 Colab 分阶段训练并保存中间模型到 Google Drive。  
- **数据不平衡**：  
  - 在训练中启用类别权重或 Focal Loss；  
  - 对少数类使用图像增强增强数量。  
- **时间不够**：  
  - 确保至少完成 Baseline + 核心消融；  
  - 报告中说明未完成部分及原因，展示过程思考。  
- **LLM 实验无法完成**：  
  - 至少完成 CLIP 零样本测试并在报告中讨论其效果与局限；  
  - 若完全无法进行，需在报告中说明原因并提出未来计划。

---

## 13. 参考资源

- PlantDoc 原始仓库与数据集：https://github.com/AI-Lab-Research/PlantDoc-Dataset  
- torchvision 迁移学习教程：https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html  
- timm 模型库文档：https://rwightman.github.io/pytorch-image-models/  
- Albumentations 数据增强文档：https://albumentations.ai/docs/  
- Grad-CAM 实现参考：https://github.com/jacobgil/pytorch-grad-cam  
- CLIP 使用示例：https://github.com/openai/CLIP  
- 传统图像特征与分类器综述（可作为参考文献）：  
  - Barbedo, J. G. A. (2013). Digital image processing techniques for detecting, quantifying and classifying plant diseases. *SpringerPlus*, 2(660).  
  - Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311–318.

---

## 14. 执行建议速查表

- 每日保持 10 分钟站会式同步，更新实验记录。  
- 任何实验需先写下“目的 → 配置 → 预期 → 实际结果 → 结论”。  
- 结果不理想时，优先检查：数据加载、学习率、归一化、标签映射。  
- 及时保存最优模型与指标；不要等到最后一天再整理。  
- 报告草稿至少提前 3 天完成，预留时间校对与完善图表。  
- 提交前进行 Checklist 对照，确保所有附件齐全。

---

祝顺利完成作业！若遇到阻塞问题，优先记录在共享文档，并在每日同步时讨论解决方案。保持良好协作与实验记录，将大幅提升最终报告质量与得分。


