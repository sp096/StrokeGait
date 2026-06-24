<div align="center">

# 🚶‍♂️ StrokeGait: 基于深度学习的脑卒中视频步态识别系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**面向脑卒中患者病情严重程度的自动化三分类步态评估框架**

[核心功能](#-核心功能) • [系统架构](#-系统架构) • [性能表现](#-性能表现) • [快速开始](#-快速开始) • [项目文档](#-项目文档)

</div>

---

## 📖 项目简介

步态识别作为一种远距离、非接触式的生物特征识别技术，在医疗康复方面有着重要的应用价值。本项目（**StrokeGait**）以真实脑卒中患者的步态视频为研究对象，基于主流的 **GaitSet** 模型进行扩展，旨在实现对患者病情严重程度（Brunnstrom分级 3、4、5类）的自动化分类评估。

为了解决医疗小样本场景下的**数据类别不平衡**问题，本项目创新性地提出了一种**“损失层、采样层、特征层”三层协同的不平衡处理机制**，显著提升了少数类别的识别率以及模型的跨主体（LOSO）泛化能力。

---

## ✨ 核心功能与创新点

- 🎯 **GaitSet 分类化改造**：将原有的检索式输出改造为三分类输出，引入分类头 (SeparateBNNecks) 和固定标签映射机制。
- ⚖️ **三层协同不平衡处理**：
  - **损失层 (Loss Layer)**：代价敏感学习（加权交叉熵）。
  - **采样层 (Sampling Layer)**：边界清洗（Tomek Links）。
  - **特征层 (Feature Layer)**：特征合成扩展（SMOTE）。
- 🔄 **全自动化 LOSO 评估**：支持“留一被试法”（Leave-One-Subject-Out）的自动化训练、测试脚本与配置自动回滚。
- 📊 **可观测训练链路**：引入 `softmax_smote_enabled` 等显式日志指标，实时监控重采样等策略的生效状态。

---

## 🏗️ 系统架构

### 1. 视频数据处理流程

建立从原始视频到剪影序列的标准化预处理流程。利用背景减除和目标分割提取稳定的人体轮廓特征，经过质量把控和对齐后打包成 PKL 格式。

<div align="center">
  <img src="image/图3-1%20视频数据处理流程.png" alt="视频数据处理流程" width="80%">
</div>

### 2. 步态分类模型结构

在 GaitSet 集合特征（Set Pooling）与多尺度特征聚合（HPP）的基础上，接入 SeparateBNNecks 分类头，并在特征层、损失层融入了针对类别不平衡的协同补偿链路。

<div align="center">
  <img src="image/图3-2%20GaitSet扩展为分类模型结构图.png" alt="GaitSet分类模型" width="80%">
</div>

---

## 📈 性能表现 (实验结果)

在复旦大学附属华山医院采集的 30 名真实脑卒中患者步态数据集上，采用严格的**留一被试法（LOSO）**进行验证，实验证明了三层协同策略在高度类别失衡情况下的有效性。

<div align="center">
  <img src="image/图4-1%20类别分布统计图.png" alt="类别分布" width="40%">
  <p><i>数据集类别分布情况 (存在明显的类别失衡)</i></p>
</div>

### 表1：三种单不平衡方法与三层协同方法实验结果对比

| 方案 | 准确率 (Accuracy) | 平衡准确率 (Balanced Acc) | 宏平均 F1 (Macro-F1) |
| :--- | :---: | :---: | :---: |
| **基线模型** | 60.72% | 45.63% | 52.10% |
| **仅损失层** | 54.76% | 46.80% | 50.47% |
| **仅采样层** | 69.43% | 60.71% | 64.78% |
| **仅特征层** | 66.67% | 58.24% | 62.17% |
| **三层协同 (Ours)** | **86.96%** | **84.90%** | **85.92%** |

### 表2：基线模型与三层协同方法在各分类上的详细指标对比

| 类别 (病情严重程度) | 方法 | 召回率 (Recall) | 准确率 (Precision) | F1-Score |
| :---: | :--- | :---: | :---: | :---: |
| **类别 3** (少数类) | 基线模型 | 50.33% | 60.77% | 55.06% |
| | **三层协同** | **87.78%** | **85.85%** | **86.80%** |
| **类别 4** (多数类) | 基线模型 | 95.61% | 90.35% | 92.91% |
| | **三层协同** | 88.36% | 88.89% | 88.62% |
| **类别 5** (少数类) | 基线模型 | 69.56% | 70.54% | 70.05% |
| | **三层协同** | **78.57%** | **80.71%** | **79.63%** |

> **结论**：三层协同策略有效解决了模型因数据不均衡导致的偏置问题，在少数类（类别3、类别5）的召回率和 F1 得分上有显著提升，同时保持了多数类（类别4）的稳定，实现了“少数类提高—总体稳定”的最佳平衡。

---

## 🚀 快速开始

### 环境依赖

- **Python**: 3.8+
- **框架**: PyTorch (配合 `torch.distributed`), torchvision
- **其他包**: `opencv-python`, `numpy`, `PyYAML`, `tqdm`, `scikit-learn`, `imbalanced-learn` (用于 SMOTE 采样)

### 1. 数据预处理

如原始数据未转为 PKL 格式，执行以下命令提取标准化步态特征：
```bash
python datasets/pretreatment.py -i "D:\Downloads\Anonymous18" -o "D:\Downloads\Anonymous-pkl" -n 4 --preserve_structure
```

### 2. 模型训练 (单卡/单进程示例)

训练配置文件：`configs/gaitset/anonymous_test.yaml`
```bash
python -m torch.distributed.launch --nproc_per_node=1 opengait/main.py --cfgs ./configs/gaitset/anonymous_test.yaml --phase train
```
> **注**：LOSO 批量训练可直接运行 `python scripts/leave_one_out_train.py`

### 3. 模型测试

确保配置文件中的 `evaluator_cfg.restore_hint` 已指向训练好的 checkpoint：
```bash
python -m torch.distributed.launch --nproc_per_node=1 opengait/main.py --cfgs ./configs/gaitset/anonymous_test.yaml --phase test
```
> **注**：LOSO 批量测试可运行 `python scripts/leave_one_out_test.py`

---

## 📂 项目结构

```text
StrokeGait/
├── configs/            # 配置文件目录 (包含模型、损失、采样策略等)
├── datasets/           # 数据处理与划分脚本 (如 pretreatment.py)
├── image/              # README 配图
├── opengait/           # 核心代码库
│   ├── data/           # 数据集加载与采样 (Sampler)
│   ├── evaluation/     # 评估函数 (分类准确率计算)
│   ├── modeling/       # 模型架构、Backbone、Loss 定义
│   └── main.py         # 训练/测试统一入口
├── scripts/            # 自动化脚本 (LOSO 批量训练/测试)
├── README.md           # 项目说明文档 (本文档)
└── 相关文档...           # 系统开发文档、代码使用文档等
```

---

## 📚 项目文档参考

如果你需要进一步了解系统的实现细节，请参阅：
- [代码使用说明文档](代码使用说明文档.md) - 详尽的训练、测试及参数配置指南。
- [系统开发文档](系统开发文档.md) - 框架结构与代码设计说明。
- [核心功能与创新点](core_features_innovations.md) - 深入了解三层不平衡处理协同机制。

---

<div align="center">
  <small>该项目为<b>李顺鹏</b>本科毕业论文《基于深度学习的视频步态识别》配套代码库</small>
</div>
