# OpenGait-脑卒中步态三分类：核心功能与创新点

## 项目定位

本项目以 `configs/gaitset/anonymous_test.yaml` 为统一实验入口，面向**脑卒中患者步态数据集**，按照**病情严重程度进行三分类**任务，构建了一个可复现的 LOSO（Leave-One-Subject-Out）训练与评估流程。核心目标是：

- 在跨患者（LOSO）场景下获得稳定分类性能。
- 在轻度类别不平衡下提升不同严重程度类别的识别能力。
- 保证从配置到训练、测试、日志分析的工程闭环。

---

## 一、核心功能

### 1) 单配置驱动的分类训练/测试

- 核心配置文件：`configs/gaitset/anonymous_test.yaml`
- 训练入口：`opengait/main.py --phase train`
- 测试入口：`opengait/main.py --phase test`
- 评估函数：`evaluate_scoliosis`（`opengait/evaluation/evaluator.py`）

该功能把模型结构、标签映射、损失函数、采样器、feature_smote、增强策略统一到一个 YAML 文件中，便于持续迭代和对照实验。

### 2) GaitSet 分类模式改造

在 `opengait/modeling/models/gaitset.py` 中，GaitSet 从检索式输出扩展为分类式输出：

- 保留 `Triplet` 分支（可配置权重）。
- 新增 `SeparateBNNecks` 分类头输出 `logits`。
- 支持固定 `label_mapping`，将“轻度/中度/重度”或对应的三种病情等级稳定映射为 3 个类别，避免跨 fold 标签索引漂移。

这使得原步态识别骨干可直接用于有监督分类任务。

### 3) 三层不平衡处理机制

项目不是只依赖一种方法，而是采用三层协同：

- **损失层**：`CrossEntropyLoss(weight=...)` 类权重（代价敏感学习）。
- **采样层**：`trainer_cfg.sampler.type: ImbalancedSampler` + `sampling_method`。
- **特征层**：`feature_smote` 对 embedding 执行 `smote/tomek/smote_tomek/random_over/interpolate`。

对应实现：`opengait/modeling/losses/ce.py` 与 `opengait/utils/feature_smote.py`。

### 4) LOSO 自动化训练与自动化测试

- 训练脚本：`scripts/leave_one_out_train.py`
  - 每次从 `TRAIN_SET` 留出 1 个被试。
  - 自动更新 `trainer_cfg.save_name`。
  - 运行结束自动回滚 JSON 和 YAML。
- 测试脚本：`scripts/leave_one_out_test.py`
  - 每次将 `TEST_SET` 设置为单被试。
  - 自动更新 `evaluator_cfg.restore_hint` 指向对应 fold checkpoint。
  - 同样自动回滚配置。

该机制保证了 LOSO 全流程可重复执行并减少手工配置错误。

### 5) 可观测训练链路

训练日志中持续输出：

- `softmax_base_loss`
- `softmax_loss`
- `softmax_accuracy`
- `softmax_smote_enabled`
- `softmax_smote_synthetic`

这些指标能直接反映 feature_smote 在每个 batch 是否真实触发及其合成强度，也方便观察少数严重程度类别是否被有效补偿。

---

## 二、核心创新点（工程与方法结合）

### 创新点 1：在 GaitSet 分类化改造中引入“可控标签映射”

通过配置级 `label_mapping` 在训练和评估两端统一标签 ID 语义，解决了 LOSO 场景常见的“每折标签重编码不一致”问题，尤其适合脑卒中病情严重程度这种固定三分类标签。

### 创新点 2：将不平衡处理从数据层扩展到特征层

除常规 class weight 和采样器外，项目在 embedding 层引入 `feature_smote`，并支持多模式切换（`smote`、`tomek`、`smote_tomek`、`random_over`、`interpolate`），形成可插拔的特征重采样机制，适合缓解不同病情等级样本不均衡的问题。

### 创新点 3：为分布式训练稳定性增加“防崩溃保护逻辑”

在 `ce.py` 中加入对 feature_smote 辅助分支的安全约束（如重采样后样本过少时跳过该分支），避免 BatchNorm 小 batch 崩溃，同时保留主损失路径，提升训练鲁棒性。

### 创新点 4：LOSO 全自动脚本化与配置自动回滚

训练/测试脚本均实现“动态改配置 -> 执行 -> 回滚”的闭环，减少人工切换折次带来的配置污染，是工程上非常实用的稳定化创新。

### 创新点 5：将“方法生效性”显式日志化

`softmax_smote_enabled` 和 `softmax_smote_synthetic` 的引入，使 feature-level 重采样从“黑盒开关”变为“可观测状态变量”，便于定位为什么某种模式（如 `smote_tomek`）在某些 batch 未生效，也便于观察对少数严重程度类别是否产生了实际增益。

---

## 三、与你当前配置直接对应的实现要点

基于当前 `anonymous_test.yaml`：

- 模型：`GaitSet + SeparateBNNecks(class_num=3)`
- 损失：`TripletLoss(weight=0.0) + CrossEntropyLoss(weighted)`
- 采样：`ImbalancedSampler + TomekLinks`
- 特征重采样：`feature_smote.enabled=true, mode=smote`
- 训练策略：`SGD(lr=0.01), total_iter=6000, fp16`

这是一条“以脑卒中病情严重程度分类性能为主，兼顾不平衡处理”的实验主线。

---

## 四、项目价值

1. 形成了可复用的 LOSO 实验模板（可迁移到其他小样本医学步态分类任务）。
2. 提供了“配置驱动 + 自动回滚 + 日志可解释”的完整工程化范式。
3. 在不平衡问题上实现了从损失、采样到特征层的多级联动，具备进一步发表/汇报的技术叙事基础。

---

## 五、当前局限与后续方向

- `torch.distributed.launch` 已被官方标注弃用，可迁移到 `torchrun`。
- `smote_tomek` 在小 batch 下可能出现净增为 0，需要在 batch 组成和策略参数上继续优化。
- 建议新增 LOSO 汇总脚本，自动统计各折 `Accuracy / Macro-F1 / Recall / 混淆矩阵`，提升报告效率。

---

## 六、建议在汇报中使用的一句话

本项目的核心创新是：**将 GaitSet 分类化、LOSO 自动化与多层不平衡学习（loss/sampler/feature-smote）进行一体化工程落地，并通过显式日志机制实现可解释的训练与评估闭环，从而服务于脑卒中患者步态严重程度三分类任务。**

