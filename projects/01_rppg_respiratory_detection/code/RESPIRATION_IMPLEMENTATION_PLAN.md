# rPPG 远程呼吸检测实验方案

> 基于光流引导多ROI + 时空特征融合 + 半监督学习的远程呼吸率估计实现指南

---

## 一、课题概述

### 1.1 核心创新点

| 创新点 | 描述 | 参考文献 |
|--------|------|----------|
| 多区域ROI融合 | 面部+颈部+胸部 6通道异构ROI | FactorizePhys, JAMSNet |
| 光流引导胸部ROI定位 | 光流识别呼吸运动区域 + 皮肤检测 | RAFT, CVPR2024 |
| STREAM-Net时空模块 | SS-FEM + STLAM + WTSM组合 | STREAM-Net, BigSmall |
| 半监督呼吸率估计 | 30%标注 + SNR课程伪标签 | Semi-rPPG, ContrastPhys |

### 1.2 整体架构流程

```
RGB视频 → 多区域ROI提取 → 时空特征提取 → 多ROI融合 → 呼吸信号重建 → 呼吸率
```

---

## 二、混合训练策略（你选择的方案）

### 2.1 策略说明

由于 UBFC-rPPG 没有真实呼吸标签，采用以下混合策略：

```
阶段1: 监督预训练（有呼吸标签数据集）
    ↓
    使用 PURE 数据集（可从PPG高精度推导呼吸）
    训练 PhysNet + 呼吸损失函数
    ↓
阶段2: 半监督微调（UBFC-rPPG）
    ↓
    加载预训练模型
    + 有标签部分（PPG推导呼吸）监督损失
    + 无标签部分 一致性损失 + 伪标签损失
    + SNR课程学习逐步放宽伪标签阈值
```

### 2.2 数据集使用计划

| 阶段 | 数据集 | 标签来源 | 用途 |
|------|--------|----------|------|
| 预训练 | PURE | PPG→呼吸（高质量推导） | 监督学习基础能力 |
| 微调-有标签 | UBFC-rPPG 30% | PPG→呼吸（推导标签） | 监督损失 |
| 微调-无标签 | UBFC-rPPG 70% | 无标签 | 一致性+伪标签损失 |
| 测试 | UBFC-rPPG 或 PURE | PPG→呼吸 | 评估 |

---

## 三、实验流程详细步骤

### 阶段1: 环境准备与数据预处理

#### 步骤 1.1: 下载数据集

1. **UBFC-rPPG 数据集**
   - 官网: https://sites.google.com/view/ybenezeth/ubfcrppg
   - 包含: 42个受试者的面部视频 + PPG真值

2. **PURE 数据集** (可选，用于预训练)
   - 官网: https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure
   - 包含: 10个受试者 × 6种运动条件的视频 + PPG真值

#### 步骤 1.2: 配置数据路径

修改配置文件中的路径（以下是示例）:

```yaml
# configs/train_configs/xxx.yaml
TRAIN:
  DATA:
    DATA_PATH: "你的数据集路径/UBFC-rPPG/DATASET_2"
    CACHED_PATH: "你的缓存路径/UBFC-rPPG/PreprocessedData"
```

### 阶段2: 监督预训练（可选但推荐）

#### 步骤 2.1: 运行预训练

```bash
python main.py --config_file configs/train_configs/PRETRAIN_PURE_RESPIRATION.yaml
```

#### 步骤 2.2: 预训练配置说明

关键配置项:
- `MODEL.NAME: Physnet` - 使用标准PhysNet作为backbone
- `RESPIRATION.ENABLE: True` - 启用呼吸检测模式
- `TRAIN.DATA.USE_RESPIRATION_LABEL: True` - 从PPG提取呼吸标签
- `TRAIN.DATA.RESP_EXTRACTION_METHOD: 'fusion'` - 使用融合方法提取呼吸标签

### 阶段3: 半监督微调

#### 步骤 3.1: 运行半监督训练

```bash
python main.py --config_file configs/train_configs/SEMI_SUPERVISED_UBFC_RESPIRATION.yaml
```

#### 步骤 3.2: 半监督配置说明

关键配置项:
- `MODEL.NAME: SemiSupervisedRespiration` - 使用半监督训练器
- `SEMI_SUPERVISED.LAMBDA_CONSISTENCY: 0.5` - 一致性损失权重
- `SEMI_SUPERVISED.LAMBDA_PSEUDO: 0.5` - 伪标签损失权重
- `SEMI_SUPERVISED.SNR_THRESHOLD_START: 10.0` - 初始SNR阈值（严格）
- `SEMI_SUPERVISED.SNR_THRESHOLD_END: 5.0` - 最终SNR阈值（宽松）
- `SEMI_SUPERVISED.CURRICULUM_EPOCHS: 20` - 课程学习持续epoch数

### 阶段4: 测试与评估

#### 步骤 4.1: 运行测试

```bash
python main.py --config_file configs/train_configs/SEMI_SUPERVISED_UBFC_RESPIRATION.yaml
```

测试会自动进行（如果配置 `TOOLBOX_MODE: train_and_test`）

#### 步骤 4.2: 评估指标

呼吸检测会输出以下指标:
- **MAE**: 平均绝对误差 (breaths/min)
- **RMSE**: 均方根误差 (breaths/min)
- **MAPE**: 平均绝对百分比误差 (%)
- **Pearson**: 皮尔逊相关系数
- **SNR**: 信噪比 (dB)

---

## 四、关键文件说明

### 4.1 你需要修改/创建的文件

| 文件路径 | 用途 | 状态 |
|----------|------|------|
| `configs/train_configs/PRETRAIN_PURE_RESPIRATION.yaml` | 预训练配置 | 需创建 |
| `configs/train_configs/SEMI_SUPERVISED_UBFC_RESPIRATION.yaml` | 半监督微调配置 | 需创建 |
| `dataset/data_loader/UBFCrPPGLoader.py` | 数据加载（已支持呼吸标签） | 已完成 |
| `neural_methods/trainer/SemiSupervisedRespirationTrainer.py` | 半监督训练器 | 已完成 |
| `neural_methods/loss/ConsistencyLoss.py` | 一致性损失函数 | 已完成 |
| `evaluation/metrics.py` | 呼吸率评估指标 | 已完成 |
| `evaluation/post_process.py` | 呼吸率计算函数 | 已完成 |
| `dataset/respiration_label_extractor.py` | 呼吸标签提取器 | 已完成 |

### 4.2 已实现的核心功能

1. **呼吸标签提取** (`respiration_label_extractor.py`)
   - RIIV: 基线漂移方法
   - RIAV: 幅度调制方法
   - RSA: HeartPy 的呼吸性窦性心律不齐方法
   - EDR: NeuroKit2 的心率导出呼吸方法
   - fusion: 融合以上所有方法（推荐）

2. **半监督训练** (`SemiSupervisedRespirationTrainer.py`)
   - 监督损失（有标签数据）
   - PSD一致性损失（无标签数据弱/强增强）
   - 伪标签损失（高质量伪标签）
   - SNR课程学习（逐步放宽阈值）

3. **评估指标** (`metrics.py`, `post_process.py`)
   - 呼吸率 MAE/RMSE/MAPE/Pearson
   - 呼吸信号 SNR
   - Bland-Altman 图

---

## 五、呼吸标签提取方法说明

### 5.1 为什么需要从PPG提取呼吸标签?

UBFC-rPPG 数据集只提供了 PPG 真值，没有呼吸真值。但呼吸会通过以下方式调制PPG信号:

1. **RIIV (Respiratory-Induced Intensity Variation)**: 呼吸导致的基线漂移
2. **RIAV (Respiratory-Induced Amplitude Variation)**: 呼吸导致的脉搏波幅度变化  
3. **RSA (Respiratory Sinus Arrhythmia)**: 呼吸导致的心率变异

### 5.2 推荐提取方法

```yaml
TRAIN:
  DATA:
    RESP_EXTRACTION_METHOD: 'fusion'  # 融合多种方法，效果最好
```

可选值:
- `'simple_bandpass'`: 简单带通滤波（0.1-0.5Hz）
- `'riiv'`: 基线漂移方法
- `'riav'`: 幅度调制方法
- `'rsa'`: HeartPy方法
- `'edr'`: NeuroKit2方法
- `'fusion'`: 融合方法（**推荐**）

---

## 六、消融实验设计

### 6.1 建议的消融实验

| 实验ID | 配置 | 目的 |
|--------|------|------|
| A1 | Baseline: 仅面部ROI + PhysNet | 基准性能 |
| A2 | + 呼吸频带损失 | 验证呼吸损失效果 |
| A3 | + 半监督学习(30%标注) | 验证半监督效果 |
| A4 | + SNR课程学习 | 验证课程学习效果 |
| A5 | + PSD一致性损失 | 验证一致性正则化效果 |

### 6.2 与SOTA对比

对比方法（需要自行复现或获取结果）:
- PhysNet (BMVC 2019)
- BigSmall (WACV 2024)
- STREAM-Net (KBS 2025)

---

## 七、常见问题解答

### Q1: 训练时出现 "No respiration label" 错误?

确保配置文件中设置了:
```yaml
TRAIN:
  DATA:
    USE_RESPIRATION_LABEL: True
    RESP_EXTRACTION_METHOD: 'fusion'
```

### Q2: 呼吸率评估结果异常（>60 或 <5 breaths/min）?

检查:
1. 呼吸频带设置是否正确（0.1-0.5Hz）
2. 采样率是否正确（UBFC是30fps）
3. 窗口长度是否足够（建议至少10秒=300帧）

### Q3: 半监督训练伪标签比例太低?

降低SNR阈值或加速课程学习:
```yaml
SEMI_SUPERVISED:
  SNR_THRESHOLD_START: 8.0  # 降低初始阈值
  SNR_THRESHOLD_END: 3.0    # 降低最终阈值
```

### Q4: GPU显存不足?

减少batch_size或chunk_length:
```yaml
TRAIN:
  BATCH_SIZE: 2  # 减少batch size
  DATA:
    PREPROCESS:
      CHUNK_LENGTH: 160  # 减少chunk长度
```

---

## 八、下一步工作建议

### 8.1 短期目标（本次实验）

1. 完成 UBFC-rPPG 上的基线实验
2. 验证半监督学习的有效性
3. 记录消融实验结果

### 8.2 中期目标（论文完善）

1. 实现多ROI融合模块（面部+颈部+胸部）
2. 集成光流引导的胸部ROI定位
3. 实现STLAM侧向注意力机制

### 8.3 长期目标（创新完善）

1. 跨数据集泛化实验
2. 真实场景鲁棒性测试
3. 与SOTA方法全面对比

---

## 九、参考文献

1. MediaPipe (CVPR 2020)
2. RAFT: Optical Flow (ECCV 2020)
3. STREAM-Net (KBS 2025)
4. BigSmall (WACV 2024)
5. FactorizePhys (NeurIPS 2024)
6. PhysFormer++ (IJCV 2023)
7. Semi-rPPG (IEEE TIM 2025)
8. ContrastPhys (ECCV 2022)

---

*文档创建时间: 2026-02-05*
*rPPG-Toolbox 呼吸检测扩展版*
