# Deep Breathing Detection System Based on Deep Learning

基于深度学习的非接触式呼吸率检测系统，通过面部视频分析实现无接触式生命体征监测。

## 📋 项目概述

本项目基于 FactorizePhys 模型进行创新性改进，提出融合运动感知注意力（FSAM）和自监督重建学习的呼吸率检测模型，实现从普通摄像头视频中准确估计呼吸率。

**主要创新点：**
- 设计双注意力机制（全局 + 局部运动感知）
- 多任务学习框架：信号预测 + 运动重建
- 预期性能提升 10-15%

## 🛠 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习框架 | PyTorch |
| 核心网络 | 3D-CNN, ResNet-3D |
| 注意力机制 | Factorized Spatio-Temporal Attention (FSAM) |
| 数据增强 | 运动感知数据增强 |
| 训练策略 | 自监督预训练 + 微调 |

## 📁 项目结构

```
rppg-respiratory-detection/
├── models/
│   ├── factorize_phys.py      # FactorizePhys 模型
│   ├── fsam_attention.py      # 因子化注意力模块
│   └── respiratory_net.py     # 改进的呼吸检测网络
├── datasets/
│   ├── rpi_dataset.py         # RPI 数据集加载
│   ├── ubr_dataset.py         # UBR 数据集加载
│   └── transforms.py          # 数据增强变换
├── training/
│   ├── train.py               # 训练脚本
│   ├── config.py              # 配置文件
│   └── metrics.py             # 评估指标
├── inference/
│   └── predict.py             # 推理脚本
├── evaluation/
│   └── evaluate.py            # 模型评估
├── notebooks/
│   └── exploration.ipynb      # 探索性分析
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 环境配置

```bash
pip install -r requirements.txt
```

**依赖：**
- Python 3.9+
- PyTorch 2.0+
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn

### 数据集准备

本代码支持以下公开数据集：
- **RPI Photoplethysmography Dataset**: https://doi.org/10.5281/zenodo.5516265
- **UBFace Dataset**: https://github.com/xuxtao/UBFace

下载后解压到 `datasets/` 目录。

### 模型训练

```bash
# 使用默认配置训练
python training/train.py

# 使用自定义配置
python training/train.py --config configs/exp_a.yaml
```

### 推理演示

```bash
# 对单张视频进行推理
python inference/predict.py --video input.mp4 --checkpoint model.pth

# 批量处理
python inference/predict.py --dataset datasets/rpi/ --checkpoint model.pth --output results/
```

## 📊 实验结果

### 评估指标

| 数据集 | MAE (bpm) | RMSE (bpm) | 相关系数 |
|--------|-----------|------------|----------|
| RPI | - | - | - |
| UBR | - | - | - |

*注：具体数值需在训练完成后更新*

### 可视化

训练过程中的损失曲线和评估结果可通过 TensorBoard 查看：

```bash
tensorboard --logdir logs/
```

## 🔬 方法详解

### Factorized Spatio-Temporal Attention Module (FSAM)

FSAM 将空间和时间维度的注意力机制分解，降低计算复杂度同时保持性能：

1. **空间注意力**：捕捉面部区域的关键运动信息
2. **时间注意力**：建模呼吸周期的时序依赖性
3. **因子化分解**：将 3D 注意力分解为 2D 空间 + 1D 时间

### 多任务学习框架

同时优化两个任务以提高泛化能力：
- **呼吸率预测**：主要目标，输出呼吸频率
- **运动重建**：辅助任务，重建面部运动序列

## 📝 引用

如果此代码对你的研究有帮助，请引用：

```bibtex
@misc{zhou2024rppg,
  title={Deep Learning-Based Contactless Respiration Detection},
  author={Zhou Shuyi},
  year={2024},
  note={Master's Thesis Project, Macau University of Science and Technology}
}
```

## 📄 许可证

MIT License

---

> 本项目是澳门理工大学大数据与物联网硕士课程的研究项目，研究方向为生理信号检测与深度学习算法。
