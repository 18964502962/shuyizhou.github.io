# rPPG-Toolbox 呼吸率检测改造实施清单

## 📋 快速检查清单

### Phase 1: 环境准备 ✅
- [ ] 安装 MediaPipe: `pip install mediapipe`
- [ ] 确认 OpenCV 包含光流功能: `pip install opencv-contrib-python`
- [ ] 检查现有依赖是否完整

### Phase 2: 数据预处理层 🔄
- [ ] 创建 `dataset/data_loader/OpticalFlowLoader.py`
  - [ ] 实现 `detect_chest_roi()` - MediaPipe 检测胸部区域
  - [ ] 实现 `compute_optical_flow()` - Farneback 稠密光流
  - [ ] 实现 `select_respiratory_roi()` - 根据运动幅度选择 ROI
  - [ ] 重写 `preprocess()` - 集成光流计算
- [ ] 修改 `dataset/data_loader/BaseLoader.py`
  - [ ] 支持 4 通道输入（RGB + Motion）
  - [ ] 添加光流预处理选项

### Phase 3: 模型层 🔄
- [ ] 修改 `neural_methods/model/PhysNet.py`
  - [ ] 将 `ConvBlock1` 输入通道改为可配置（默认 4）
  - [ ] 创建 `ModifiedPhysNet` 类
- [ ] 创建 `neural_methods/model/SemiSupervisedRespirationNet.py`
  - [ ] 实现时间增强（时间位移、时间反转）
  - [ ] 实现一致性分支
  - [ ] 集成 PhysNet backbone
- [ ] 创建 `neural_methods/loss/ConsistencyLoss.py`
  - [ ] 实现功率谱密度（PSD）计算
  - [ ] 实现一致性损失（MSE on PSD）

### Phase 4: 训练器层 🔄
- [ ] 创建 `neural_methods/trainer/SemiSupervisedRespirationTrainer.py`
  - [ ] 实现 `calculate_respiration_snr()` - SNR 计算
  - [ ] 实现 `select_pseudo_labels()` - 课程伪标签选择
  - [ ] 实现训练循环（有监督 + 半监督）
  - [ ] 实现一致性损失计算
  - [ ] 实现验证循环

### Phase 5: 评估和配置 🔄
- [ ] 修改 `evaluation/metrics.py`
  - [ ] 添加 `calculate_respiration_rate()` - 呼吸率计算（0.1-0.5 Hz）
  - [ ] 修改 FFT 评估使用呼吸频带
- [ ] 创建 `configs/train_configs/RESPIRATION_PHYSNET_SEMISUPERVISED.yaml`
  - [ ] 配置胸部 ROI 检测参数
  - [ ] 配置光流参数
  - [ ] 配置半监督学习参数
  - [ ] 配置课程学习参数

### Phase 6: 测试和验证 ✅
- [ ] 单元测试：MediaPipe 胸部检测
- [ ] 单元测试：光流计算
- [ ] 单元测试：SNR 计算
- [ ] 单元测试：模型前向传播（4 通道）
- [ ] 集成测试：完整训练流程（小数据集）

### Phase 7: 实验运行 🔬
- [ ] Baseline 1: 仅 rPPG（3 通道 RGB）
- [ ] Baseline 2: 仅光流（1 通道 Motion）
- [ ] 实验 1: RGB + Motion（4 通道拼接）
- [ ] 实验 2: 半监督（10% 有标注）
- [ ] 实验 3: 半监督（30% 有标注）
- [ ] 实验 4: 跨数据集验证（UBFC-rPPG → PURE）

---

## 🚀 快速开始命令

### 1. 数据预处理
```bash
# 修改配置文件中的 DATA_PATH 和 CACHED_PATH
# 设置 DO_PREPROCESS: True
python main.py --config_file configs/train_configs/RESPIRATION_PHYSNET_SEMISUPERVISED.yaml
```

### 2. 训练模型
```bash
# 设置 DO_PREPROCESS: False（使用已预处理数据）
python main.py --config_file configs/train_configs/RESPIRATION_PHYSNET_SEMISUPERVISED.yaml
```

### 3. 测试模型
```bash
# 设置 TOOLBOX_MODE: "only_test"
# 设置 MODEL_PATH: "./runs/respiration_exp/.../best_model.pth"
python main.py --config_file configs/train_configs/RESPIRATION_PHYSNET_SEMISUPERVISED.yaml
```

---

## 📝 关键代码片段位置

### MediaPipe 胸部检测
```python
# OpticalFlowLoader.py
import mediapipe as mp
mp_pose = mp.solutions.pose.Pose()
# 检测关键点：左肩(11)、右肩(12)、左髋(23)、右髋(24)
```

### 光流计算
```python
# OpticalFlowLoader.py
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, curr_gray, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)
```

### SNR 计算
```python
# SemiSupervisedRespirationTrainer.py
freqs = torch.fft.rfftfreq(signal.shape[-1], 1/fps)
resp_mask = (freqs >= 0.1) & (freqs <= 0.5)  # 呼吸频带
```

### 一致性损失
```python
# ConsistencyLoss.py
psd_weak = torch.abs(torch.fft.rfft(pred_weak, dim=-1)) ** 2
psd_strong = torch.abs(torch.fft.rfft(pred_strong, dim=-1)) ** 2
loss = F.mse_loss(psd_weak, psd_strong)
```

---

## ⚠️ 常见问题

### Q1: MediaPipe 检测不到关键点？
- **解决**: 检查视频质量，确保人物完整出现在画面中
- **备选**: 使用固定 ROI（基于人脸检测结果下移）

### Q2: 光流计算太慢？
- **解决**: 降低光流分辨率，或使用稀疏光流（LK）
- **优化**: 仅在 ROI 区域计算光流

### Q3: SNR 计算为负值？
- **解决**: 检查信号预处理，确保呼吸信号存在
- **调试**: 可视化功率谱，确认呼吸频带是否有峰值

### Q4: 伪标签质量差？
- **解决**: 提高初始 SNR 阈值（从 10.0 提升到 12.0）
- **调整**: 降低伪标签损失权重（从 0.5 降到 0.3）

---

## 📊 预期结果

### Baseline 性能（仅 rPPG）
- MAE: ~2-3 bpm
- RMSE: ~3-4 bpm
- Pearson: ~0.7-0.8

### 融合后性能（RGB + Motion）
- MAE: ~1.5-2 bpm（提升 20-30%）
- RMSE: ~2-3 bpm
- Pearson: ~0.8-0.9

### 半监督性能（30% 有标注）
- MAE: ~2-2.5 bpm（接近全监督）
- RMSE: ~2.5-3 bpm
- Pearson: ~0.75-0.85

---

## 📚 参考文档

- 详细规划: `RESPIRATION_RPPG_TRANSFORMATION_PLAN.md`
- rPPG-Toolbox 文档: `README.md`
- MediaPipe Pose: https://google.github.io/mediapipe/solutions/pose
- OpenCV 光流: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html

---

**最后更新**: 2025-01-29
