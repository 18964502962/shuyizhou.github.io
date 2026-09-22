# rPPG-Toolbox 呼吸率检测改造规划

## 项目概述

将 rPPG-Toolbox 从心率（HR）检测改造为呼吸率（RR）检测，并加入以下创新点：
1. **MediaPipe + 光流定位胸部 ROI**
2. **半监督学习框架**（基于 Semi-rPPG 2025）
3. **课程学习策略**（Curriculum Learning）
4. **光流与 rPPG 信号融合**（路径 B：特征层融合）

---

## 一、数据预处理层改造

### 1.1 新建光学流数据集类

**文件路径**: `dataset/data_loader/OpticalFlowLoader.py`

**核心功能**:
- 继承 `BaseLoader`
- 使用 MediaPipe 检测肩膀和髋部关键点，定位胸部区域
- 计算稠密光流（Farneback）
- 根据光流运动幅度选择呼吸相关的子区域

**关键方法**:

```python
class OpticalFlowLoader(BaseLoader):
    def __init__(self, name, data_path, config_data, device=None):
        super().__init__(name, data_path, config_data, device)
        # 初始化 MediaPipe Pose
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
    
    def detect_chest_roi(self, frame):
        """使用 MediaPipe 检测肩膀和髋部，定位胸部 ROI"""
        # 检测关键点：左肩、右肩、左髋、右髋
        # 返回胸部区域的 bbox: [x, y, w, h]
        pass
    
    def compute_optical_flow(self, prev_frame, curr_frame, roi_bbox):
        """计算胸部区域的稠密光流"""
        # 1. 裁剪 ROI 区域
        # 2. 转换为灰度图
        # 3. cv2.calcOpticalFlowFarneback()
        # 4. 返回光流场 (H, W, 2) 和运动幅度 (H, W)
        pass
    
    def select_respiratory_roi(self, flow_magnitude, top_k=0.3):
        """根据运动幅度选择呼吸相关的子区域"""
        # 1. 计算光流幅度
        # 2. 选择运动幅度最大的 top_k% 区域
        # 3. 返回最终 ROI mask
        pass
    
    def preprocess(self, frames, bvps, config_preprocess):
        """重写预处理方法，加入光流计算"""
        # 1. 检测胸部 ROI（使用第一帧）
        # 2. 对每对相邻帧计算光流
        # 3. 选择呼吸相关子区域
        # 4. 裁剪并 resize 到 128x128
        # 5. 将光流幅度作为第 4 通道（R, G, B, Motion）
        pass
```

### 1.2 修改 BaseLoader

**文件**: `dataset/data_loader/BaseLoader.py`

**修改点**:
- 在 `preprocess()` 方法中添加光流计算选项
- 支持 4 通道输入（RGB + Motion）

---

## 二、模型层改造

### 2.1 新建半监督呼吸网络

**文件路径**: `neural_methods/model/SemiSupervisedRespirationNet.py`

**架构设计**:
- **Backbone**: PhysNet（3D-CNN）
- **输入**: 4 通道（R, G, B, Motion Magnitude）
- **输出**: 呼吸信号波形

**核心组件**:

```python
class SemiSupervisedRespirationNet(nn.Module):
    def __init__(self, frames=128, in_channels=4):
        super().__init__()
        # 修改 PhysNet 的输入通道数
        self.backbone = ModifiedPhysNet(frames=frames, in_channels=in_channels)
        
        # 一致性分支（用于半监督）
        self.consistency_head = nn.Sequential(
            nn.Linear(frames, 64),
            nn.ReLU(),
            nn.Linear(64, frames)
        )
    
    def forward(self, x, apply_augmentation=False):
        """
        Args:
            x: [B, C, T, H, W] 其中 C=4 (RGB + Motion)
            apply_augmentation: 是否应用时间增强
        """
        if apply_augmentation:
            # 弱增强：时间位移
            x_weak = self.temporal_shift(x, shift_range=5)
            # 强增强：时间反转
            x_strong = self.temporal_reverse(x)
            
            pred_weak = self.backbone(x_weak)
            pred_strong = self.backbone(x_strong)
            
            return pred_weak, pred_strong
        
        pred = self.backbone(x)
        return pred
    
    def temporal_shift(self, x, shift_range=5):
        """时间位移增强"""
        pass
    
    def temporal_reverse(self, x):
        """时间反转增强"""
        return torch.flip(x, dims=[2])
```

### 2.2 修改 PhysNet 支持 4 通道输入

**文件**: `neural_methods/model/PhysNet.py`

**修改点**:
- 将 `ConvBlock1` 的输入通道从 3 改为 4（或可配置）
- 保持其他结构不变

```python
class ModifiedPhysNet(nn.Module):
    def __init__(self, frames=128, in_channels=4):
        super().__init__()
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            # ... 其余保持不变
        )
```

### 2.3 一致性损失函数

**文件路径**: `neural_methods/loss/ConsistencyLoss.py`

```python
class ConsistencyLoss(nn.Module):
    """
    一致性损失：确保弱增强和强增强后的预测功率谱保持一致
    """
    def __init__(self, fps=30):
        super().__init__()
        self.fps = fps
    
    def forward(self, pred_weak, pred_strong):
        # 1. 计算功率谱密度 (PSD)
        psd_weak = self.compute_psd(pred_weak)
        psd_strong = self.compute_psd(pred_strong)
        
        # 2. 计算 MSE 损失
        loss = F.mse_loss(psd_weak, psd_strong)
        return loss
    
    def compute_psd(self, signal):
        """计算功率谱密度"""
        # FFT -> 功率谱
        fft = torch.fft.rfft(signal, dim=-1)
        psd = torch.abs(fft) ** 2
        return psd
```

---

## 三、训练器层改造

### 3.1 新建半监督训练器

**文件路径**: `neural_methods/trainer/SemiSupervisedRespirationTrainer.py`

**核心逻辑**:

```python
class SemiSupervisedRespirationTrainer(BaseTrainer):
    def __init__(self, model, config, device):
        super().__init__()
        self.model = model
        self.config = config
        self.device = device
        
        # 损失函数
        self.supervised_loss = nn.MSELoss()  # 有监督损失（MSE）
        self.consistency_loss = ConsistencyLoss(fps=config.TRAIN.DATA.FS)
        
        # 课程学习参数
        self.initial_snr_threshold = 10.0  # 初始 SNR 阈值
        self.final_snr_threshold = 5.0    # 最终 SNR 阈值
        self.max_epochs = config.TRAIN.EPOCHS
    
    def calculate_respiration_snr(self, signal, fps=30):
        """
        计算呼吸信号的信噪比 (SNR)
        呼吸频带：0.1-0.5 Hz (6-30 bpm)
        """
        # FFT
        fft = torch.fft.rfft(signal, dim=-1)
        power = torch.abs(fft) ** 2
        
        # 定义呼吸频带
        freqs = torch.fft.rfftfreq(signal.shape[-1], 1/fps)
        resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
        
        # 计算 SNR
        resp_power = power[..., resp_mask].sum(dim=-1)
        total_power = power.sum(dim=-1)
        noise_power = total_power - resp_power
        
        snr = 10 * torch.log10(resp_power / (noise_power + 1e-8))
        return snr
    
    def select_pseudo_labels(self, model_outputs, current_epoch):
        """
        课程伪标签选择：根据 SNR 和 epoch 动态调整阈值
        """
        # 计算当前 epoch 的 SNR 阈值（线性衰减）
        current_threshold = self.initial_snr_threshold - \
            (self.initial_snr_threshold - self.final_snr_threshold) * \
            (current_epoch / self.max_epochs)
        
        # 计算所有样本的 SNR
        snrs = self.calculate_respiration_snr(model_outputs)
        
        # 选择 SNR > threshold 的样本
        valid_mask = snrs > current_threshold
        
        # 课程比例：随 epoch 增加，选择比例从 0.2 提升到 0.8
        ratio = 0.2 + (0.6 * (current_epoch / self.max_epochs))
        num_to_select = int(len(snrs) * ratio)
        
        # 按 SNR 降序排列，选择前 ratio% 的高质量样本
        _, top_indices = torch.topk(snrs, k=min(num_to_select, len(snrs)))
        
        return top_indices, valid_mask
    
    def train(self, data_loader):
        """
        训练循环：有监督 + 半监督（伪标签）
        """
        for epoch in range(self.max_epoch_num):
            # === 阶段 1：有监督训练 ===
            self.model.train()
            for batch in data_loader["train"]:  # 有标注数据
                data, labels = batch[0].to(self.device), batch[1].to(self.device)
                
                # 前向传播
                pred = self.model(data)
                
                # 有监督损失
                sup_loss = self.supervised_loss(pred, labels)
                
                # 反向传播
                self.optimizer.zero_grad()
                sup_loss.backward()
                self.optimizer.step()
            
            # === 阶段 2：半监督训练（伪标签）===
            if data_loader.get("unlabeled") is not None:
                self.model.eval()
                pseudo_labels_list = []
                pseudo_data_list = []
                
                # 生成伪标签
                with torch.no_grad():
                    for batch in data_loader["unlabeled"]:
                        data_unlabeled = batch[0].to(self.device)
                        pred_unlabeled = self.model(data_unlabeled)
                        
                        # 计算 SNR 并筛选
                        top_indices, valid_mask = self.select_pseudo_labels(
                            pred_unlabeled, epoch
                        )
                        
                        # 保存高质量伪标签
                        pseudo_labels_list.append(pred_unlabeled[top_indices])
                        pseudo_data_list.append(data_unlabeled[top_indices])
                
                # 使用伪标签训练
                if len(pseudo_labels_list) > 0:
                    self.model.train()
                    pseudo_data = torch.cat(pseudo_data_list, dim=0)
                    pseudo_labels = torch.cat(pseudo_labels_list, dim=0)
                    
                    # 应用增强并计算一致性损失
                    pred_weak, pred_strong = self.model(
                        pseudo_data, apply_augmentation=True
                    )
                    
                    # 一致性损失
                    cons_loss = self.consistency_loss(pred_weak, pred_strong)
                    
                    # 伪标签监督损失
                    pseudo_sup_loss = self.supervised_loss(pred_weak, pseudo_labels)
                    
                    # 总损失
                    total_loss = cons_loss + 0.5 * pseudo_sup_loss
                    
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
            
            # 验证和保存模型
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
            self.save_model(epoch)
```

---

## 四、配置文件改造

### 4.1 新建呼吸检测配置文件

**文件路径**: `configs/train_configs/RESPIRATION_PHYSNET_SEMISUPERVISED.yaml`

```yaml
BASE: ['']
TOOLBOX_MODE: "train_and_test"

TRAIN:
  EPOCHS: 100
  BATCH_SIZE: 8
  DATA:
    FS: 30
    DATASET: UBFC-rPPG  # 或 PURE
    DO_PREPROCESS: True
    DATA_FORMAT: NCDHW
    DATA_PATH: "/your/path/to/UBFC-rPPG"
    CACHED_PATH: "/your/path/to/cached_respiration"
    EXP_DATA_NAME: "Respiration_PhysNet_SemiSupervised"
    BEGIN: 0.0
    END: 0.8
    PREPROCESS:
      DATA_TYPE: ['DiffNormalized']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 160
      # === 新增：胸部 ROI 检测 ===
      CROP_CHEST:
        DO_CROP_CHEST: True
        USE_MEDIAPIPE: True
        USE_OPTICAL_FLOW: True
        OPTICAL_FLOW_METHOD: "Farneback"
        SELECT_ROI_METHOD: "magnitude_top_k"  # 根据光流幅度选择
        TOP_K_RATIO: 0.3  # 选择运动幅度最大的 30% 区域
      RESIZE:
        H: 128
        W: 128
      # === 信号融合配置 ===
      FUSION:
        ENABLE_FUSION: True
        FUSION_METHOD: "channel_concat"  # 通道拼接（简易方案）
        # 可选: "clifford" (Clifford 几何代数融合，高级方案)

MODEL:
  NAME: SemiSupervisedRespirationNet
  BACKBONE: PhysNet
  IN_CHANNELS: 4  # RGB + Motion
  FRAMES: 128
  DROP_RATE: 0.1

# === 半监督学习配置 ===
SEMISUPERVISED:
  ENABLE: True
  UNLABELED_DATA_PATH: "/your/path/to/unlabeled_data"
  CONSISTENCY_LOSS_WEIGHT: 1.0
  PSEUDO_LABEL_LOSS_WEIGHT: 0.5
  # === 课程学习配置 ===
  CURRICULUM:
    INITIAL_SNR_THRESHOLD: 10.0
    FINAL_SNR_THRESHOLD: 5.0
    INITIAL_SELECTION_RATIO: 0.2  # 初始选择 20% 高质量样本
    FINAL_SELECTION_RATIO: 0.8    # 最终选择 80% 样本

TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'BA']
  USE_LAST_EPOCH: False
  DATA:
    FS: 30
    DATASET: PURE
    DO_PREPROCESS: False
    DATA_FORMAT: NCDHW
    DATA_PATH: "/your/path/to/PURE"
    CACHED_PATH: "/your/path/to/cached_respiration_test"
    EXP_DATA_NAME: ""
    BEGIN: 0.8
    END: 1.0
    PREPROCESS:
      DATA_TYPE: ['DiffNormalized']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 160
      CROP_CHEST:
        DO_CROP_CHEST: True
        USE_MEDIAPIPE: True
        USE_OPTICAL_FLOW: True
      RESIZE:
        H: 128
        W: 128
  EVALUATION_METHOD: "FFT"  # 呼吸率评估使用 FFT（频域分析）
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: True
    WINDOW_SIZE: 30  # 30 秒窗口（呼吸信号需要更长窗口）

DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1
LOG:
  PATH: runs/respiration_exp
```

---

## 五、评估指标改造

### 5.1 修改评估频率范围

**文件**: `evaluation/metrics.py`

**修改点**:
- 心率评估：0.7-3.0 Hz (42-180 bpm)
- **呼吸率评估：0.1-0.5 Hz (6-30 bpm)**

```python
def calculate_respiration_rate(pred_signal, label_signal, fps=30):
    """
    计算呼吸率（RR）
    呼吸频带：0.1-0.5 Hz
    """
    # FFT
    pred_fft = np.fft.rfft(pred_signal)
    label_fft = np.fft.rfft(label_signal)
    
    freqs = np.fft.rfftfreq(len(pred_signal), 1/fps)
    
    # 呼吸频带 mask
    resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
    
    # 找到峰值频率
    pred_power = np.abs(pred_fft[resp_mask]) ** 2
    label_power = np.abs(label_fft[resp_mask]) ** 2
    
    pred_rr = freqs[resp_mask][np.argmax(pred_power)] * 60  # 转换为 bpm
    label_rr = freqs[resp_mask][np.argmax(label_power)] * 60
    
    return pred_rr, label_rr
```

---

## 六、实验方案

### 6.1 Baseline 实验

1. **单模态测试**:
   - **仅 rPPG（颜色）**: 使用标准 PhysNet，输入 3 通道 RGB
   - **仅光流（位移）**: 仅使用光流幅度作为输入

2. **融合测试**:
   - **RGB + Motion（通道拼接）**: 4 通道输入
   - **RGB + Motion（Clifford 融合）**: 高级几何代数融合（可选）

### 6.2 半监督实验

1. **有标注数据比例**:
   - 10% 有标注 + 90% 无标注
   - 30% 有标注 + 70% 无标注
   - 50% 有标注 + 50% 无标注

2. **跨数据集验证**:
   - 训练集：UBFC-rPPG（有标注）
   - 测试集：PURE（无标注，使用伪标签）

### 6.3 消融实验

1. **光流 ROI 选择策略**:
   - 固定 ROI（MediaPipe 检测的胸部区域）
   - 动态 ROI（根据光流幅度选择）

2. **一致性损失权重**:
   - 0.5, 1.0, 2.0

3. **课程学习策略**:
   - 固定 SNR 阈值 vs 动态衰减阈值

---

## 七、实施步骤

### Phase 1: 数据预处理层（1-2 周）
1. ✅ 安装 MediaPipe: `pip install mediapipe`
2. ✅ 实现 `OpticalFlowLoader.py`
3. ✅ 修改 `BaseLoader.py` 支持 4 通道
4. ✅ 测试胸部 ROI 检测和光流计算

### Phase 2: 模型层（1-2 周）
1. ✅ 修改 `PhysNet.py` 支持 4 通道输入
2. ✅ 实现 `SemiSupervisedRespirationNet.py`
3. ✅ 实现 `ConsistencyLoss.py`
4. ✅ 测试模型前向传播

### Phase 3: 训练器层（1-2 周）
1. ✅ 实现 `SemiSupervisedRespirationTrainer.py`
2. ✅ 实现 SNR 计算和伪标签筛选
3. ✅ 实现课程学习逻辑
4. ✅ 测试训练循环

### Phase 4: 评估和调试（1-2 周）
1. ✅ 修改评估指标（呼吸频带）
2. ✅ 运行 Baseline 实验
3. ✅ 运行半监督实验
4. ✅ 结果分析和论文撰写

---

## 八、依赖安装

```bash
# 新增依赖
pip install mediapipe
pip install opencv-contrib-python  # 确保包含 calcOpticalFlowFarneback

# 现有依赖（rPPG-Toolbox 已有）
# torch, torchvision, numpy, scipy, opencv-python, tqdm, etc.
```

---

## 九、文件结构总结

```
rPPG-Toolbox-main/
├── dataset/
│   └── data_loader/
│       ├── BaseLoader.py                    # 修改：支持 4 通道
│       └── OpticalFlowLoader.py             # 新建：光流数据集加载器
├── neural_methods/
│   ├── model/
│   │   ├── PhysNet.py                       # 修改：支持 4 通道输入
│   │   └── SemiSupervisedRespirationNet.py  # 新建：半监督呼吸网络
│   ├── loss/
│   │   └── ConsistencyLoss.py               # 新建：一致性损失
│   └── trainer/
│       └── SemiSupervisedRespirationTrainer.py  # 新建：半监督训练器
├── evaluation/
│   └── metrics.py                           # 修改：呼吸率评估
└── configs/
    └── train_configs/
        └── RESPIRATION_PHYSNET_SEMISUPERVISED.yaml  # 新建：呼吸检测配置
```

---

## 十、参考文献

1. **Semi-rPPG (2025)**: 半监督学习框架
2. **CliffPhys (2024)**: Clifford 几何代数融合
3. **Non-Contact Breathing Rate Detection Using Optical Flow (2023)**: 光流呼吸检测
4. **PhysNet (BMVC 2019)**: 3D-CNN 骨干网络

---

## 注意事项

1. **数据格式**: 确保数据集包含呼吸率标签（RR），而非心率（HR）
2. **频率范围**: 呼吸率评估使用 0.1-0.5 Hz，而非心率的 0.7-3.0 Hz
3. **窗口长度**: 呼吸信号需要更长的分析窗口（建议 30 秒）
4. **光流计算**: 确保使用稠密光流（Farneback），而非稀疏光流
5. **MediaPipe**: 需要检测 Pose 关键点（肩膀、髋部），而非 Face

---

**最后更新**: 2025-01-29
**作者**: 呼吸率检测项目组
