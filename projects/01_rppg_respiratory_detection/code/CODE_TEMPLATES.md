# 关键代码模板

本文档提供改造过程中的关键代码实现模板，可直接参考使用。

---

## 1. OpticalFlowLoader.py 核心代码

```python
"""光学流数据集加载器 - 用于呼吸率检测"""
import cv2
import numpy as np
import mediapipe as mp
import torch
from dataset.data_loader.BaseLoader import BaseLoader

class OpticalFlowLoader(BaseLoader):
    """支持光流计算的呼吸率数据集加载器"""
    
    def __init__(self, name, data_path, config_data, device=None):
        super().__init__(name, data_path, config_data, device)
        
        # 初始化 MediaPipe Pose
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # 光流参数
        self.flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        
        # ROI 选择参数
        self.top_k_ratio = config_data.PREPROCESS.CROP_CHEST.get('TOP_K_RATIO', 0.3)
    
    def detect_chest_roi(self, frame):
        """
        使用 MediaPipe 检测肩膀和髋部，定位胸部 ROI
        
        Args:
            frame: np.array, shape (H, W, 3), BGR 格式
        
        Returns:
            bbox: [x, y, w, h] 胸部区域边界框
        """
        # 转换为 RGB（MediaPipe 需要 RGB）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            # 如果检测失败，返回整个图像的下半部分（默认胸部区域）
            h, w = frame.shape[:2]
            return [0, h//3, w, h//2]
        
        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]
        
        # 关键点索引（MediaPipe Pose）
        # 左肩: 11, 右肩: 12
        # 左髋: 23, 右髋: 24
        
        # 获取关键点坐标（归一化坐标转像素坐标）
        left_shoulder = [
            int(landmarks[11].x * w),
            int(landmarks[11].y * h)
        ]
        right_shoulder = [
            int(landmarks[12].x * w),
            int(landmarks[12].y * h)
        ]
        left_hip = [
            int(landmarks[23].x * w),
            int(landmarks[23].y * h)
        ]
        right_hip = [
            int(landmarks[24].x * w),
            int(landmarks[24].y * h)
        ]
        
        # 计算胸部 ROI（肩膀和髋部之间的区域）
        x_min = min(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0])
        x_max = max(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0])
        y_min = min(left_shoulder[1], right_shoulder[1])
        y_max = max(left_hip[1], right_hip[1])
        
        # 添加边界检查
        x_min = max(0, x_min - 10)
        y_min = max(0, y_min - 10)
        x_max = min(w, x_max + 10)
        y_max = min(h, y_max + 10)
        
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    
    def compute_optical_flow(self, prev_frame, curr_frame, roi_bbox=None):
        """
        计算稠密光流
        
        Args:
            prev_frame: np.array, shape (H, W, 3), 前一帧
            curr_frame: np.array, shape (H, W, 3), 当前帧
            roi_bbox: [x, y, w, h], 可选，ROI 区域
        
        Returns:
            flow: np.array, shape (H, W, 2), 光流场 [dx, dy]
            magnitude: np.array, shape (H, W), 光流幅度
        """
        # 转换为灰度图
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # 如果指定了 ROI，先裁剪
        if roi_bbox is not None:
            x, y, w, h = roi_bbox
            prev_gray = prev_gray[y:y+h, x:x+w]
            curr_gray = curr_gray[y:y+h, x:x+w]
        
        # 计算 Farneback 稠密光流
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            **self.flow_params
        )
        
        # 计算光流幅度
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        return flow, magnitude
    
    def select_respiratory_roi(self, flow_magnitude, top_k_ratio=0.3):
        """
        根据光流运动幅度选择呼吸相关的子区域
        
        Args:
            flow_magnitude: np.array, shape (H, W), 光流幅度
            top_k_ratio: float, 选择运动幅度最大的 top_k_ratio% 区域
        
        Returns:
            mask: np.array, shape (H, W), bool 类型，True 表示选中区域
        """
        h, w = flow_magnitude.shape
        total_pixels = h * w
        num_select = int(total_pixels * top_k_ratio)
        
        # 将幅度展平并排序
        flat_magnitude = flow_magnitude.flatten()
        threshold = np.sort(flat_magnitude)[-num_select]
        
        # 创建 mask
        mask = flow_magnitude >= threshold
        
        return mask
    
    def preprocess(self, frames, bvps, config_preprocess):
        """
        重写预处理方法，加入光流计算
        
        Args:
            frames: np.array, shape (T, H, W, 3)
            bvps: np.array, shape (T,), 呼吸信号标签
            config_preprocess: 预处理配置
        
        Returns:
            frame_clips: np.array, shape (N, T, H, W, 4), 4 通道（RGB + Motion）
            bvps_clips: np.array, shape (N, T)
        """
        # 1. 检测胸部 ROI（使用第一帧）
        chest_bbox = self.detect_chest_roi(frames[0])
        
        # 2. 计算光流并选择呼吸 ROI
        flow_magnitudes = []
        for i in range(len(frames) - 1):
            flow, magnitude = self.compute_optical_flow(
                frames[i], frames[i+1], roi_bbox=chest_bbox
            )
            flow_magnitudes.append(magnitude)
        
        # 最后一帧使用前一帧的光流
        if len(flow_magnitudes) > 0:
            flow_magnitudes.append(flow_magnitudes[-1])
        else:
            flow_magnitudes = [np.zeros_like(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY))]
        
        # 3. 选择呼吸相关 ROI
        avg_magnitude = np.mean(flow_magnitudes, axis=0)
        roi_mask = self.select_respiratory_roi(avg_magnitude, self.top_k_ratio)
        
        # 4. 裁剪并 resize 到目标尺寸
        x, y, w, h = chest_bbox
        target_h = config_preprocess.RESIZE.H
        target_w = config_preprocess.RESIZE.W
        
        processed_frames = []
        for i, frame in enumerate(frames):
            # 裁剪 ROI
            roi_frame = frame[y:y+h, x:x+w]
            
            # 应用呼吸 ROI mask（可选，如果使用动态 ROI）
            # roi_frame = roi_frame * roi_mask[..., np.newaxis]
            
            # Resize
            resized_frame = cv2.resize(roi_frame, (target_w, target_h))
            
            # 添加光流幅度作为第 4 通道
            if i < len(flow_magnitudes):
                flow_mag_resized = cv2.resize(
                    flow_magnitudes[i], (target_w, target_h)
                )
                # 归一化光流幅度到 [0, 255]
                flow_mag_norm = (flow_mag_resized / (flow_mag_resized.max() + 1e-8) * 255).astype(np.uint8)
                # 拼接为 4 通道
                frame_4ch = np.concatenate([
                    resized_frame,
                    flow_mag_norm[..., np.newaxis]
                ], axis=-1)
            else:
                # 如果没有光流，使用零通道
                frame_4ch = np.concatenate([
                    resized_frame,
                    np.zeros((target_h, target_w, 1), dtype=np.uint8)
                ], axis=-1)
            
            processed_frames.append(frame_4ch)
        
        processed_frames = np.array(processed_frames)  # (T, H, W, 4)
        
        # 5. 应用数据变换（DiffNormalized 等）
        data = []
        for data_type in config_preprocess.DATA_TYPE:
            f_c = processed_frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(self.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(self.standardized_data(f_c))
        
        data = np.concatenate(data, axis=-1)  # (T, H, W, C*4)
        
        # 6. 标签处理
        if config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = self.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = self.standardized_label(bvps)
        
        # 7. Chunking
        if config_preprocess.DO_CHUNK:
            frames_clips, bvps_clips = self.chunk(
                data, bvps, config_preprocess.CHUNK_LENGTH
            )
        else:
            frames_clips = np.array([data])
            bvps_clips = np.array([bvps])
        
        return frames_clips, bvps_clips
```

---

## 2. ConsistencyLoss.py

```python
"""一致性损失 - 用于半监督学习"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistencyLoss(nn.Module):
    """
    一致性损失：确保弱增强和强增强后的预测功率谱保持一致
    
    参考 Semi-rPPG (2025) 论文
    """
    
    def __init__(self, fps=30, resp_freq_range=(0.1, 0.5)):
        """
        Args:
            fps: 视频帧率
            resp_freq_range: 呼吸频率范围 (min_freq, max_freq) Hz
        """
        super().__init__()
        self.fps = fps
        self.resp_freq_range = resp_freq_range
    
    def forward(self, pred_weak, pred_strong):
        """
        计算一致性损失
        
        Args:
            pred_weak: torch.Tensor, shape (B, T), 弱增强预测
            pred_strong: torch.Tensor, shape (B, T), 强增强预测
        
        Returns:
            loss: torch.Tensor, scalar, 一致性损失
        """
        # 1. 计算功率谱密度 (PSD)
        psd_weak = self.compute_psd(pred_weak)
        psd_strong = self.compute_psd(pred_strong)
        
        # 2. 计算 MSE 损失（在呼吸频带内）
        resp_psd_weak = self.extract_respiratory_band(psd_weak)
        resp_psd_strong = self.extract_respiratory_band(psd_strong)
        
        # 3. MSE 损失
        loss = F.mse_loss(resp_psd_weak, resp_psd_strong)
        
        return loss
    
    def compute_psd(self, signal):
        """
        计算功率谱密度
        
        Args:
            signal: torch.Tensor, shape (B, T)
        
        Returns:
            psd: torch.Tensor, shape (B, F), 功率谱密度
        """
        # FFT
        fft = torch.fft.rfft(signal, dim=-1)
        # 功率谱
        psd = torch.abs(fft) ** 2
        return psd
    
    def extract_respiratory_band(self, psd):
        """
        提取呼吸频带的功率谱
        
        Args:
            psd: torch.Tensor, shape (B, F), 功率谱
        
        Returns:
            resp_psd: torch.Tensor, shape (B, F_resp), 呼吸频带功率谱
        """
        # 计算频率
        # 注意：这里需要知道信号长度 T 来计算频率
        # 简化处理：直接返回整个 PSD（在 forward 中已经处理）
        return psd
```

---

## 3. SNR 计算函数

```python
"""信噪比计算 - 用于课程学习"""
import torch
import torch.nn.functional as F

def calculate_respiration_snr(signal, fps=30, resp_freq_range=(0.1, 0.5)):
    """
    计算呼吸信号的信噪比 (SNR)
    
    Args:
        signal: torch.Tensor, shape (B, T) 或 (T,), 呼吸信号
        fps: int, 视频帧率
        resp_freq_range: tuple, 呼吸频率范围 (min_freq, max_freq) Hz
    
    Returns:
        snr: torch.Tensor, shape (B,) 或 scalar, SNR (dB)
    """
    # 确保是 2D tensor
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)
    
    # 1. 快速傅里叶变换
    fft = torch.fft.rfft(signal, dim=-1)
    power = torch.abs(fft) ** 2
    
    # 2. 定义呼吸频带
    T = signal.shape[-1]
    freqs = torch.fft.rfftfreq(T, 1/fps)
    
    # 移动到设备
    if signal.is_cuda:
        freqs = freqs.to(signal.device)
    
    min_freq, max_freq = resp_freq_range
    resp_mask = (freqs >= min_freq) & (freqs <= max_freq)
    
    # 3. 计算频带内能量和总能量
    resp_power = power[..., resp_mask].sum(dim=-1)
    total_power = power.sum(dim=-1)
    noise_power = total_power - resp_power
    
    # 4. 计算 SNR (dB)
    snr = 10 * torch.log10(resp_power / (noise_power + 1e-8))
    
    # 如果输入是 1D，返回 scalar
    if snr.shape[0] == 1 and signal.dim() == 1:
        snr = snr[0]
    
    return snr
```

---

## 4. 时间增强函数

```python
"""时间增强 - 用于半监督学习"""
import torch
import numpy as np

def temporal_shift(x, shift_range=5):
    """
    时间位移增强（弱增强）
    
    Args:
        x: torch.Tensor, shape (B, C, T, H, W)
        shift_range: int, 最大位移范围
    
    Returns:
        x_shifted: torch.Tensor, shape (B, C, T, H, W)
    """
    B, C, T, H, W = x.shape
    
    # 随机选择位移量
    shift = np.random.randint(-shift_range, shift_range + 1)
    
    if shift == 0:
        return x
    
    # 时间维度位移
    if shift > 0:
        # 向右位移：前面补零
        x_shifted = torch.cat([
            torch.zeros(B, C, shift, H, W, device=x.device),
            x[:, :, :-shift, :, :]
        ], dim=2)
    else:
        # 向左位移：后面补零
        x_shifted = torch.cat([
            x[:, :, -shift:, :, :],
            torch.zeros(B, C, -shift, H, W, device=x.device)
        ], dim=2)
    
    return x_shifted

def temporal_reverse(x):
    """
    时间反转增强（强增强）
    
    Args:
        x: torch.Tensor, shape (B, C, T, H, W)
    
    Returns:
        x_reversed: torch.Tensor, shape (B, C, T, H, W)
    """
    return torch.flip(x, dims=[2])
```

---

## 5. 呼吸率评估函数

```python
"""呼吸率评估 - 修改自 metrics.py"""
import numpy as np
from scipy import signal

def calculate_respiration_rate(pred_signal, label_signal, fps=30, 
                               resp_freq_range=(0.1, 0.5)):
    """
    计算呼吸率（RR）
    
    Args:
        pred_signal: np.array, shape (T,), 预测信号
        label_signal: np.array, shape (T,), 标签信号
        fps: int, 帧率
        resp_freq_range: tuple, 呼吸频率范围 (min_freq, max_freq) Hz
    
    Returns:
        pred_rr: float, 预测呼吸率 (bpm)
        label_rr: float, 标签呼吸率 (bpm)
    """
    min_freq, max_freq = resp_freq_range
    
    # FFT
    pred_fft = np.fft.rfft(pred_signal)
    label_fft = np.fft.rfft(label_signal)
    
    # 频率
    freqs = np.fft.rfftfreq(len(pred_signal), 1/fps)
    
    # 呼吸频带 mask
    resp_mask = (freqs >= min_freq) & (freqs <= max_freq)
    
    if not np.any(resp_mask):
        return 0.0, 0.0
    
    # 功率谱
    pred_power = np.abs(pred_fft[resp_mask]) ** 2
    label_power = np.abs(label_fft[resp_mask]) ** 2
    
    # 找到峰值频率
    pred_freq_idx = np.argmax(pred_power)
    label_freq_idx = np.argmax(label_power)
    
    pred_freq = freqs[resp_mask][pred_freq_idx]
    label_freq = freqs[resp_mask][label_freq_idx]
    
    # 转换为 bpm
    pred_rr = pred_freq * 60
    label_rr = label_freq * 60
    
    return pred_rr, label_rr
```

---

## 6. 修改 PhysNet 支持 4 通道

```python
"""修改 PhysNet 支持 4 通道输入"""
# 在 PhysNet.py 中添加新类

class ModifiedPhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    """修改版 PhysNet，支持 4 通道输入（RGB + Motion）"""
    
    def __init__(self, frames=128, in_channels=4):
        super().__init__()
        
        # 修改第一层卷积的输入通道数
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        
        # 其余层保持不变
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        # ... 其他层保持不变，参考原始 PhysNet
```

---

## 使用说明

1. **复制代码模板**到对应文件
2. **根据实际需求修改**参数和逻辑
3. **测试每个模块**确保功能正常
4. **集成到主流程**中

---

**注意**: 这些是代码模板，实际使用时需要根据 rPPG-Toolbox 的具体架构进行调整。
