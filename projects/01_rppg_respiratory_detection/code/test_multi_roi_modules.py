"""多ROI呼吸检测模块集成测试

测试课题核心创新模块：
1. MultiRegionROIExtractor - 6区域ROI提取 + 两阶段预处理
2. SS-FEM - 自分离特征增强
3. MultiROIFusion - 多ROI自适应融合
4. RespirationNet - 完整呼吸检测网络（含MotionGuidedSTLAM + 混合输入）
5. ROIConsistencyLoss - 跨ROI一致性损失（含NaN稳定性测试）
6. ROIVisualizer - ROI可视化

运行方式:
    python test_multi_roi_modules.py

注意: 需要安装依赖: pip install torch numpy mediapipe opencv-python matplotlib
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_ss_fem():
    """测试SS-FEM模块"""
    print("\n" + "="*60)
    print("测试 SS-FEM 自分离特征增强模块")
    print("="*60)
    
    try:
        from neural_methods.model.modules.SS_FEM import SS_FEM, TemporalSS_FEM
        
        # 测试2D SS-FEM
        print("\n1. 测试 2D SS-FEM:")
        ss_fem_2d = SS_FEM(channels=64, dilations=[1, 2, 4, 8], use_3d=False)
        x_2d = torch.randn(2, 64, 32, 32)
        out_2d = ss_fem_2d(x_2d)
        print(f"   输入: {x_2d.shape} -> 输出: {out_2d.shape}")
        assert out_2d.shape == x_2d.shape, "输出形状不匹配"
        print("   [PASS] 2D SS-FEM 测试通过")
        
        # 测试3D SS-FEM
        print("\n2. 测试 3D SS-FEM:")
        ss_fem_3d = SS_FEM(channels=64, dilations=[1, 2, 4], use_3d=True)
        x_3d = torch.randn(2, 64, 32, 16, 16)
        out_3d = ss_fem_3d(x_3d)
        print(f"   输入: {x_3d.shape} -> 输出: {out_3d.shape}")
        assert out_3d.shape == x_3d.shape, "输出形状不匹配"
        print("   [PASS] 3D SS-FEM 测试通过")
        
        # 测试时序SS-FEM
        print("\n3. 测试 Temporal SS-FEM:")
        temporal_ss_fem = TemporalSS_FEM(channels=64)
        x_temporal = torch.randn(2, 64, 32, 16, 16)
        out_temporal = temporal_ss_fem(x_temporal)
        print(f"   输入: {x_temporal.shape} -> 输出: {out_temporal.shape}")
        assert out_temporal.shape == x_temporal.shape, "输出形状不匹配"
        print("   [PASS] Temporal SS-FEM 测试通过")
        
        print("\n[PASS] SS-FEM 模块测试全部通过!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] SS-FEM 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_roi_fusion():
    """测试多ROI融合模块"""
    print("\n" + "="*60)
    print("测试 MultiROIFusion 多ROI自适应融合模块")
    print("="*60)
    
    try:
        from neural_methods.model.modules.MultiROIFusion import (
            MultiROIFusion, FactorizedAttention, SNREstimator
        )
        
        batch_size = 4
        num_rois = 6
        signal_len = 160
        
        # 测试FactorizedAttention
        print("\n1. 测试 FactorizedAttention:")
        fa = FactorizedAttention(dim=64, rank=4, num_heads=1)
        x_fa = torch.randn(batch_size, num_rois, 64)
        out_fa, attn = fa(x_fa)
        print(f"   输入: {x_fa.shape} -> 输出: {out_fa.shape}, 注意力: {attn.shape}")
        assert out_fa.shape == x_fa.shape, "输出形状不匹配"
        print("   [PASS] FactorizedAttention 测试通过")
        
        # 测试SNREstimator
        print("\n2. 测试 SNREstimator:")
        snr_est = SNREstimator(signal_len=signal_len)
        signal = torch.randn(batch_size, signal_len)
        snr_score = snr_est(signal)
        print(f"   输入: {signal.shape} -> SNR分数: {snr_score.shape}")
        assert snr_score.shape == (batch_size, 1), "SNR分数形状不匹配"
        assert (snr_score >= 0).all() and (snr_score <= 1).all(), "SNR分数应在[0,1]范围内"
        print("   [PASS] SNREstimator 测试通过")
        
        # 测试MultiROIFusion
        print("\n3. 测试 MultiROIFusion:")
        fusion = MultiROIFusion(
            num_rois=num_rois,
            signal_len=signal_len,
            feature_dim=64,
            use_snr_weighting=True
        )
        
        # 创建字典输入
        roi_signals = {
            'forehead': torch.randn(batch_size, signal_len),
            'left_cheek': torch.randn(batch_size, signal_len),
            'right_cheek': torch.randn(batch_size, signal_len),
            'neck_left': torch.randn(batch_size, signal_len),
            'neck_right': torch.randn(batch_size, signal_len),
            'chest': torch.randn(batch_size, signal_len),
        }
        
        fused_signal, weights, attention = fusion(roi_signals, return_weights=True)
        print(f"   输入: 6个ROI信号 (B={batch_size}, T={signal_len})")
        print(f"   输出: 融合信号 {fused_signal.shape}, 权重 {weights.shape}")
        assert fused_signal.shape == (batch_size, signal_len), "融合信号形状不匹配"
        print("   [PASS] MultiROIFusion 测试通过")
        
        # 测试一致性损失
        print("\n4. 测试 FrequencyConsistencyLoss:")
        consistency_loss = fusion.consistency_loss(roi_signals)
        print(f"   一致性损失: {consistency_loss.item():.4f}")
        assert consistency_loss.item() >= 0, "损失应为非负"
        print("   [PASS] FrequencyConsistencyLoss 测试通过")
        
        print("\n[PASS] MultiROIFusion 模块测试全部通过!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] MultiROIFusion 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_roi_consistency_loss():
    """测试ROI一致性损失"""
    print("\n" + "="*60)
    print("测试 ROIConsistencyLoss 跨ROI一致性损失")
    print("="*60)
    
    try:
        from neural_methods.model.modules.ROIConsistencyLoss import (
            ROIConsistencyLoss, TemporalConsistencyLoss, 
            FrequencyDomainLoss, RespirationLossBundle
        )
        
        batch_size = 4
        num_rois = 6
        signal_len = 160
        
        # 创建测试信号（相似的信号加噪声）
        base_signal = torch.sin(torch.linspace(0, 4*np.pi, signal_len)).unsqueeze(0).repeat(batch_size, 1)
        roi_signals = torch.stack([
            base_signal + 0.1 * torch.randn_like(base_signal)
            for _ in range(num_rois)
        ], dim=1)  # (B, N, T)
        
        # 测试时域一致性
        print("\n1. 测试 TemporalConsistencyLoss:")
        temporal_loss = TemporalConsistencyLoss()
        loss_t = temporal_loss(roi_signals)
        print(f"   时域一致性损失: {loss_t.item():.4f}")
        assert loss_t.item() >= 0, "损失应为非负"
        print("   [PASS] 测试通过")
        
        # 测试频域一致性
        print("\n2. 测试 FrequencyDomainLoss:")
        freq_loss = FrequencyDomainLoss(fs=30.0)
        loss_f = freq_loss(roi_signals)
        print(f"   频域一致性损失: {loss_f.item():.4f}")
        assert loss_f.item() >= 0, "损失应为非负"
        print("   [PASS] 测试通过")
        
        # 测试综合ROI一致性损失
        print("\n3. 测试 ROIConsistencyLoss:")
        roi_loss = ROIConsistencyLoss(temporal_weight=1.0, frequency_weight=1.0)
        loss_total, components = roi_loss(roi_signals, return_components=True)
        print(f"   总损失: {loss_total.item():.4f}")
        print(f"   - 时域: {components['temporal']:.4f}")
        print(f"   - 频域: {components['frequency']:.4f}")
        print(f"   - 相位: {components['phase']:.4f}")
        print("   [PASS] 测试通过")
        
        # 测试损失函数包
        print("\n4. 测试 RespirationLossBundle:")
        loss_bundle = RespirationLossBundle()
        pred_signal = torch.randn(batch_size, signal_len)
        target_signal = torch.randn(batch_size, signal_len)
        
        total_loss, loss_dict = loss_bundle(
            pred_signal=pred_signal,
            target_signal=target_signal,
            roi_signals={'roi_' + str(i): roi_signals[:, i] for i in range(num_rois)}
        )
        print(f"   总损失: {loss_dict['total']:.4f}")
        print(f"   - 重建损失: {loss_dict.get('recon', 'N/A')}")
        print(f"   - 一致性损失: {loss_dict.get('consistency', 'N/A')}")
        print("   [PASS] 测试通过")
        
        # 测试NaN稳定性 - 常数信号
        print("\n5. 测试NaN稳定性 (常数信号):")
        constant_signals = torch.ones(batch_size, num_rois, signal_len) * 5.0
        loss_const_t = temporal_loss(constant_signals)
        loss_const_f = freq_loss(constant_signals)
        print(f"   常数信号时域损失: {loss_const_t.item():.4f} (NaN: {torch.isnan(loss_const_t).item()})")
        print(f"   常数信号频域损失: {loss_const_f.item():.4f} (NaN: {torch.isnan(loss_const_f).item()})")
        assert not torch.isnan(loss_const_t), "常数信号不应产生NaN时域损失"
        assert not torch.isnan(loss_const_f), "常数信号不应产生NaN频域损失"
        print("   [PASS] NaN稳定性测试通过")
        
        # 测试NaN稳定性 - 部分常数信号
        print("\n6. 测试NaN稳定性 (部分常数 + 部分正常信号):")
        mixed_signals = torch.zeros(batch_size, num_rois, signal_len)
        mixed_signals[:, 0, :] = torch.sin(torch.linspace(0, 4*np.pi, signal_len))  # 正常
        mixed_signals[:, 1, :] = 3.0  # 常数
        mixed_signals[:, 2, :] = torch.cos(torch.linspace(0, 4*np.pi, signal_len))  # 正常
        mixed_signals[:, 3:, :] = 0.0  # 常数（全零）
        
        loss_mixed_t = temporal_loss(mixed_signals)
        loss_mixed_f = freq_loss(mixed_signals)
        print(f"   混合信号时域损失: {loss_mixed_t.item():.4f} (NaN: {torch.isnan(loss_mixed_t).item()})")
        print(f"   混合信号频域损失: {loss_mixed_f.item():.4f} (NaN: {torch.isnan(loss_mixed_f).item()})")
        assert not torch.isnan(loss_mixed_t), "混合信号不应产生NaN时域损失"
        assert not torch.isnan(loss_mixed_f), "混合信号不应产生NaN频域损失"
        print("   [PASS] 部分常数信号NaN稳定性测试通过")
        
        # 测试NaN稳定性 - 极小值信号
        print("\n7. 测试NaN稳定性 (极小值信号):")
        tiny_signals = torch.randn(batch_size, num_rois, signal_len) * 1e-10
        loss_tiny_t = temporal_loss(tiny_signals)
        loss_tiny_f = freq_loss(tiny_signals)
        print(f"   极小值时域损失: {loss_tiny_t.item():.6f} (NaN: {torch.isnan(loss_tiny_t).item()})")
        print(f"   极小值频域损失: {loss_tiny_f.item():.6f} (NaN: {torch.isnan(loss_tiny_f).item()})")
        assert not torch.isnan(loss_tiny_t), "极小值信号不应产生NaN"
        assert not torch.isnan(loss_tiny_f), "极小值信号不应产生NaN"
        print("   [PASS] 极小值NaN稳定性测试通过")
        
        print("\n[PASS] ROIConsistencyLoss 模块测试全部通过 (含NaN稳定性)!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] ROIConsistencyLoss 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_respiration_net():
    """测试完整的RespirationNet（含MotionGuidedSTLAM和混合输入）"""
    print("\n" + "="*60)
    print("测试 RespirationNet 完整呼吸检测网络")
    print("="*60)
    
    try:
        from neural_methods.model.RespirationNet import (
            RespirationNet, RespirationNetLite, SingleROIEncoder,
            MotionGuidedSTLAM
        )
        
        batch_size = 2
        num_rois = 6
        frames = 160
        height, width = 64, 64
        
        # 测试SingleROIEncoder
        print("\n1. 测试 SingleROIEncoder:")
        encoder = SingleROIEncoder(in_channels=3, base_channels=16, frames=frames)
        x_single = torch.randn(batch_size, 3, frames, height, width)
        signal_single = encoder(x_single)
        print(f"   输入: {x_single.shape} -> 输出: {signal_single.shape}")
        assert signal_single.shape == (batch_size, frames), "信号形状不匹配"
        
        params = sum(p.numel() for p in encoder.parameters())
        print(f"   参数量: {params:,}")
        print("   [PASS] SingleROIEncoder 测试通过")
        
        # 测试2通道光流编码器
        print("\n1b. 测试 SingleROIEncoder (2通道光流):")
        flow_encoder = SingleROIEncoder(in_channels=2, base_channels=16, frames=frames)
        x_flow = torch.randn(batch_size, 2, frames, height, width)
        signal_flow = flow_encoder(x_flow)
        print(f"   输入: {x_flow.shape} -> 输出: {signal_flow.shape}")
        assert signal_flow.shape == (batch_size, frames), "光流信号形状不匹配"
        print("   [PASS] 2通道光流编码器测试通过")
        
        # 测试MotionGuidedSTLAM
        print("\n1c. 测试 MotionGuidedSTLAM:")
        stlam = MotionGuidedSTLAM(signal_dim=frames)
        rppg_signal = torch.randn(batch_size, frames)
        motion_signal = torch.randn(batch_size, frames)
        enhanced, attn = stlam(rppg_signal, motion_signal)
        print(f"   rPPG输入: {rppg_signal.shape}, 运动输入: {motion_signal.shape}")
        print(f"   增强输出: {enhanced.shape}, 注意力: {attn.shape}")
        assert enhanced.shape == (batch_size, frames), "STLAM输出形状不匹配"
        assert attn.shape == (batch_size, frames), "STLAM注意力形状不匹配"
        print("   [PASS] MotionGuidedSTLAM 测试通过")
        
        # 测试RespirationNetLite
        print("\n2. 测试 RespirationNetLite:")
        net_lite = RespirationNetLite(frames=frames)
        x_lite = torch.randn(batch_size, 3, frames, height, width)
        signal_lite = net_lite(x_lite)
        print(f"   输入: {x_lite.shape} -> 输出: {signal_lite.shape}")
        assert signal_lite.shape == (batch_size, frames), "信号形状不匹配"
        
        params = sum(p.numel() for p in net_lite.parameters())
        print(f"   参数量: {params:,}")
        print("   [PASS] RespirationNetLite 测试通过")
        
        # 测试完整RespirationNet（混合输入模式）
        print("\n3. 测试 RespirationNet (多ROI + 混合输入):")
        net_full = RespirationNet(
            num_rois=num_rois,
            frames=frames,
            base_channels=16,
            chest_flow_channels=2,
            use_dual_branch=False,
            share_encoder=True
        )
        
        # 显示模型结构信息
        print(f"   RGB ROI: {net_full.RGB_ROI_NAMES}")
        print(f"   Flow ROI: {net_full.FLOW_ROI_NAME}")
        print(f"   光流通道数: {net_full.chest_flow_channels}")
        has_stlam = hasattr(net_full, 'motion_stlam')
        print(f"   MotionGuidedSTLAM: {'YES' if has_stlam else 'NO'}")
        
        # 测试dict混合输入（RGB 3通道 + 光流 2通道）
        print("\n4. 测试 dict 混合输入 (RGB 3ch + Flow 2ch):")
        roi_dict_mixed = {
            'forehead': torch.randn(batch_size, 3, frames, height, width),
            'left_cheek': torch.randn(batch_size, 3, frames, height, width),
            'right_cheek': torch.randn(batch_size, 3, frames, height, width),
            'neck_left': torch.randn(batch_size, 3, frames, height, width),
            'neck_right': torch.randn(batch_size, 3, frames, height, width),
            'chest': torch.randn(batch_size, 2, frames, height, width),  # 光流2通道
        }
        
        resp_signal, roi_signals, fusion_weights, attention = net_full(
            roi_dict_mixed, return_intermediate=True
        )
        
        print(f"   输入: dict (5x RGB 3ch + 1x Flow 2ch)")
        print(f"   输出: 呼吸信号 {resp_signal.shape}")
        print(f"   ROI信号数量: {len(roi_signals)}")
        if fusion_weights is not None:
            print(f"   融合权重: {fusion_weights.shape}")
            print(f"   权重分布: {fusion_weights[0].detach().numpy()}")
        assert resp_signal.shape == (batch_size, frames), "呼吸信号形状不匹配"
        print("   [PASS] 混合输入测试通过")
        
        # 测试tensor输入（向后兼容）
        print("\n5. 测试 tensor 输入 (向后兼容):")
        roi_frames_tensor = torch.randn(batch_size, num_rois, 3, frames, height, width)
        resp_signal_tensor = net_full(roi_frames_tensor)
        print(f"   输入: tensor {roi_frames_tensor.shape}")
        print(f"   输出: {resp_signal_tensor.shape}")
        assert resp_signal_tensor.shape == (batch_size, frames), "tensor输入形状不匹配"
        print("   [PASS] tensor 输入测试通过")
        
        # 测试反向传播
        print("\n6. 测试反向传播 (混合输入):")
        net_full.zero_grad()
        resp_out = net_full(roi_dict_mixed)
        target = torch.randn(batch_size, frames)
        loss = nn.MSELoss()(resp_out, target)
        loss.backward()
        print(f"   损失: {loss.item():.4f}")
        
        # 检查各编码器梯度
        rgb_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                         for p in net_full.rgb_encoder.parameters())
        flow_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                          for p in net_full.flow_encoder.parameters())
        stlam_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                           for p in net_full.motion_stlam.parameters()) if has_stlam else False
        
        print(f"   RGB编码器梯度: {rgb_has_grad}")
        print(f"   光流编码器梯度: {flow_has_grad}")
        print(f"   STLAM梯度: {stlam_has_grad}")
        assert rgb_has_grad, "RGB编码器应有梯度"
        assert flow_has_grad, "光流编码器应有梯度"
        print("   [PASS] 反向传播测试通过")
        
        params = sum(p.numel() for p in net_full.parameters())
        trainable = sum(p.numel() for p in net_full.parameters() if p.requires_grad)
        print(f"\n   总参数: {params:,}, 可训练: {trainable:,}")
        
        print("\n[PASS] RespirationNet 模块测试全部通过!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] RespirationNet 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_region_roi_extractor():
    """测试多区域ROI提取器"""
    print("\n" + "="*60)
    print("测试 MultiRegionROIExtractor 多区域ROI提取器")
    print("="*60)
    
    try:
        from dataset.respiratory_utils import MultiRegionROIExtractor
        
        print("\n1. 初始化 MultiRegionROIExtractor:")
        extractor = MultiRegionROIExtractor(
            roi_size=(64, 64),
            use_optical_flow=False  # 单帧测试时关闭光流
        )
        print("   [PASS] 初始化成功")
        
        # 创建测试帧
        print("\n2. 测试单帧ROI提取:")
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        rois, roi_boxes, success = extractor.extract(test_frame)
        
        print(f"   检测到的ROI:")
        for roi_name, roi_data in rois.items():
            box = roi_boxes.get(roi_name, 'N/A')
            status = '[PASS]' if success.get(roi_name, False) else '[FALLBACK] (fallback)'
            print(f"   - {roi_name}: {roi_data.shape} {status}")
        
        # 验证ROI大小
        for roi_name, roi_data in rois.items():
            assert roi_data.shape[:2] == (64, 64), f"{roi_name} ROI大小不正确"
        print("   [PASS] 单帧提取测试通过")
        
        # 测试批量提取
        print("\n3. 测试批量ROI提取:")
        batch_frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        batch_rois = extractor.extract_batch(batch_frames)
        print(f"   批量帧数: {len(batch_frames)}")
        print(f"   提取结果:")
        for roi_name, roi_array in batch_rois.items():
            print(f"   - {roi_name}: {roi_array.shape}")
        print("   [PASS] 批量提取测试通过")
        
        # 打印统计
        print("\n4. ROI检测统计:")
        print(f"   {extractor.roi_stats}")
        
        # 清理
        extractor.close()
        print("\n[PASS] MultiRegionROIExtractor 测试全部通过!")
        return True
        
    except ImportError as e:
        print(f"\n[WARN] 跳过测试 (依赖未安装): {e}")
        return True  # 不算失败，只是跳过
    except Exception as e:
        print(f"\n[FAIL] MultiRegionROIExtractor 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_two_stage_preprocess():
    """测试两阶段预处理方法"""
    print("\n" + "="*60)
    print("测试 两阶段预处理 (Stage1: 胸部光流, Stage2: 面部+颈部)")
    print("="*60)
    
    try:
        from dataset.respiratory_utils import MultiRegionROIExtractor
        
        print("\n1. 初始化 MultiRegionROIExtractor:")
        extractor = MultiRegionROIExtractor(
            roi_size=(64, 64),
            use_optical_flow=True
        )
        print("   [PASS] 初始化成功")
        
        # Stage 1: 胸部光流提取
        print("\n2. 测试 Stage 1 - 胸部光流提取:")
        test_frame_1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_frame_2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 第一帧（需要prev_frame初始化）
        flow_1 = extractor.extract_chest_optical_flow(test_frame_1)
        print(f"   第一帧光流: shape={flow_1.shape}, type={flow_1.dtype}")
        assert flow_1.shape == (64, 64, 2), f"光流形状应为(64,64,2)，实际{flow_1.shape}"
        
        # 第二帧（应有实际光流）
        flow_2 = extractor.extract_chest_optical_flow(test_frame_2)
        print(f"   第二帧光流: shape={flow_2.shape}, range=[{flow_2.min():.4f}, {flow_2.max():.4f}]")
        assert flow_2.shape == (64, 64, 2), f"光流形状应为(64,64,2)"
        assert flow_2.min() >= -1.0 and flow_2.max() <= 1.0, "光流应归一化到[-1,1]"
        print("   [PASS] 胸部光流提取通过 (双通道dx,dy)")
        
        # Stage 2: 面部+颈部ROI
        print("\n3. 测试 Stage 2 - 面部ROI提取 (裁剪帧):")
        cropped_frame = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        face_rois = extractor.extract_face_rois(cropped_frame)
        print(f"   提取到的面部ROI: {list(face_rois.keys())}")
        for name, roi in face_rois.items():
            print(f"   - {name}: shape={roi.shape}")
            assert roi.shape[:2] == (64, 64), f"{name} ROI大小不正确"
        print("   [PASS] 面部ROI提取通过")
        
        print("\n4. 测试 Stage 2 - 颈部ROI提取 (裁剪帧):")
        neck_rois = extractor.extract_neck_rois(cropped_frame)
        print(f"   提取到的颈部ROI: {list(neck_rois.keys())}")
        for name, roi in neck_rois.items():
            print(f"   - {name}: shape={roi.shape}")
            assert roi.shape[:2] == (64, 64), f"{name} ROI大小不正确"
        print("   [PASS] 颈部ROI提取通过")
        
        # 测试重置光流状态
        print("\n5. 测试 reset_flow_state:")
        extractor.reset_flow_state()
        assert extractor.prev_frame is None, "重置后prev_frame应为None"
        print("   [PASS] 光流状态重置通过")
        
        extractor.close()
        print("\n[PASS] 两阶段预处理测试全部通过!")
        return True
        
    except ImportError as e:
        print(f"\n[WARN] 跳过测试 (依赖未安装): {e}")
        return True
    except Exception as e:
        print(f"\n[FAIL] 两阶段预处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_roi_visualizer():
    """测试ROI可视化模块"""
    print("\n" + "="*60)
    print("测试 ROIVisualizer ROI可视化模块")
    print("="*60)
    
    try:
        from visualization.roi_visualizer import ROIVisualizer
        
        print("\n1. 初始化 ROIVisualizer:")
        visualizer = ROIVisualizer()
        print("   [PASS] 初始化成功")
        
        # 测试ROI框可视化
        print("\n2. 测试 ROI框可视化:")
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        roi_boxes = {
            'forehead': [200, 50, 80, 40],
            'left_cheek': [160, 120, 60, 60],
            'right_cheek': [280, 120, 60, 60],
            'neck_left': [170, 200, 50, 70],
            'neck_right': [280, 200, 50, 70],
            'chest': [180, 300, 160, 100],
        }
        success = {k: True for k in roi_boxes}
        success['chest'] = True
        
        vis_frame = visualizer.visualize_rois(test_frame, roi_boxes, success)
        print(f"   输出帧形状: {vis_frame.shape}")
        assert vis_frame.shape == test_frame.shape, "可视化帧形状应与输入一致"
        print("   [PASS] ROI框可视化通过")
        
        # 测试注意力权重可视化
        print("\n3. 测试 注意力权重可视化:")
        try:
            import matplotlib
            weights = np.array([0.20, 0.15, 0.15, 0.18, 0.17, 0.15])
            roi_names = ['forehead', 'left_cheek', 'right_cheek', 'neck_left', 'neck_right', 'chest']
            fig = visualizer.visualize_attention_weights(weights, roi_names)
            print(f"   图表对象: {type(fig)}")
            print("   [PASS] 注意力权重可视化通过")
        except ImportError:
            print("   [WARN] matplotlib未安装，跳过注意力可视化")
        
        # 测试torch tensor输入
        print("\n4. 测试 torch tensor 注意力权重:")
        try:
            import matplotlib
            weights_tensor = torch.tensor([[0.20, 0.15, 0.15, 0.18, 0.17, 0.15],
                                          [0.22, 0.14, 0.13, 0.19, 0.18, 0.14]])
            fig = visualizer.visualize_attention_weights(weights_tensor, roi_names)
            print("   [PASS] tensor注意力权重可视化通过")
        except ImportError:
            print("   [WARN] matplotlib未安装，跳过")
        
        print("\n[PASS] ROIVisualizer 模块测试全部通过!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] ROIVisualizer 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("多ROI呼吸检测模块集成测试")
    print("="*60)
    
    results = {}
    
    # 运行各模块测试
    results['SS-FEM'] = test_ss_fem()
    results['MultiROIFusion'] = test_multi_roi_fusion()
    results['ROIConsistencyLoss (+ NaN)'] = test_roi_consistency_loss()
    results['RespirationNet (+ STLAM + Mixed)'] = test_respiration_net()
    results['MultiRegionROIExtractor'] = test_multi_region_roi_extractor()
    results['TwoStagePreprocess'] = test_two_stage_preprocess()
    results['ROIVisualizer'] = test_roi_visualizer()
    
    # 打印总结
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "[PASS] 通过" if passed else "[FAIL] 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("所有测试通过! 多ROI呼吸检测模块已就绪。")
    else:
        print("部分测试失败，请检查错误信息。")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
