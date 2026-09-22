"""
rPPG 呼吸检测完整实验执行脚本

分阶段执行实验：
- 阶段0: COHFACE创新方法验证（全监督 + 多ROI融合）
- 阶段1: Baseline实验 (B2, B3, B4)
- 阶段2: 主方法实验 (M1, M2)
- 阶段3: 消融实验

使用方法:
    python run_experiments.py --phase 0          # 运行阶段0
    python run_experiments.py --phase 1          # 运行阶段1
    python run_experiments.py --phase all        # 运行所有阶段
    python run_experiments.py --list             # 列出所有实验
    python run_experiments.py --check            # 检查配置文件路径
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime

# 实验配置定义
EXPERIMENTS = {
    # 阶段0: COHFACE创新方法验证
    "phase0": {
        "name": "阶段0: COHFACE创新方法验证",
        "experiments": [
            {
                "id": "PHASE0_MULTIROI",
                "name": "COHFACE多ROI全监督 (创新方法)",
                "config": "configs/train_configs/COHFACE_MULTIROI_SUPERVISED.yaml",
                "description": "验证课题创新点：RespirationNet + 多ROI融合 + 一致性损失"
            },
            {
                "id": "PHASE0_BASELINE",
                "name": "COHFACE PhysNet基线",
                "config": "configs/train_configs/COHFACE_COHFACE_COHFACE_PHYSNET_RESPIRATION.yaml",
                "description": "对比基线：标准PhysNet"
            }
        ]
    },
    
    # 阶段1: Baseline实验
    "phase1": {
        "name": "阶段1: Baseline实验",
        "experiments": [
            {
                "id": "B2",
                "name": "PURE预训练",
                "config": "configs/train_configs/PRETRAIN_PURE_RESPIRATION.yaml",
                "description": "在PURE数据集上预训练，测试在COHFACE上的迁移效果"
            },
            {
                "id": "B3",
                "name": "UBFC 100%全监督",
                "config": "configs/train_configs/BASELINE_UBFC_100PERCENT.yaml",
                "description": "使用UBFC全部标签训练，作为全监督基线"
            },
            {
                "id": "B4",
                "name": "UBFC 30%监督",
                "config": "configs/train_configs/BASELINE_UBFC_30PERCENT.yaml",
                "description": "仅使用30%标签，作为部分监督基线"
            }
        ]
    },
    
    # 阶段2: 主方法实验
    "phase2": {
        "name": "阶段2: 主方法实验",
        "experiments": [
            {
                "id": "M1",
                "name": "UBFC半监督 (无预训练)",
                "config": "configs/train_configs/SEMI_SUPERVISED_UBFC.yaml",
                "description": "30%标签 + 70%无标签，验证半监督学习有效性"
            },
            {
                "id": "M2",
                "name": "PURE预训练 + UBFC半监督",
                "config": "configs/train_configs/SEMI_SUPERVISED_UBFC_PRETRAINED.yaml",
                "description": "完整课题方案：预训练 + 半监督微调"
            }
        ]
    },
    
    # 阶段3: 消融实验
    "phase3": {
        "name": "阶段3: 消融实验",
        "experiments": [
            {
                "id": "A1",
                "name": "仅一致性损失",
                "config": "configs/train_configs/ABLATION_CONSISTENCY_ONLY.yaml",
                "description": "消融：移除伪标签损失"
            },
            {
                "id": "A2",
                "name": "仅伪标签损失",
                "config": "configs/train_configs/ABLATION_PSEUDO_ONLY.yaml",
                "description": "消融：移除一致性损失"
            },
            {
                "id": "A3",
                "name": "10%标签比例",
                "config": "configs/train_configs/ABLATION_LABEL_10PERCENT.yaml",
                "description": "消融：减少标签比例到10%"
            },
            {
                "id": "A4",
                "name": "20%标签比例",
                "config": "configs/train_configs/ABLATION_LABEL_20PERCENT.yaml",
                "description": "消融：减少标签比例到20%"
            },
            {
                "id": "A5",
                "name": "无课程学习",
                "config": "configs/train_configs/ABLATION_NO_CURRICULUM.yaml",
                "description": "消融：移除课程学习策略"
            }
        ]
    }
}


def print_header(title):
    """打印格式化标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def list_experiments():
    """列出所有实验"""
    print_header("rPPG呼吸检测实验列表")
    
    for phase_key, phase_info in EXPERIMENTS.items():
        print(f"\n【{phase_info['name']}】")
        print("-" * 50)
        for exp in phase_info["experiments"]:
            status = "[OK]" if os.path.exists(exp["config"]) else "[MISSING]"
            print(f"  {status} {exp['id']}: {exp['name']}")
            print(f"       配置: {exp['config']}")
            print(f"       说明: {exp['description']}")
        print()


def check_configs():
    """检查所有配置文件"""
    print_header("检查配置文件")
    
    missing = []
    found = []
    
    for phase_key, phase_info in EXPERIMENTS.items():
        for exp in phase_info["experiments"]:
            if os.path.exists(exp["config"]):
                found.append((exp["id"], exp["config"]))
            else:
                missing.append((exp["id"], exp["config"]))
    
    print(f"找到 {len(found)} 个配置文件:")
    for exp_id, config in found:
        print(f"  [OK] {exp_id}: {config}")
    
    if missing:
        print(f"\n缺失 {len(missing)} 个配置文件:")
        for exp_id, config in missing:
            print(f"  [MISSING] {exp_id}: {config}")
        return False
    
    print("\n所有配置文件就绪!")
    return True


def run_experiment(exp_info, dry_run=False):
    """运行单个实验"""
    config = exp_info["config"]
    
    if not os.path.exists(config):
        print(f"[ERROR] 配置文件不存在: {config}")
        return False
    
    print(f"\n{'=' * 60}")
    print(f"实验: {exp_info['id']} - {exp_info['name']}")
    print(f"配置: {config}")
    print(f"说明: {exp_info['description']}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")
    
    cmd = [sys.executable, "main.py", "--config_file", config]
    
    if dry_run:
        print(f"[DRY RUN] 将执行: {' '.join(cmd)}")
        return True
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] 实验 {exp_info['id']} 完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] 实验 {exp_info['id']} 失败，退出码: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] 实验 {exp_info['id']} 被用户中断")
        return False


def run_phase(phase_key, dry_run=False, skip_existing=False):
    """运行指定阶段的所有实验"""
    if phase_key not in EXPERIMENTS:
        print(f"[ERROR] 未知阶段: {phase_key}")
        print(f"可用阶段: {', '.join(EXPERIMENTS.keys())}")
        return False
    
    phase_info = EXPERIMENTS[phase_key]
    print_header(phase_info["name"])
    
    results = []
    for exp in phase_info["experiments"]:
        success = run_experiment(exp, dry_run)
        results.append((exp["id"], success))
        
        if not success and not dry_run:
            print(f"\n[WARNING] 实验 {exp['id']} 失败，继续下一个实验...")
    
    # 打印阶段总结
    print(f"\n{'=' * 60}")
    print(f"{phase_info['name']} 总结")
    print(f"{'=' * 60}")
    for exp_id, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {exp_id}")
    
    return all(success for _, success in results)


def run_all_phases(dry_run=False):
    """运行所有阶段"""
    print_header("运行所有实验阶段")
    
    phase_results = []
    for phase_key in ["phase0", "phase1", "phase2", "phase3"]:
        success = run_phase(phase_key, dry_run)
        phase_results.append((phase_key, success))
    
    # 最终总结
    print_header("实验完成总结")
    for phase_key, success in phase_results:
        status = "[PASS]" if success else "[FAIL]"
        phase_name = EXPERIMENTS[phase_key]["name"]
        print(f"  {status} {phase_name}")
    
    return all(success for _, success in phase_results)


def main():
    parser = argparse.ArgumentParser(
        description="rPPG呼吸检测实验执行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_experiments.py --list              # 列出所有实验
  python run_experiments.py --check             # 检查配置文件
  python run_experiments.py --phase 0           # 运行阶段0
  python run_experiments.py --phase 1           # 运行阶段1
  python run_experiments.py --phase all         # 运行所有阶段
  python run_experiments.py --phase 0 --dry-run # 干运行（不实际执行）
  python run_experiments.py --exp PHASE0_MULTIROI  # 运行指定实验
        """
    )
    
    parser.add_argument("--list", action="store_true", help="列出所有实验")
    parser.add_argument("--check", action="store_true", help="检查配置文件是否存在")
    parser.add_argument("--phase", type=str, help="运行指定阶段 (0, 1, 2, 3, all)")
    parser.add_argument("--exp", type=str, help="运行指定实验ID")
    parser.add_argument("--dry-run", action="store_true", help="干运行，只打印命令不执行")
    
    args = parser.parse_args()
    
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if args.list:
        list_experiments()
        return 0
    
    if args.check:
        success = check_configs()
        return 0 if success else 1
    
    if args.exp:
        # 查找指定实验
        for phase_info in EXPERIMENTS.values():
            for exp in phase_info["experiments"]:
                if exp["id"] == args.exp:
                    success = run_experiment(exp, args.dry_run)
                    return 0 if success else 1
        print(f"[ERROR] 未找到实验: {args.exp}")
        return 1
    
    if args.phase:
        if args.phase.lower() == "all":
            success = run_all_phases(args.dry_run)
        else:
            phase_key = f"phase{args.phase}"
            success = run_phase(phase_key, args.dry_run)
        return 0 if success else 1
    
    # 默认显示帮助
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
