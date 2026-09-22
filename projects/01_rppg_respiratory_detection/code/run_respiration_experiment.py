"""
rPPG 远程呼吸检测实验一键运行脚本

本脚本实现混合训练策略：
1. 阶段1: 在 PURE 数据集上进行监督预训练
2. 阶段2: 在 UBFC-rPPG 上进行半监督微调

使用方法:
    python run_respiration_experiment.py --stage all
    python run_respiration_experiment.py --stage pretrain
    python run_respiration_experiment.py --stage finetune
    python run_respiration_experiment.py --stage test

依赖:
    - PyTorch
    - OpenCV
    - NumPy, SciPy
    - tqdm
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime


def print_header(title):
    """打印格式化的标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def run_command(cmd, description):
    """运行命令并打印状态"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"  Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] {description} completed.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {description} failed with exit code {e.returncode}\n")
        return False


def check_config_paths(config_path):
    """检查配置文件中的数据路径是否存在"""
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    paths_to_check = []
    
    # 收集所有数据路径
    for section in ['TRAIN', 'VALID', 'TEST', 'TRAIN_UNLABELED']:
        if section in config and 'DATA' in config[section]:
            data_path = config[section]['DATA'].get('DATA_PATH', '')
            if data_path:
                paths_to_check.append((section, data_path))
    
    # 检查路径
    missing = []
    for section, path in paths_to_check:
        if not os.path.exists(path):
            missing.append((section, path))
    
    if missing:
        print("\n[WARNING] The following data paths do not exist:")
        for section, path in missing:
            print(f"  - {section}: {path}")
        print("\nPlease update the paths in the configuration file before running.\n")
        return False
    
    return True


def stage_pretrain(args):
    """阶段1: 监督预训练"""
    print_header("Stage 1: Supervised Pre-training on PURE Dataset")
    
    config_file = args.pretrain_config
    if not os.path.exists(config_file):
        print(f"[ERROR] Config file not found: {config_file}")
        return False
    
    print(f"Using config: {config_file}")
    
    # 检查路径
    if not args.skip_path_check:
        if not check_config_paths(config_file):
            if not args.force:
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    return False
    
    cmd = [
        sys.executable, 'main.py',
        '--config_file', config_file
    ]
    
    return run_command(cmd, "Pre-training on PURE dataset")


def stage_finetune(args):
    """阶段2: 半监督微调"""
    print_header("Stage 2: Semi-Supervised Fine-tuning on UBFC-rPPG")
    
    config_file = args.finetune_config
    if not os.path.exists(config_file):
        print(f"[ERROR] Config file not found: {config_file}")
        return False
    
    print(f"Using config: {config_file}")
    
    # 如果指定了预训练模型路径，更新配置
    if args.pretrain_model:
        print(f"Using pretrained model: {args.pretrain_model}")
        # 这里可以动态修改配置文件，但为简单起见，假设用户已经在配置文件中设置
    
    # 检查路径
    if not args.skip_path_check:
        if not check_config_paths(config_file):
            if not args.force:
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    return False
    
    cmd = [
        sys.executable, 'main.py',
        '--config_file', config_file
    ]
    
    return run_command(cmd, "Semi-supervised fine-tuning on UBFC-rPPG")


def stage_test(args):
    """仅测试"""
    print_header("Testing Only")
    
    config_file = args.test_config or args.finetune_config
    if not os.path.exists(config_file):
        print(f"[ERROR] Config file not found: {config_file}")
        return False
    
    print(f"Using config: {config_file}")
    
    # 需要在配置文件中设置 TOOLBOX_MODE: only_test
    # 或者创建一个专门的测试配置文件
    
    cmd = [
        sys.executable, 'main.py',
        '--config_file', config_file
    ]
    
    return run_command(cmd, "Testing")


def main():
    parser = argparse.ArgumentParser(
        description='rPPG Respiration Detection Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment (pretrain + finetune)
  python run_respiration_experiment.py --stage all

  # Run only pre-training
  python run_respiration_experiment.py --stage pretrain

  # Run only fine-tuning (assumes pretrained model exists)
  python run_respiration_experiment.py --stage finetune

  # Run with custom config paths
  python run_respiration_experiment.py --stage all \\
      --pretrain-config configs/train_configs/PRETRAIN_PURE_RESPIRATION.yaml \\
      --finetune-config configs/train_configs/SEMI_SUPERVISED_UBFC_RESPIRATION.yaml
        """
    )
    
    parser.add_argument(
        '--stage',
        choices=['all', 'pretrain', 'finetune', 'test'],
        default='all',
        help='Which stage to run (default: all)'
    )
    
    parser.add_argument(
        '--pretrain-config',
        default='configs/train_configs/PRETRAIN_PURE_RESPIRATION.yaml',
        help='Path to pre-training config file'
    )
    
    parser.add_argument(
        '--finetune-config',
        default='configs/train_configs/SEMI_SUPERVISED_UBFC_RESPIRATION.yaml',
        help='Path to fine-tuning config file'
    )
    
    parser.add_argument(
        '--test-config',
        default=None,
        help='Path to test config file (defaults to finetune config)'
    )
    
    parser.add_argument(
        '--pretrain-model',
        default=None,
        help='Path to pretrained model for fine-tuning'
    )
    
    parser.add_argument(
        '--skip-path-check',
        action='store_true',
        help='Skip checking if data paths exist'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force continue even if paths are missing'
    )
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("\n" + "=" * 70)
    print("  rPPG Remote Respiration Detection Experiment")
    print("  Based on: Light Flow Guided Multi-ROI + Semi-Supervised Learning")
    print("=" * 70)
    print(f"\nSelected stage: {args.stage}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    success = True
    
    if args.stage in ['all', 'pretrain']:
        success = stage_pretrain(args)
        if not success and args.stage == 'all':
            print("[WARNING] Pre-training failed. Continuing to fine-tuning anyway...")
    
    if args.stage in ['all', 'finetune']:
        success = stage_finetune(args) and success
    
    if args.stage == 'test':
        success = stage_test(args)
    
    # 打印总结
    print("\n" + "=" * 70)
    if success:
        print("  Experiment completed successfully!")
    else:
        print("  Experiment completed with some errors.")
    print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
