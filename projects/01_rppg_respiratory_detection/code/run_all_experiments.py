#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rPPG 呼吸检测完整实验运行脚本

使用方法:
    python run_all_experiments.py --stage all        # 运行全部实验
    python run_all_experiments.py --stage baseline   # 只运行baseline
    python run_all_experiments.py --stage main       # 只运行主方法
    python run_all_experiments.py --stage ablation   # 只运行消融实验
    python run_all_experiments.py --list             # 列出所有实验配置

注意事项:
    1. 运行前请修改配置文件中的数据路径
    2. 首次运行需要预处理数据 (DO_PREPROCESS: True)
    3. 预处理完成后可设为False加快后续运行
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

# 配置文件路径
CONFIG_DIR = "configs/train_configs"

# 实验配置定义
EXPERIMENTS = {
    "baseline": {
        "B2_PURE_PRETRAIN": {
            "config": "PRETRAIN_PURE_RESPIRATION.yaml",
            "description": "PURE数据集预训练 (Baseline B2)",
            "order": 1
        },
        "B3_UBFC_100PERCENT": {
            "config": "BASELINE_UBFC_100PERCENT.yaml",
            "description": "UBFC-rPPG 100%标签全监督 (Baseline B3)",
            "order": 2
        },
        "B4_UBFC_30PERCENT": {
            "config": "BASELINE_UBFC_30PERCENT.yaml",
            "description": "UBFC-rPPG 30%标签监督 (Baseline B4)",
            "order": 3
        }
    },
    "main": {
        "M1_SEMI_SUPERVISED": {
            "config": "SEMI_SUPERVISED_UBFC.yaml",
            "description": "半监督学习 (方法 M1)",
            "order": 1
        },
        "M2_PRETRAIN_SEMI": {
            "config": "SEMI_SUPERVISED_UBFC_PRETRAINED.yaml",
            "description": "预训练+半监督 (方法 M2, 完整方案)",
            "order": 2,
            "depends_on": "B2_PURE_PRETRAIN"
        }
    },
    "ablation": {
        "A2_CONSISTENCY_ONLY": {
            "config": "ABLATION_CONSISTENCY_ONLY.yaml",
            "description": "消融: 只用一致性损失",
            "order": 1
        },
        "A3_PSEUDO_ONLY": {
            "config": "ABLATION_PSEUDO_ONLY.yaml",
            "description": "消融: 只用伪标签损失",
            "order": 2
        },
        "L1_LABEL_10PERCENT": {
            "config": "ABLATION_LABEL_10PERCENT.yaml",
            "description": "消融: 10%标签比例",
            "order": 3
        },
        "L2_LABEL_20PERCENT": {
            "config": "ABLATION_LABEL_20PERCENT.yaml",
            "description": "消融: 20%标签比例",
            "order": 4
        },
        "C1_NO_CURRICULUM": {
            "config": "ABLATION_NO_CURRICULUM.yaml",
            "description": "消融: 无课程学习(固定阈值)",
            "order": 5
        }
    }
}


def print_header(text, char='='):
    """打印格式化标题"""
    width = 70
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}\n")


def list_experiments():
    """列出所有实验配置"""
    print_header("所有实验配置列表")
    
    for stage, exps in EXPERIMENTS.items():
        print(f"\n【{stage.upper()}】")
        sorted_exps = sorted(exps.items(), key=lambda x: x[1].get('order', 99))
        for name, info in sorted_exps:
            config_path = os.path.join(CONFIG_DIR, info['config'])
            exists = "✓" if os.path.exists(config_path) else "✗"
            print(f"  {exists} {name}: {info['description']}")
            print(f"      配置: {info['config']}")
            if 'depends_on' in info:
                print(f"      依赖: {info['depends_on']}")


def run_experiment(name, config_file, description):
    """运行单个实验"""
    config_path = os.path.join(CONFIG_DIR, config_file)
    
    if not os.path.exists(config_path):
        print(f"[ERROR] 配置文件不存在: {config_path}")
        return False
    
    print_header(f"运行实验: {name}", '-')
    print(f"描述: {description}")
    print(f"配置: {config_file}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 构建命令
    cmd = [sys.executable, "main.py", "--config_file", config_path]
    
    try:
        # 运行实验
        result = subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] {name} 完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {name} 失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] {name} 被用户中断")
        return False


def run_stage(stage_name):
    """运行指定阶段的所有实验"""
    if stage_name not in EXPERIMENTS:
        print(f"[ERROR] 未知阶段: {stage_name}")
        print(f"可用阶段: {list(EXPERIMENTS.keys())}")
        return
    
    print_header(f"开始运行 {stage_name.upper()} 阶段实验")
    
    exps = EXPERIMENTS[stage_name]
    sorted_exps = sorted(exps.items(), key=lambda x: x[1].get('order', 99))
    
    results = {}
    for name, info in sorted_exps:
        # 检查依赖
        if 'depends_on' in info:
            dep = info['depends_on']
            print(f"\n[NOTE] {name} 依赖 {dep}")
            print(f"请确保已运行 {dep} 并更新 MODEL_PATH 配置")
            response = input("是否继续? (y/n): ")
            if response.lower() != 'y':
                print(f"跳过 {name}")
                results[name] = "SKIPPED"
                continue
        
        success = run_experiment(name, info['config'], info['description'])
        results[name] = "SUCCESS" if success else "FAILED"
    
    # 打印结果摘要
    print_header(f"{stage_name.upper()} 阶段完成")
    for name, status in results.items():
        print(f"  {name}: {status}")


def run_all():
    """运行所有实验"""
    print_header("开始运行所有实验")
    print("实验顺序: baseline → main → ablation")
    print()
    
    for stage in ["baseline", "main", "ablation"]:
        run_stage(stage)
        print()


def check_data_paths():
    """检查数据路径配置"""
    print_header("检查数据路径配置")
    
    # 需要检查的配置文件
    configs_to_check = [
        "PRETRAIN_PURE_RESPIRATION.yaml",
        "BASELINE_UBFC_100PERCENT.yaml",
        "TEST_COHFACE_GOLD.yaml"
    ]
    
    datasets = {
        "PURE": "F:/data/PURE",
        "UBFC-rPPG": "F:/UBFC-rPPG-data/data/UBFC-rPPG/DATASET_2",
        "COHFACE": "F:/data/COHFACE"
    }
    
    print("请确认以下数据路径配置正确:")
    for dataset, default_path in datasets.items():
        print(f"  - {dataset}: {default_path}")
    
    print("\n如果路径不正确，请修改对应的配置文件。")


def main():
    parser = argparse.ArgumentParser(
        description="rPPG呼吸检测实验运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python run_all_experiments.py --list             # 查看所有实验
    python run_all_experiments.py --stage baseline   # 运行baseline
    python run_all_experiments.py --stage main       # 运行主方法
    python run_all_experiments.py --stage ablation   # 运行消融实验
    python run_all_experiments.py --stage all        # 运行全部
    python run_all_experiments.py --check            # 检查数据路径
        """
    )
    
    parser.add_argument(
        '--stage', 
        choices=['baseline', 'main', 'ablation', 'all'],
        help='要运行的实验阶段'
    )
    parser.add_argument(
        '--list', 
        action='store_true',
        help='列出所有实验配置'
    )
    parser.add_argument(
        '--check', 
        action='store_true',
        help='检查数据路径配置'
    )
    parser.add_argument(
        '--exp',
        type=str,
        help='运行指定的单个实验 (例如: B2_PURE_PRETRAIN)'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
    elif args.check:
        check_data_paths()
    elif args.exp:
        # 查找并运行指定实验
        found = False
        for stage, exps in EXPERIMENTS.items():
            if args.exp in exps:
                info = exps[args.exp]
                run_experiment(args.exp, info['config'], info['description'])
                found = True
                break
        if not found:
            print(f"[ERROR] 未找到实验: {args.exp}")
            list_experiments()
    elif args.stage:
        if args.stage == 'all':
            run_all()
        else:
            run_stage(args.stage)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
