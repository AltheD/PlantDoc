"""
E6 超参数消融实验批量运行脚本

使用方法：
    python -m src.training.run_e6_ablation --device cpu --num-workers 0

或者逐个运行（见脚本底部的实验配置列表）。
"""

import argparse
import subprocess
import sys
from pathlib import Path


E6_EXPERIMENTS = [
    {
        "name": "E6-1_adamw_lr5e4",
        "description": "AdamW, lr_head=5e-4, lr_backbone=5e-5",
        "args": {
            "optimizer": "adamw",
            "lr-head": 5e-4,
            "lr-backbone": 5e-5,
            "augment": "basic",
            "batch-size": 16,
        },
    },
    {
        "name": "E6-2_adamw_lr1e3_uniform",
        "description": "AdamW, 统一学习率 lr=1e-3",
        "args": {
            "optimizer": "adamw",
            "lr-head": 1e-3,
            "lr-backbone": 1e-3,
            "augment": "basic",
            "batch-size": 16,
        },
    },
    {
        "name": "E6-3_sgd_lr3e3",
        "description": "SGD (momentum=0.9), lr=3e-3",
        "args": {
            "optimizer": "sgd",
            "lr-head": 3e-3,
            "lr-backbone": 3e-3,
            "augment": "basic",
            "batch-size": 16,
        },
    },
    {
        "name": "E6-4_adamw_batch8",
        "description": "AdamW, batch_size=8 (vs E3 的 16)",
        "args": {
            "optimizer": "adamw",
            "lr-head": 1e-3,
            "lr-backbone": 1e-4,
            "augment": "basic",
            "batch-size": 8,
        },
    },
]


def build_command(exp_config: dict, base_args: argparse.Namespace) -> list:
    """构建单个实验的命令行参数列表"""
    cmd = [
        sys.executable,
        "-m",
        "src.training.resnet50_baseline",
        "--experiment-name",
        exp_config["name"],
        "--log-csv",
        f"outputs/logs/{exp_config['name']}.csv",
        "--pred-csv",
        f"outputs/predictions/{exp_config['name']}_test.csv",
    ]
    for key, value in exp_config["args"].items():
        key_arg = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(key_arg)
        elif value is not None:
            cmd.append(key_arg)
            cmd.append(str(value))
    for attr in ["device", "num-workers", "freeze-epochs", "finetune-epochs", "seed"]:
        val = getattr(base_args, attr.replace("-", "_"), None)
        if val is not None:
            cmd.append(f"--{attr}")
            cmd.append(str(val))
    return cmd


def run_single_experiment(exp_config: dict, base_args: argparse.Namespace, dry_run: bool = False):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"实验: {exp_config['name']}")
    print(f"描述: {exp_config['description']}")
    print(f"{'='*60}")
    cmd = build_command(exp_config, base_args)
    if dry_run:
        print("命令（预览）:", " ".join(cmd))
        return
    print("开始运行...")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    if result.returncode == 0:
        print(f"✓ {exp_config['name']} 完成")
    else:
        print(f"✗ {exp_config['name']} 失败 (exit code {result.returncode})")
    return result.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E6 超参数消融实验批量运行")
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", type=str, help="只运行指定实验（如 E6-1_adamw_lr5e4）")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不实际运行")
    parser.add_argument("--list", action="store_true", help="列出所有实验配置")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list:
        print("E6 实验配置列表：\n")
        for i, exp in enumerate(E6_EXPERIMENTS, 1):
            print(f"{i}. {exp['name']}")
            print(f"   {exp['description']}")
            print(f"   参数: {exp['args']}\n")
        return
    experiments_to_run = E6_EXPERIMENTS
    if args.experiment:
        experiments_to_run = [e for e in E6_EXPERIMENTS if e["name"] == args.experiment]
        if not experiments_to_run:
            print(f"错误：未找到实验 '{args.experiment}'")
            print("可用实验：")
            for exp in E6_EXPERIMENTS:
                print(f"  - {exp['name']}")
            sys.exit(1)
    print(f"准备运行 {len(experiments_to_run)} 个 E6 实验")
    print(f"设备: {args.device}, 工作进程: {args.num_workers}")
    if args.dry_run:
        print("（预览模式，不会实际运行）\n")
    results = []
    for exp in experiments_to_run:
        success = run_single_experiment(exp, args, dry_run=args.dry_run)
        results.append((exp["name"], success))
    print(f"\n{'='*60}")
    print("实验总结：")
    for name, success in results:
        status = "✓ 完成" if success else "✗ 失败"
        print(f"  {name}: {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

