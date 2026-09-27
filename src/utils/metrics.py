"""指标计算与可视化工具"""

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np

from src.utils.helpers import logger


def extract_metrics_from_results(results_path: Path) -> dict:
    """从 YOLO 训练结果目录中提取核心指标

    Args:
        results_path: YOLO 训练产生的 runs/detect/trainXX/ 目录

    Returns:
        包含 mAP@50, mAP@50-95, precision, recall 的字典
    """
    metrics = {}
    results_path = Path(results_path)

    # 读取 results.csv (YOLO 自动生成)
    csv_path = results_path / "results.csv"
    if csv_path.exists():
        import csv
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                # 取最后一行（最终指标），去除开头多余空格
                last = {}
                for k, v in rows[-1].items():
                    last[k.strip()] = v.strip()
                metrics["mAP50"] = float(last.get("metrics/mAP50(B)", 0))
                metrics["mAP50-95"] = float(last.get("metrics/mAP50-95(B)", 0))
                metrics["precision"] = float(last.get("metrics/precision(B)", 0))
                metrics["recall"] = float(last.get("metrics/recall(B)", 0))
    else:
        logger.warning(f"找不到 results.csv: {csv_path}")

    return metrics


def plot_training_curves(results_path: Path, output_dir: Path) -> None:
    """绘制训练曲线（loss、mAP），保存到 output_dir

    Args:
        results_path: YOLO 训练产生的 runs/detect/trainXX/ 目录
        output_dir: 输出图片的目录
    """
    csv_path = Path(results_path) / "results.csv"
    if not csv_path.exists():
        logger.warning(f"无法绘制训练曲线: results.csv 不存在 ({csv_path})")
        return

    import pandas as pd
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 左上：Loss 曲线
    ax = axes[0, 0]
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    for col in loss_cols:
        ax.plot(df[col], label=col, linewidth=0.8, alpha=0.8)
    ax.set_title("Training Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 右上：mAP 曲线
    ax = axes[0, 1]
    map_cols = [c for c in df.columns if "map" in c.lower()]
    for col in map_cols:
        ax.plot(df[col], label=col, linewidth=1.2)
    ax.set_title("mAP Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 左下：Precision
    ax = axes[1, 0]
    if "metrics/precision(B)" in df.columns:
        ax.plot(df["metrics/precision(B)"], label="Precision", color="green")
    ax.set_title("Precision")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 右下：Recall
    ax = axes[1, 1]
    if "metrics/recall(B)" in df.columns:
        ax.plot(df["metrics/recall(B)"], label="Recall", color="orange")
    ax.set_title("Recall")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Training Curves - {results_path.name}", fontsize=14)
    plt.tight_layout()

    save_path = output_dir / "training_curves.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"训练曲线已保存: {save_path}")


def plot_experiment_comparison(
    experiment_dirs: list[Path],
    output_path: Path,
) -> None:
    """绘制多组实验的指标对比柱状图

    Args:
        experiment_dirs: 各实验目录列表
        output_path: 输出图片路径
    """
    names = []
    map50_list = []
    map5095_list = []

    for exp_dir in experiment_dirs:
        metrics_path = Path(exp_dir) / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                m = json.load(f)
            names.append(exp_dir.name)
            map50_list.append(m.get("mAP50", 0))
            map5095_list.append(m.get("mAP50-95", 0))

    if not names:
        logger.warning("没有可用的实验指标用于对比")
        return

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, map50_list, width, label="mAP@50", color="#2196F3")
    bars2 = ax.bar(x + width / 2, map5095_list, width, label="mAP@50-95", color="#4CAF50")

    # 在柱子上标注数值
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_ylabel("mAP")
    ax.set_title("Experiment Comparison: mAP@50 vs mAP@50-95")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"实验对比图已保存: {output_path}")
