#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot training metrics from results.csv and dataset statistics from master.csv.

Usage:
    python plot_training_and_dataset.py \
        --results results.csv \
        --master master.csv \
        --outdir plots

Example:
    python plot_training_and_dataset.py \
        --results /home/shahriar/Documents/bank_did_auth/results.csv \
        --master /home/shahriar/Documents/bank_did_auth/data/master.csv \
        --outdir /home/shahriar/Documents/bank_did_auth/plots
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Style Config
# ============================================================

def set_plot_style():
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.85,
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "legend.frameon": True,
        },
    )

    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def save_fig(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ============================================================
# Utility Functions
# ============================================================

def add_value_labels(ax, rotation=0, fontsize=9):
    """
    Add numeric labels on top of bars.
    """
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", label_type="edge", fontsize=fontsize, rotation=rotation)
    ax.margins(y=0.15)


def safe_countplot(data, x, ax, title=None, hue=None, order=None, rotate_xticks=True):
    """
    Countplot wrapper with empty-column checks.
    """
    if x not in data.columns:
        ax.text(0.5, 0.5, f"Missing column: {x}", ha="center", va="center")
        ax.set_axis_off()
        return

    sns.countplot(data=data, x=x, hue=hue, order=order, ax=ax)

    if title:
        ax.set_title(title)

    ax.set_xlabel(x)
    ax.set_ylabel("Count")

    if rotate_xticks:
        ax.tick_params(axis="x", rotation=35)

    add_value_labels(ax)


def safe_lineplot(df, x, y, ax, title=None, marker="o", color=None):
    """
    Lineplot wrapper for training metrics.
    """
    if y not in df.columns:
        ax.text(0.5, 0.5, f"Missing column: {y}", ha="center", va="center")
        ax.set_axis_off()
        return

    sns.lineplot(data=df, x=x, y=y, marker=marker, ax=ax, color=color)

    if title:
        ax.set_title(title)

    ax.set_xlabel(x)
    ax.set_ylabel(y)


def shorten_labels(ax, max_len=18):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        if len(text) > max_len:
            text = text[:max_len] + "..."
        labels.append(text)
    ax.set_xticklabels(labels)


# ============================================================
# Results.csv Analysis
# ============================================================

def prepare_results_df(results_path, emission_factor_kg_per_kwh=0.494, overhead_multiplier=1.15):
    df = pd.read_csv(results_path)

    if "epoch" in df.columns:
        df = df.sort_values("epoch")

    # Energy calculation if possible
    if {"power_w", "elapsed_s"}.issubset(df.columns):
        df["energy_kwh_raw"] = df["power_w"] * df["elapsed_s"] / 3_600_000
        df["energy_kwh_adjusted"] = df["energy_kwh_raw"] * overhead_multiplier
        df["co2_kg"] = df["energy_kwh_adjusted"] * emission_factor_kg_per_kwh
        df["co2_g"] = df["co2_kg"] * 1000
        df["elapsed_min"] = df["elapsed_s"] / 60

    return df


def plot_training_losses(df, outdir):
    """
    Loss dashboard.
    """
    x = "epoch"

    loss_cols = [
        "loss_total",
        "loss_df",
        "loss_sp",
        "loss_temp",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for ax, col, color in zip(axes, loss_cols, colors):
        safe_lineplot(df, x, col, ax, title=col, color=color)

    fig.suptitle("Training Loss Curves", fontsize=20, fontweight="bold")
    save_fig(fig, Path(outdir) / "01_training_losses.png")


def plot_training_metrics_dashboard(df, outdir):
    """
    Main metrics dashboard.
    """
    x = "epoch"

    metric_cols = [
        "df_auc_roc",
        "df_ap",
        "df_eer",
        "df_acc_best_thresh",
        "df_video_auc",
        "sp_auc",
        "sp_acer",
        "sp_hter",
        "sp_tpr_at_fpr1",
        "temp_bin_acc",
        "temp_auc",
        "temp_df_auc",
    ]

    n_cols = 3
    n_rows = int(np.ceil(len(metric_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.8 * n_rows))
    axes = axes.flatten()

    palette = sns.color_palette("tab10", len(metric_cols))

    for i, col in enumerate(metric_cols):
        safe_lineplot(df, x, col, axes[i], title=col, color=palette[i % len(palette)])

        # For metrics mostly in [0,1], make the axis easier to compare
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0 and values.min() >= 0 and values.max() <= 1:
                axes[i].set_ylim(0, 1.05)

    for j in range(len(metric_cols), len(axes)):
        axes[j].set_axis_off()

    fig.suptitle("Training / Validation Metrics Dashboard", fontsize=22, fontweight="bold")
    save_fig(fig, Path(outdir) / "02_training_metrics_dashboard.png")


def plot_power_energy_dashboard(df, outdir):
    """
    Power, elapsed time, energy, CO2 dashboard.
    """
    available_any = any(c in df.columns for c in ["power_w", "elapsed_min", "energy_kwh_adjusted", "co2_g", "lr"])

    if not available_any:
        print("[Skip] Power/Energy dashboard: required columns not found.")
        return

    x = "epoch"

    cols = [
        ("power_w", "Average Power per Epoch [W]"),
        ("elapsed_min", "Elapsed Time per Epoch [min]"),
        ("energy_kwh_raw", "Raw Energy per Epoch [kWh]"),
        ("energy_kwh_adjusted", "Adjusted Energy per Epoch [kWh]"),
        ("co2_g", "Estimated CO2 per Epoch [g]"),
        ("lr", "Learning Rate"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    axes = axes.flatten()

    colors = ["#e67e22", "#3498db", "#2ecc71", "#16a085", "#7f8c8d", "#9b59b6"]

    for ax, (col, title), color in zip(axes, cols, colors):
        safe_lineplot(df, x, col, ax, title=title, color=color)

    fig.suptitle("Power, Energy, CO2 and LR Dashboard", fontsize=22, fontweight="bold")
    save_fig(fig, Path(outdir) / "03_power_energy_co2_dashboard.png")


def plot_task_weights(df, outdir):
    """
    Plot task weights if they exist.
    """
    weight_cols = ["w_df", "w_sp", "w_temp"]

    if not any(c in df.columns for c in weight_cols):
        print("[Skip] Task weights plot: columns not found.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    for col in weight_cols:
        if col in df.columns:
            sns.lineplot(data=df, x="epoch", y=col, marker="o", ax=ax, label=col)

    ax.set_title("Task Loss Weights")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight")
    ax.legend()

    save_fig(fig, Path(outdir) / "04_task_weights.png")


def save_results_summary(df, outdir):
    """
    Save enriched CSV and text summary.
    """
    outdir = Path(outdir)
    enriched_path = outdir / "results_with_energy_co2.csv"
    df.to_csv(enriched_path, index=False)
    print(f"[Saved] {enriched_path}")

    summary_lines = []
    summary_lines.append("=== Results Summary ===")

    if "epoch" in df.columns:
        summary_lines.append(f"Epochs: {df['epoch'].min()} to {df['epoch'].max()}")
        summary_lines.append(f"Number of rows: {len(df)}")

    if "elapsed_s" in df.columns:
        summary_lines.append(f"Total elapsed time: {df['elapsed_s'].sum():.2f} s")
        summary_lines.append(f"Total elapsed time: {df['elapsed_s'].sum() / 60:.2f} min")
        summary_lines.append(f"Total elapsed time: {df['elapsed_s'].sum() / 3600:.4f} h")

    if "power_w" in df.columns:
        summary_lines.append(f"Mean power: {df['power_w'].mean():.4f} W")
        summary_lines.append(f"Min power: {df['power_w'].min():.4f} W")
        summary_lines.append(f"Max power: {df['power_w'].max():.4f} W")

    if "energy_kwh_raw" in df.columns:
        summary_lines.append(f"Total raw energy: {df['energy_kwh_raw'].sum():.8f} kWh")

    if "energy_kwh_adjusted" in df.columns:
        summary_lines.append(f"Total adjusted energy: {df['energy_kwh_adjusted'].sum():.8f} kWh")

    if "co2_g" in df.columns:
        summary_lines.append(f"Total CO2: {df['co2_g'].sum():.4f} g")
        summary_lines.append(f"Total CO2: {df['co2_kg'].sum():.8f} kg")

    summary_text = "\n".join(summary_lines)
    summary_path = outdir / "results_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"[Saved] {summary_path}")


# ============================================================
# Master.csv Analysis
# ============================================================

def prepare_master_df(master_path):
    df = pd.read_csv(master_path)

    # Normalize potentially missing text columns
    for col in ["dataset", "label", "task", "spoof_type", "split", "subject_id", "video_path"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    return df


def plot_dataset_overview(master, outdir):
    """
    Overview count plots.
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 13))
    axes = axes.flatten()

    safe_countplot(master, "dataset", axes[0], title="Frames per Dataset")
    safe_countplot(master, "split", axes[1], title="Frames per Split")
    safe_countplot(master, "label", axes[2], title="Frames per Label")
    safe_countplot(master, "task", axes[3], title="Frames per Task")
    safe_countplot(master, "spoof_type", axes[4], title="Frames per Spoof Type")
    safe_countplot(master, "dataset", axes[5], title="Dataset x Split", hue="split")

    for ax in axes:
        shorten_labels(ax, max_len=22)

    fig.suptitle("Dataset Overview from master.csv", fontsize=22, fontweight="bold")
    save_fig(fig, Path(outdir) / "05_dataset_overview_counts.png")


def plot_dataset_cross_tabs(master, outdir):
    """
    Heatmaps for dataset-label/split/task relations.
    """
    pairs = [
        ("dataset", "split"),
        ("dataset", "label"),
        ("dataset", "task"),
        ("split", "label"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()

    for ax, (row_col, col_col) in zip(axes, pairs):
        if row_col not in master.columns or col_col not in master.columns:
            ax.text(0.5, 0.5, f"Missing: {row_col} or {col_col}", ha="center", va="center")
            ax.set_axis_off()
            continue

        ct = pd.crosstab(master[row_col], master[col_col])
        sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{row_col} × {col_col}")
        ax.set_xlabel(col_col)
        ax.set_ylabel(row_col)

    fig.suptitle("Dataset Cross-tab Heatmaps", fontsize=22, fontweight="bold")
    save_fig(fig, Path(outdir) / "06_dataset_crosstab_heatmaps.png")


def plot_subject_video_frame_stats(master, outdir):
    """
    Subject/video/frame statistics by dataset/split/label.
    """
    required = ["dataset", "subject_id", "video_path", "frame_path"]
    for col in required:
        if col not in master.columns:
            print(f"[Skip] Subject/video/frame stats: missing column {col}")
            return

    # Basic group stats per dataset
    stats_dataset = (
        master.groupby("dataset")
        .agg(
            frames=("frame_path", "count"),
            subjects=("subject_id", "nunique"),
            videos=("video_path", "nunique"),
        )
        .reset_index()
        .sort_values("frames", ascending=False)
    )

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    sns.barplot(data=stats_dataset, x="dataset", y="frames", ax=axes[0], color="#3498db")
    axes[0].set_title("Number of Frames per Dataset")
    axes[0].tick_params(axis="x", rotation=35)
    add_value_labels(axes[0])

    sns.barplot(data=stats_dataset, x="dataset", y="subjects", ax=axes[1], color="#2ecc71")
    axes[1].set_title("Number of Subjects per Dataset")
    axes[1].tick_params(axis="x", rotation=35)
    add_value_labels(axes[1])

    sns.barplot(data=stats_dataset, x="dataset", y="videos", ax=axes[2], color="#e67e22")
    axes[2].set_title("Number of Videos per Dataset")
    axes[2].tick_params(axis="x", rotation=35)
    add_value_labels(axes[2])

    fig.suptitle("Dataset Size Statistics", fontsize=22, fontweight="bold")
    save_fig(fig, Path(outdir) / "07_subject_video_frame_stats.png")

    # Save stats CSV
    stats_path = Path(outdir) / "dataset_stats_by_dataset.csv"
    stats_dataset.to_csv(stats_path, index=False)
    print(f"[Saved] {stats_path}")


def plot_split_label_stats(master, outdir):
    """
    Statistics grouped by split and label.
    """
    required = ["split", "label", "subject_id", "video_path", "frame_path"]
    for col in required:
        if col not in master.columns:
            print(f"[Skip] Split/label stats: missing column {col}")
            return

    stats = (
        master.groupby(["split", "label"])
        .agg(
            frames=("frame_path", "count"),
            subjects=("subject_id", "nunique"),
            videos=("video_path", "nunique"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    sns.barplot(data=stats, x="split", y="frames", hue="label", ax=axes[0])
    axes[0].set_title("Frames by Split and Label")
    add_value_labels(axes[0])

    sns.barplot(data=stats, x="split", y="subjects", hue="label", ax=axes[1])
    axes[1].set_title("Subjects by Split and Label")
    add_value_labels(axes[1])

    sns.barplot(data=stats, x="split", y="videos", hue="label", ax=axes[2])
    axes[2].set_title("Videos by Split and Label")
    add_value_labels(axes[2])

    fig.suptitle("Split/Label Statistics", fontsize=22, fontweight="bold")
    save_fig(fig, Path(outdir) / "08_split_label_stats.png")

    stats_path = Path(outdir) / "dataset_stats_by_split_label.csv"
    stats.to_csv(stats_path, index=False)
    print(f"[Saved] {stats_path}")


def plot_clip_distribution(master, outdir):
    """
    Clip/video/frame distribution plots.
    """
    available = any(c in master.columns for c in ["clip_index", "clip_start_frame", "frame_num", "src_frame_idx"])

    if not available:
        print("[Skip] Clip distribution: clip/frame numeric columns not found.")
        return

    numeric_cols = []
    for col in ["clip_index", "clip_start_frame", "frame_num", "src_frame_idx", "video_index"]:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce")
            numeric_cols.append(col)

    if not numeric_cols:
        print("[Skip] Clip distribution: no valid numeric columns.")
        return

    n_cols = 2
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, numeric_cols):
        sns.histplot(data=master, x=col, bins=40, kde=False, ax=ax, color="#34495e")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_axis_off()

    fig.suptitle("Clip / Frame Numeric Distributions", fontsize=22, fontweight="bold")
    save_fig(fig, Path(outdir) / "09_clip_frame_distributions.png")


def plot_frames_per_video_subject(master, outdir):
    """
    Distributions of frames per video and frames per subject.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    plotted = False

    if "video_path" in master.columns and "frame_path" in master.columns:
        frames_per_video = master.groupby("video_path")["frame_path"].count().reset_index(name="frames")
        sns.histplot(frames_per_video["frames"], bins=50, kde=True, ax=axes[0], color="#2980b9")
        axes[0].set_title("Frames per Video Distribution")
        axes[0].set_xlabel("Frames per Video")
        axes[0].set_ylabel("Number of Videos")
        plotted = True
    else:
        axes[0].text(0.5, 0.5, "Missing video_path/frame_path", ha="center", va="center")
        axes[0].set_axis_off()

    if "subject_id" in master.columns and "frame_path" in master.columns:
        frames_per_subject = master.groupby("subject_id")["frame_path"].count().reset_index(name="frames")
        sns.histplot(frames_per_subject["frames"], bins=50, kde=True, ax=axes[1], color="#27ae60")
        axes[1].set_title("Frames per Subject Distribution")
        axes[1].set_xlabel("Frames per Subject")
        axes[1].set_ylabel("Number of Subjects")
        plotted = True
    else:
        axes[1].text(0.5, 0.5, "Missing subject_id/frame_path", ha="center", va="center")
        axes[1].set_axis_off()

    if plotted:
        fig.suptitle("Frames per Video / Subject", fontsize=22, fontweight="bold")
        save_fig(fig, Path(outdir) / "10_frames_per_video_subject.png")
    else:
        plt.close(fig)
        print("[Skip] Frames per video/subject plot.")


def save_master_summary(master, outdir):
    """
    Save dataset summary text.
    """
    outdir = Path(outdir)

    lines = []
    lines.append("=== Master Dataset Summary ===")
    lines.append(f"Total rows/frames: {len(master):,}")

    if "dataset" in master.columns:
        lines.append(f"Datasets: {master['dataset'].nunique():,}")
        lines.append("")
        lines.append("Frames per dataset:")
        lines.append(master["dataset"].value_counts().to_string())

    if "split" in master.columns:
        lines.append("")
        lines.append("Frames per split:")
        lines.append(master["split"].value_counts().to_string())

    if "label" in master.columns:
        lines.append("")
        lines.append("Frames per label:")
        lines.append(master["label"].value_counts().to_string())

    if "task" in master.columns:
        lines.append("")
        lines.append("Frames per task:")
        lines.append(master["task"].value_counts().to_string())

    if "spoof_type" in master.columns:
        lines.append("")
        lines.append("Frames per spoof_type:")
        lines.append(master["spoof_type"].value_counts().to_string())

    if "subject_id" in master.columns:
        lines.append("")
        lines.append(f"Unique subjects: {master['subject_id'].nunique():,}")

    if "video_path" in master.columns:
        lines.append(f"Unique videos: {master['video_path'].nunique():,}")

    if {"dataset", "subject_id", "video_path", "frame_path"}.issubset(master.columns):
        lines.append("")
        lines.append("Stats by dataset:")
        stats_dataset = (
            master.groupby("dataset")
            .agg(
                frames=("frame_path", "count"),
                subjects=("subject_id", "nunique"),
                videos=("video_path", "nunique"),
            )
            .sort_values("frames", ascending=False)
        )
        lines.append(stats_dataset.to_string())

    summary_text = "\n".join(lines)
    summary_path = outdir / "master_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"[Saved] {summary_path}")


# ============================================================
# Combined Report Plot
# ============================================================

def plot_compact_final_report(results, master, outdir):
    """
    A compact final report figure combining key training and dataset stats.
    """
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    safe_lineplot(results, "epoch", "loss_total", ax1, title="Total Loss", color="#e74c3c")

    ax2 = fig.add_subplot(gs[0, 1])
    safe_lineplot(results, "epoch", "df_auc_roc", ax2, title="Deepfake AUC ROC", color="#3498db")
    if "df_auc_roc" in results.columns:
        ax2.set_ylim(0, 1.05)

    ax3 = fig.add_subplot(gs[0, 2])
    safe_lineplot(results, "epoch", "sp_auc", ax3, title="Spoof AUC", color="#2ecc71")
    if "sp_auc" in results.columns:
        ax3.set_ylim(0, 1.05)

    ax4 = fig.add_subplot(gs[1, 0])
    safe_lineplot(results, "epoch", "power_w", ax4, title="Power [W]", color="#e67e22")

    ax5 = fig.add_subplot(gs[1, 1])
    safe_lineplot(results, "epoch", "energy_kwh_adjusted", ax5, title="Energy [kWh]", color="#16a085")

    ax6 = fig.add_subplot(gs[1, 2])
    safe_lineplot(results, "epoch", "co2_g", ax6, title="CO2 [g]", color="#7f8c8d")

    ax7 = fig.add_subplot(gs[2, 0])
    safe_countplot(master, "dataset", ax7, title="Frames per Dataset")
    shorten_labels(ax7)

    ax8 = fig.add_subplot(gs[2, 1])
    safe_countplot(master, "split", ax8, title="Frames per Split")
    shorten_labels(ax8)

    ax9 = fig.add_subplot(gs[2, 2])
    safe_countplot(master, "label", ax9, title="Frames per Label")
    shorten_labels(ax9)

    fig.suptitle("Compact Training + Dataset Report", fontsize=26, fontweight="bold")
    save_fig(fig, Path(outdir) / "11_compact_final_report.png")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results.csv",
    )

    parser.add_argument(
        "--master",
        type=str,
        required=True,
        help="Path to master.csv",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="plots",
        help="Output directory for generated plots",
    )

    parser.add_argument(
        "--emission-factor",
        type=float,
        default=0.494,
        help="Grid emission factor in kg CO2/kWh. Default: 0.494 for Iran estimate.",
    )

    parser.add_argument(
        "--overhead",
        type=float,
        default=1.15,
        help="Overhead multiplier for system/power losses. Default: 1.15",
    )

    args = parser.parse_args()

    set_plot_style()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Loading results.csv ===")
    results = prepare_results_df(
        args.results,
        emission_factor_kg_per_kwh=args.emission_factor,
        overhead_multiplier=args.overhead,
    )

    print("=== Loading master.csv ===")
    master = prepare_master_df(args.master)

    print("=== Plotting training results ===")
    plot_training_losses(results, outdir)
    plot_training_metrics_dashboard(results, outdir)
    plot_power_energy_dashboard(results, outdir)
    plot_task_weights(results, outdir)
    save_results_summary(results, outdir)

    print("=== Plotting dataset statistics ===")
    plot_dataset_overview(master, outdir)
    plot_dataset_cross_tabs(master, outdir)
    plot_subject_video_frame_stats(master, outdir)
    plot_split_label_stats(master, outdir)
    plot_clip_distribution(master, outdir)
    plot_frames_per_video_subject(master, outdir)
    save_master_summary(master, outdir)

    print("=== Plotting compact final report ===")
    plot_compact_final_report(results, master, outdir)

    print("\nDone.")
    print(f"All outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
