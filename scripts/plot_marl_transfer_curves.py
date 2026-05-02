"""Plot selected MARL transfer learning curves for the LaTeX draft.

This script intentionally uses a fixed whitelist of Sacred run ids. The local
results directory contains older experiments, failed runs, and unpaired VDN
candidates that should not enter the paper figures.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SACRED_DIR = ROOT / "results" / "sacred"
FIGURE_DIR = ROOT / "paper" / "latex" / "figures"
DIAGNOSTICS_DIR = ROOT.parent / "pymarl-results" / "diagnostics"


@dataclass(frozen=True)
class Series:
    run_id: int
    x: np.ndarray
    y: np.ndarray


QMIX_VARIANTS_5M = {
    "QMIX": [58, 60, 79],
    "AttnRes-Full": [61, 59, 66],
    "AttnRes-L2": [75, 82, 76],
    "AttnRes-Block": [64, 67, 65],
    "Depth-MLP": [80, 89, 85],
}

QMIX_VARIANTS_3S5Z = {
    "QMIX": [55, 54, 87],
    "AttnRes-Full": [56, 53, 70],
    "AttnRes-L2": [73, 72, 74],
    "AttnRes-Block": [71, 69, 84],
    "Depth-MLP": [77, 90, 88],
}

CROSS_ALGORITHM_5M = {
    "IQL": [92, 93, 91],
    "IQL+AttnRes-L2": [98, 95, 94],
    "VDN": [100, 101, 102],
    "VDN+AttnRes-L2": [97, 99, 96],
    "QMIX": [58, 60, 79],
    "QMIX+AttnRes-L2": [75, 82, 76],
}

STYLE = {
    "QMIX": "#1f77b4",
    "AttnRes-Full": "#d62728",
    "AttnRes-L2": "#2ca02c",
    "AttnRes-Block": "#9467bd",
    "Depth-MLP": "#ff7f0e",
    "IQL": "#17becf",
    "IQL+AttnRes-L2": "#bcbd22",
    "VDN": "#8c564b",
    "VDN+AttnRes-L2": "#e377c2",
    "QMIX+AttnRes-L2": "#2ca02c",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_series(run_id: int) -> Series:
    run_dir = SACRED_DIR / str(run_id)
    info_path = run_dir / "info.json"
    run_path = run_dir / "run.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json for Sacred run {run_id}: {info_path}")

    run = load_json(run_path) if run_path.exists() else {}
    status = run.get("status")
    if status != "COMPLETED":
        raise ValueError(f"Sacred run {run_id} is not COMPLETED: status={status!r}")

    info = load_json(info_path)
    y = np.asarray(info.get("test_battle_won_mean", []), dtype=float)
    x = np.asarray(info.get("test_battle_won_mean_T", []), dtype=float)
    if y.size == 0 or x.size == 0:
        raise ValueError(f"Sacred run {run_id} has no test_battle_won_mean time series")
    if y.size != x.size:
        raise ValueError(f"Sacred run {run_id} has mismatched x/y lengths: {x.size} vs {y.size}")
    return Series(run_id=run_id, x=x, y=y)


def align_series(series: Iterable[Series]) -> tuple[np.ndarray, np.ndarray]:
    series = list(series)
    common_start = max(float(s.x[0]) for s in series)
    common_end = min(float(s.x[-1]) for s in series)
    num_points = min(len(s.x) for s in series)
    grid = np.linspace(common_start, common_end, num_points)
    values = np.vstack([np.interp(grid, s.x, s.y) for s in series])
    return grid, values


def plot_curve(groups: dict[str, list[int]], title: str, output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3))

    for label, run_ids in groups.items():
        series = [load_series(run_id) for run_id in run_ids]
        x, values = align_series(series)
        color = STYLE.get(label)

        x_million = x / 1_000_000.0
        for row in values:
            ax.plot(x_million, row, color=color, alpha=0.18, linewidth=0.8)

        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax.plot(x_million, mean, label=label, color=color, linewidth=2.0)
        ax.fill_between(
            x_million,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=color,
            alpha=0.13,
            linewidth=0,
        )

    ax.set_title(title)
    ax.set_xlabel("Environment timesteps (million)")
    ax.set_ylabel("Test battle won rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(left=0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()

    out_path = FIGURE_DIR / output_name
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def load_wall_hours() -> dict[str, float]:
    csv_path = DIAGNOSTICS_DIR / "marl_transfer_primary_qmix_table.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing diagnostics CSV: {csv_path}")

    values: dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["map"] == "5m_vs_6m":
                values[row["config"]] = float(row["wall_hours_mean"])
    return values


def plot_wall_time_bar() -> None:
    config_order = [
        ("qmix", "QMIX"),
        ("qmix_attnres", "AttnRes-Full"),
        ("qmix_attnres_l2", "AttnRes-L2"),
        ("qmix_attnres_block", "AttnRes-Block"),
        ("qmix_depth_mlp", "Depth-MLP"),
    ]
    wall_hours = load_wall_hours()
    labels = [label for _, label in config_order]
    values = [wall_hours[config] for config, _ in config_order]
    colors = [STYLE.get(label, "#777777") for label in labels]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    bars = ax.bar(labels, values, color=colors, alpha=0.82)
    ax.set_ylabel("Wall-clock hours")
    ax.set_title("Training Time on 5m_vs_6m")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    out_path = FIGURE_DIR / "5m_vs_6m_wall_time_bar.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_curve(
        QMIX_VARIANTS_5M,
        "5m_vs_6m: QMIX Variants",
        "5m_vs_6m_qmix_win_curve.pdf",
    )
    plot_curve(
        QMIX_VARIANTS_3S5Z,
        "3s5z: QMIX Variants",
        "3s5z_qmix_win_curve.pdf",
    )
    plot_curve(
        CROSS_ALGORITHM_5M,
        "5m_vs_6m: Lightweight AttnRes-L2 Cross-Algorithm Check",
        "5m_vs_6m_cross_algorithm_win_curve.pdf",
    )
    plot_wall_time_bar()


if __name__ == "__main__":
    main()
