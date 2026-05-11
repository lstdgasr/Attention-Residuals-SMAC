#!/usr/bin/env python
"""Plot MARL transfer learning curves from Sacred runs.

The script discovers the best completed run for each map/config/seed from
config.json, info.json, and run.json. This lets newly completed seed4/5 and
3s5z cross-algorithm follow-up runs enter the paper figures automatically.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SACRED_DIR = ROOT / "results" / "sacred"
DEFAULT_FIGURE_DIR = ROOT / "paper" / "latex" / "figures"

PRIMARY_MAPS = ["5m_vs_6m", "3s5z"]
SEEDS_3 = ["1", "2", "3"]
SEEDS_5 = ["1", "2", "3", "4", "5"]
HARDMIX_SEEDS_BY_MAP = {
    "5m_vs_6m": ["4", "5"],
    "8m_vs_9m": ["1", "2"],
    "3s5z_vs_3s6z": ["1", "2"],
    "MMM2": ["1", "2"],
}
PRIMARY_GROUPS = [
    ("qmix", "QMIX", SEEDS_5),
    ("qmix_attnres", "AttnRes-Full", SEEDS_3),
    ("qmix_attnres_l2", "AttnRes-L2", SEEDS_5),
    ("qmix_attnres_block", "AttnRes-Block", SEEDS_3),
    ("qmix_depth_mlp", "Depth-MLP", SEEDS_5),
    ("qmix_attncomm_l2_other", "AttnComm-L2-Other", SEEDS_3),
    ("qmix_attncomm_l2_self", "AttnComm-L2-Self", SEEDS_3),
]
HARDMIX_GROUPS = [
    ("qmix", "QMIX", None),
    ("qmix_attnres_l2", "AttnRes-L2", None),
    ("qmix_depth_mlp", "Depth-MLP", None),
]
CROSS_GROUPS = [
    ("iql", "IQL", SEEDS_3),
    ("iql_attnres_l2", "IQL+AttnRes-L2", SEEDS_3),
    ("vdn", "VDN", SEEDS_3),
    ("vdn_attnres_l2", "VDN+AttnRes-L2", SEEDS_3),
    ("qmix", "QMIX", SEEDS_3),
    ("qmix_attnres_l2", "QMIX+AttnRes-L2", SEEDS_3),
]

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
    "AttnComm-L2-Other": "#7f7f7f",
    "AttnComm-L2-Self": "#1b9e77",
}


@dataclass(frozen=True)
class Run:
    run_id: str
    map_name: str
    name: str
    seed: str
    status: str
    start_time: str
    stop_time: str
    info: dict


@dataclass(frozen=True)
class Series:
    run_id: str
    x: np.ndarray
    y: np.ndarray


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sort_key(path):
    return (0, int(path.name)) if path.name.isdigit() else (1, path.name)


def comma_list(value):
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def group_seeds(map_name, seeds, hardmix=False, seed_override=None):
    if seed_override is not None:
        return seed_override
    if seeds is not None:
        return seeds
    if hardmix and map_name in HARDMIX_SEEDS_BY_MAP:
        return HARDMIX_SEEDS_BY_MAP[map_name]
    return SEEDS_5


def _as_float(value) -> float:
    if isinstance(value, dict) and "value" in value:
        return float(value["value"])
    return float(value)


def metric_series(run: Run, metric: str):
    if metric not in run.info:
        return None, None
    values = np.asarray([_as_float(v) for v in run.info.get(metric, [])], dtype=float)
    if values.size == 0:
        return None, None
    t_key = "{}_T".format(metric)
    if t_key in run.info and len(run.info[t_key]) == len(values):
        steps = np.asarray([_as_float(v) for v in run.info[t_key]], dtype=float)
    else:
        steps = np.arange(values.size, dtype=float)
    return steps, values


def run_rank(run):
    steps, values = metric_series(run, "test_battle_won_mean")
    completed = 1 if run.status == "COMPLETED" else 0
    n_points = len(values) if values is not None else 0
    last_t = float(steps[-1]) if steps is not None and len(steps) else 0.0
    return completed, n_points, last_t


def load_best_runs(sacred_dir: Path):
    runs = {}
    if not sacred_dir.exists():
        raise FileNotFoundError("Missing Sacred directory: {}".format(sacred_dir))

    for run_dir in sorted(sacred_dir.iterdir(), key=_sort_key):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        info_path = run_dir / "info.json"
        run_path = run_dir / "run.json"
        if not config_path.exists() or not info_path.exists():
            continue

        config = load_json(config_path)
        info = load_json(info_path)
        run_meta = load_json(run_path) if run_path.exists() else {}
        map_name = config.get("env_args", {}).get("map_name")
        name = config.get("name")
        seed = str(config.get("seed"))
        if not map_name or not name or seed == "None":
            continue

        run = Run(
            run_id=run_dir.name,
            map_name=map_name,
            name=name,
            seed=seed,
            status=run_meta.get("status", ""),
            start_time=run_meta.get("start_time", ""),
            stop_time=run_meta.get("stop_time", ""),
            info=info,
        )
        key = (map_name, name, seed)
        old = runs.get(key)
        if old is None or run_rank(run) > run_rank(old):
            runs[key] = run
    return runs


def series_for_group(
    runs,
    map_name,
    config,
    seeds,
):
    series = []
    missing = []
    for seed in seeds:
        run = runs.get((map_name, config, seed))
        if run is None:
            missing.append(seed)
            continue
        if run.status and run.status != "COMPLETED":
            missing.append("{}({})".format(seed, run.status))
            continue
        x, y = metric_series(run, "test_battle_won_mean")
        if x is None or y is None or len(x) != len(y):
            missing.append("{}(no_series)".format(seed))
            continue
        series.append(Series(run_id=run.run_id, x=x, y=y))
    return series, missing


def align_series(series):
    series = list(series)
    common_start = max(float(s.x[0]) for s in series)
    common_end = min(float(s.x[-1]) for s in series)
    num_points = min(len(s.x) for s in series)
    if common_end <= common_start or num_points < 2:
        grid = series[0].x
    else:
        grid = np.linspace(common_start, common_end, num_points)
    values = np.vstack([np.interp(grid, s.x, s.y) for s in series])
    return grid, values


def plot_curve(
    runs,
    map_name,
    groups,
    title,
    output_path,
    seed_override=None,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    plotted = 0

    for config, label, seeds in groups:
        seeds = group_seeds(map_name, seeds, hardmix=(groups is HARDMIX_GROUPS), seed_override=seed_override)
        series, missing = series_for_group(runs, map_name, config, seeds)
        if missing:
            print("{} {} missing/partial seeds: {}".format(map_name, config, ",".join(missing)))
        if not series:
            continue

        x, values = align_series(series)
        color = STYLE.get(label)
        x_million = x / 1_000_000.0
        for row in values:
            ax.plot(x_million, row, color=color, alpha=0.18, linewidth=0.8)

        mean = values.mean(axis=0)
        std = values.std(axis=0)
        label_with_n = "{} (n={})".format(label, len(series))
        ax.plot(x_million, mean, label=label_with_n, color=color, linewidth=2.0)
        ax.fill_between(
            x_million,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("Skipped {} because no completed series were found.".format(output_path.name))
        return

    ax.set_title(title)
    ax.set_xlabel("Environment timesteps (million)")
    ax.set_ylabel("Test battle won rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(left=0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print("Wrote {}".format(output_path))


def _parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def wall_hours(run):
    start = _parse_time(run.start_time)
    stop = _parse_time(run.stop_time)
    if not start or not stop:
        return None
    return (stop - start).total_seconds() / 3600.0


def plot_wall_time_bar(
    runs,
    map_name,
    output_path,
):
    labels = []
    values = []
    colors = []
    for config, label, seeds in PRIMARY_GROUPS:
        config_values = []
        for seed in seeds:
            run = runs.get((map_name, config, seed))
            if run is None or (run.status and run.status != "COMPLETED"):
                continue
            value = wall_hours(run)
            if value is not None:
                config_values.append(value)
        if not config_values:
            continue
        labels.append(label)
        values.append(float(np.mean(config_values)))
        colors.append(STYLE.get(label, "#777777"))

    if not values:
        print("Skipped {} because no wall-clock metadata were found.".format(output_path.name))
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    bars = ax.bar(labels, values, color=colors, alpha=0.82)
    ax.set_ylabel("Wall-clock hours")
    ax.set_title("Training Time on {}".format(map_name))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            "{:.2f}".format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print("Wrote {}".format(output_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot paper figures for the MARL transfer follow-up.")
    parser.add_argument("--sacred-dir", default=str(DEFAULT_SACRED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--maps", default=None, help="Comma-separated maps to plot. When set, plots QMIX/AttnRes-L2/Depth-MLP only.")
    parser.add_argument("--seeds", default=None, help="Comma-separated expected seeds for every plotted map/config.")
    args = parser.parse_args(argv)

    sacred_dir = Path(args.sacred_dir)
    output_dir = Path(args.output_dir)
    runs = load_best_runs(sacred_dir)
    maps = comma_list(args.maps)
    seed_override = comma_list(args.seeds)
    primary_maps = maps if maps is not None else PRIMARY_MAPS
    primary_groups = HARDMIX_GROUPS if maps is not None else PRIMARY_GROUPS

    for map_name in primary_maps:
        plot_curve(
            runs,
            map_name,
            primary_groups,
            "{}: QMIX Variants".format(map_name),
            output_dir / "{}_qmix_win_curve.pdf".format(map_name),
            seed_override=seed_override,
        )
        if maps is not None:
            continue
        plot_curve(
            runs,
            map_name,
            CROSS_GROUPS,
            "{}: Lightweight AttnRes-L2 Cross-Algorithm Check".format(map_name),
            output_dir / "{}_cross_algorithm_win_curve.pdf".format(map_name),
        )

    plot_wall_time_bar(
        runs,
        "5m_vs_6m",
        output_dir / "5m_vs_6m_wall_time_bar.pdf",
    )


if __name__ == "__main__":
    main()
