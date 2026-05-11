#!/usr/bin/env python
"""Plot AttnComm agent-to-agent attention heatmaps from Sacred info.json."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SACRED_DIR = ROOT / "results" / "sacred"
DEFAULT_FIGURE_DIR = ROOT / "paper" / "latex" / "figures"
ATTN_KEY_RE = re.compile(r"^attn_comm_l(\d+)_to(\d+)_from(\d+)$")


@dataclass(frozen=True)
class Run:
    run_id: str
    map_name: str
    name: str
    seed: str
    status: str
    info: dict


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sort_key(path):
    return (0, int(path.name)) if path.name.isdigit() else (1, path.name)


def _as_float(value) -> float:
    if isinstance(value, dict) and "value" in value:
        return float(value["value"])
    return float(value)


def load_runs(sacred_dir, configs):
    runs = []
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
        name = config.get("name")
        if name not in configs:
            continue

        info = load_json(info_path)
        if not any(ATTN_KEY_RE.match(key) for key in info):
            continue

        run_meta = load_json(run_path) if run_path.exists() else {}
        status = run_meta.get("status", "")
        if status and status != "COMPLETED":
            continue

        map_name = config.get("env_args", {}).get("map_name")
        seed = str(config.get("seed"))
        if not map_name or seed == "None":
            continue

        runs.append(Run(
            run_id=run_dir.name,
            map_name=map_name,
            name=name,
            seed=seed,
            status=status,
            info=info,
        ))
    return runs


def run_attention_values(run):
    values = {}
    for key, raw_values in run.info.items():
        match = ATTN_KEY_RE.match(key)
        if not match or not raw_values:
            continue
        numeric = np.asarray([_as_float(value) for value in raw_values], dtype=float)
        layer = int(match.group(1))
        target = int(match.group(2))
        source = int(match.group(3))
        values[(layer, target, source)] = float(np.mean(numeric))
    return values


def aggregate_by_group(runs):
    grouped = {}
    for run in runs:
        values = run_attention_values(run)
        if not values:
            continue
        group = grouped.setdefault((run.map_name, run.name), {})
        for key, value in values.items():
            group.setdefault(key, []).append(value)
    return grouped


def matrices_from_values(values):
    max_layer = max(layer for layer, _, _ in values)
    max_target = max(target for _, target, _ in values)
    max_source = max(source for _, _, source in values)
    matrices = {}
    for layer in range(max_layer + 1):
        matrix = np.full((max_target + 1, max_source + 1), np.nan, dtype=float)
        for (layer_idx, target, source), entries in values.items():
            if layer_idx != layer:
                continue
            matrix[target, source] = float(np.mean(entries))
        matrices[layer] = matrix
    return matrices


def write_csv(path, map_name, config, layer, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["map", "config", "layer", "target_agent", "source_agent", "mean_weight"])
        for target in range(matrix.shape[0]):
            for source in range(matrix.shape[1]):
                value = matrix[target, source]
                if np.isnan(value):
                    continue
                writer.writerow([map_name, config, layer, target, source, "{:.8f}".format(float(value))])


def plot_heatmap(map_name, config, layer, matrix, output_dir):
    masked = np.ma.masked_invalid(matrix)

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    image = ax.imshow(masked, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("{} {} L{}".format(map_name, config, layer))
    ax.set_xlabel("Source agent")
    ax.set_ylabel("Target agent")
    ax.set_xticks(range(matrix.shape[1]), ["from{}".format(source) for source in range(matrix.shape[1])])
    ax.set_yticks(range(matrix.shape[0]), ["to{}".format(target) for target in range(matrix.shape[0])])

    for target in range(matrix.shape[0]):
        for source in range(matrix.shape[1]):
            value = matrix[target, source]
            if np.isnan(value):
                continue
            ax.text(
                source,
                target,
                "{:.2f}".format(float(value)),
                ha="center",
                va="center",
                color="white" if value < 0.65 else "black",
                fontsize=8,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean attention weight")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_map = map_name.replace("/", "_")
    out_path = output_dir / "{}_{}_l{}_attncomm_attention_heatmap.pdf".format(safe_map, config, layer)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Wrote {}".format(out_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot AttnComm agent-to-agent attention heatmaps.")
    parser.add_argument("--sacred-dir", default=str(DEFAULT_SACRED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument(
        "--configs",
        default="qmix_attncomm_l2_other,qmix_attncomm_l2_self",
        help="Comma-separated Sacred config names to include.",
    )
    args = parser.parse_args(argv)

    sacred_dir = Path(args.sacred_dir)
    output_dir = Path(args.output_dir)
    configs = {item.strip() for item in args.configs.split(",") if item.strip()}
    runs = load_runs(sacred_dir, configs)
    grouped = aggregate_by_group(runs)

    if not grouped:
        print("No communication attention logs found for configs: {}".format(",".join(sorted(configs))))
        return

    csv_path = output_dir / "attncomm_attention_weights.csv"
    if csv_path.exists():
        csv_path.unlink()

    for (map_name, config), values in sorted(grouped.items()):
        for layer, matrix in matrices_from_values(values).items():
            plot_heatmap(map_name, config, layer, matrix, output_dir)
            write_csv(csv_path, map_name, config, layer, matrix)
    print("Wrote {}".format(csv_path))


if __name__ == "__main__":
    main()
