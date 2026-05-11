#!/usr/bin/env python
"""Plot AttnRes-L2 depth-source attention heatmaps from Sacred info.json."""

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
ATTN_KEY_RE = re.compile(r"^attn_res_l(\d+)_src(\d+)$")


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
        if not match:
            continue
        if not raw_values:
            continue
        numeric = np.asarray([_as_float(value) for value in raw_values], dtype=float)
        values[(int(match.group(1)), int(match.group(2)))] = float(np.mean(numeric))
    return values


def aggregate_by_map(runs):
    grouped = {}
    for run in runs:
        values = run_attention_values(run)
        if not values:
            continue
        map_group = grouped.setdefault(run.map_name, {})
        for key, value in values.items():
            map_group.setdefault(key, []).append(value)
    return grouped


def matrix_from_values(values):
    max_layer = max(layer for layer, _ in values)
    max_source = max(source for _, source in values)
    matrix = np.full((max_layer + 1, max_source + 1), np.nan, dtype=float)
    layers = list(range(max_layer + 1))
    sources = list(range(max_source + 1))
    for (layer, source), entries in values.items():
        matrix[layer, source] = float(np.mean(entries))
    return matrix, layers, sources


def write_csv(path, map_name, matrix, layers, sources):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["map", "layer", "source", "mean_weight"])
        for layer_idx, layer in enumerate(layers):
            for source_idx, source in enumerate(sources):
                value = matrix[layer_idx, source_idx]
                if np.isnan(value):
                    continue
                writer.writerow([map_name, layer, source, "{:.8f}".format(float(value))])


def plot_heatmap(map_name, values, output_dir):
    matrix, layers, sources = matrix_from_values(values)
    masked = np.ma.masked_invalid(matrix)

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    image = ax.imshow(masked, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("{} AttnRes-L2 Attention".format(map_name))
    ax.set_xlabel("Depth source")
    ax.set_ylabel("AttnRes layer")
    ax.set_xticks(range(len(sources)), ["src{}".format(source) for source in sources])
    ax.set_yticks(range(len(layers)), ["l{}".format(layer) for layer in layers])

    for layer_idx in range(matrix.shape[0]):
        for source_idx in range(matrix.shape[1]):
            value = matrix[layer_idx, source_idx]
            if np.isnan(value):
                continue
            ax.text(
                source_idx,
                layer_idx,
                "{:.2f}".format(float(value)),
                ha="center",
                va="center",
                color="white" if value < 0.65 else "black",
                fontsize=9,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean attention weight")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_map = map_name.replace("/", "_")
    out_path = output_dir / "{}_attnres_l2_attention_heatmap.pdf".format(safe_map)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Wrote {}".format(out_path))

    write_csv(output_dir / "attnres_l2_attention_weights.csv", map_name, matrix, layers, sources)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot AttnRes-L2 depth-source attention heatmaps.")
    parser.add_argument("--sacred-dir", default=str(DEFAULT_SACRED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument(
        "--configs",
        default="qmix_attnres_l2",
        help="Comma-separated Sacred config names to include.",
    )
    args = parser.parse_args(argv)

    sacred_dir = Path(args.sacred_dir)
    output_dir = Path(args.output_dir)
    configs = {item.strip() for item in args.configs.split(",") if item.strip()}
    runs = load_runs(sacred_dir, configs)
    grouped = aggregate_by_map(runs)

    if not grouped:
        print("No attention weight logs found for configs: {}".format(",".join(sorted(configs))))
        return

    csv_path = output_dir / "attnres_l2_attention_weights.csv"
    if csv_path.exists():
        csv_path.unlink()
    for map_name in sorted(grouped):
        plot_heatmap(map_name, grouped[map_name], output_dir)
    print("Wrote {}".format(csv_path))


if __name__ == "__main__":
    main()
