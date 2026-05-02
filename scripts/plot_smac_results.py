#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_runs(sacred_dir, algorithms, map_name):
    sacred_dir = Path(sacred_dir)
    runs = []
    def run_sort_key(path):
        return (0, int(path.name)) if path.name.isdigit() else (1, path.name)

    for run_dir in sorted(sacred_dir.iterdir(), key=run_sort_key):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        info_path = run_dir / "info.json"
        run_path = run_dir / "run.json"
        if not config_path.exists() or not info_path.exists():
            continue

        config = json.loads(config_path.read_text())
        info = json.loads(info_path.read_text())
        run_meta = json.loads(run_path.read_text()) if run_path.exists() else {}
        name = config.get("name")
        run_map = config.get("env_args", {}).get("map_name")
        if algorithms and name not in algorithms:
            continue
        if map_name and run_map != map_name:
            continue
        runs.append({
            "run_id": run_dir.name,
            "name": name,
            "map_name": run_map,
            "seed": config.get("seed"),
            "status": run_meta.get("status"),
            "info": info,
        })
    return runs


def collect_series(runs, metric):
    grouped = {}
    t_key = "{}_T".format(metric)
    for run in runs:
        info = run["info"]
        if metric not in info or t_key not in info:
            continue
        values = np.asarray([_as_float(v) for v in info[metric]], dtype=float)
        steps = np.asarray([_as_float(v) for v in info[t_key]], dtype=float)
        if len(values) == 0 or len(steps) == 0:
            continue
        grouped.setdefault(run["name"], []).append((steps, values, run))
    return grouped


def _as_float(value):
    if isinstance(value, dict) and "value" in value:
        return float(value["value"])
    return float(value)


def aggregate_series(series):
    all_steps = np.unique(np.concatenate([steps for steps, _, _ in series]))
    interp_values = []
    for steps, values, _ in series:
        interp_values.append(np.interp(all_steps, steps, values))
    stacked = np.vstack(interp_values)
    return all_steps, stacked.mean(axis=0), stacked.std(axis=0), stacked[:, -1]


def first_threshold_step(steps, values, threshold):
    hits = np.where(values >= threshold)[0]
    if len(hits) == 0:
        return np.nan
    return float(steps[hits[0]])


def write_summary_csv(runs, metrics, thresholds, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "{}_summary.csv".format(prefix)
    rows = []

    for metric in metrics:
        grouped = collect_series(runs, metric)
        for name in sorted(grouped):
            finals = []
            bests = []
            last_steps = []
            threshold_values = {threshold: [] for threshold in thresholds}

            for steps, values, run in grouped[name]:
                finals.append(float(values[-1]))
                bests.append(float(values.max()))
                last_steps.append(float(steps[-1]))
                for threshold in thresholds:
                    threshold_values[threshold].append(first_threshold_step(steps, values, threshold))

            row = {
                "metric": metric,
                "algorithm": name,
                "n": len(grouped[name]),
                "seeds": ";".join(str(item[2].get("seed")) for item in grouped[name]),
                "run_ids": ";".join(str(item[2].get("run_id")) for item in grouped[name]),
                "last_t_mean": float(np.mean(last_steps)),
                "last_t_std": float(np.std(last_steps)),
                "final_mean": float(np.mean(finals)),
                "final_std": float(np.std(finals)),
                "best_mean": float(np.mean(bests)),
                "best_std": float(np.std(bests)),
            }
            for threshold in thresholds:
                values = np.asarray(threshold_values[threshold], dtype=float)
                finite = values[np.isfinite(values)]
                suffix = str(threshold).replace(".", "_")
                row["t_to_{}_mean".format(suffix)] = float(finite.mean()) if len(finite) else ""
                row["t_to_{}_std".format(suffix)] = float(finite.std()) if len(finite) else ""
                row["t_to_{}_n".format(suffix)] = int(len(finite))
            rows.append(row)

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def plot_metric(runs, metric, ylabel, output_dir, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = collect_series(runs, metric)
    if not grouped:
        raise ValueError("No runs contain metric {}".format(metric))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_labels = []
    final_means = []
    final_stds = []

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name in sorted(grouped):
        steps, mean, std, finals = aggregate_series(grouped[name])
        ax.plot(steps, mean, label="{} (n={})".format(name, len(grouped[name])), linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.18)
        final_labels.append(name)
        final_means.append(float(finals.mean()))
        final_stds.append(float(finals.std()))

    ax.set_title("{} on SMAC".format(ylabel))
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    curve_base = output_dir / "{}_curve".format(prefix)
    fig.savefig(str(curve_base) + ".png", dpi=180)
    fig.savefig(str(curve_base) + ".pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(final_labels))
    ax.bar(x, final_means, yerr=final_stds, capsize=5)
    ax.set_title("Final {}".format(ylabel))
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(final_labels, rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    bar_base = output_dir / "{}_final_bar".format(prefix)
    fig.savefig(str(bar_base) + ".png", dpi=180)
    fig.savefig(str(bar_base) + ".pdf")
    plt.close(fig)

    return [str(curve_base) + ".png", str(curve_base) + ".pdf",
            str(bar_base) + ".png", str(bar_base) + ".pdf"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot SMAC Sacred results for baseline vs AttnRes.")
    parser.add_argument("--sacred-dir", default="results/sacred",
                        help="Directory containing Sacred run subdirectories.")
    parser.add_argument("--output-dir", default=None,
                        help="Figure output directory. Defaults to <sacred parent>/figures.")
    parser.add_argument("--map-name", default="2s3z")
    parser.add_argument("--algorithms", nargs="+", default=["qmix", "qmix_attnres"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.8, 0.9],
                        help="Win-rate thresholds for sample-efficiency summary columns.")
    parser.add_argument("--completed-only", action="store_true",
                        help="Ignore Sacred runs whose run.json status is not COMPLETED.")
    args = parser.parse_args(argv)

    sacred_dir = Path(args.sacred_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sacred_dir.parent / "figures"
    runs = load_runs(sacred_dir, set(args.algorithms), args.map_name)
    if args.completed_only:
        runs = [run for run in runs if run.get("status") in (None, "COMPLETED")]
    if not runs:
        raise SystemExit("No matching runs found in {}".format(sacred_dir))

    written = []
    written.extend(plot_metric(runs, "test_battle_won_mean", "Test battle won mean", output_dir,
                               "{}_test_battle_won_mean".format(args.map_name)))
    written.extend(plot_metric(runs, "test_return_mean", "Test return mean", output_dir,
                               "{}_test_return_mean".format(args.map_name)))
    written.append(write_summary_csv(
        runs,
        ["test_battle_won_mean", "test_return_mean"],
        args.thresholds,
        output_dir,
        "{}_qmix_vs_attnres".format(args.map_name),
    ))
    print("Wrote outputs:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
