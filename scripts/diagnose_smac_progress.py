#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _as_float(value):
    if isinstance(value, dict) and "value" in value:
        return float(value["value"])
    return float(value)


def load_runs(sacred_dir, map_name, algorithms):
    runs = []
    sacred_dir = Path(sacred_dir)
    if not sacred_dir.exists():
        raise SystemExit("Sacred directory does not exist: {}".format(sacred_dir))

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
        name = config.get("name")
        run_map = config.get("env_args", {}).get("map_name")
        if run_map != map_name or name not in algorithms:
            continue

        info = json.loads(info_path.read_text())
        run_meta = json.loads(run_path.read_text()) if run_path.exists() else {}
        runs.append({
            "run_id": run_dir.name,
            "name": name,
            "seed": config.get("seed"),
            "status": run_meta.get("status", ""),
            "info": info,
        })
    return runs


def metric_series(run, metric):
    info = run["info"]
    t_key = "{}_T".format(metric)
    if metric not in info or t_key not in info:
        return None, None
    values = np.asarray([_as_float(v) for v in info[metric]], dtype=float)
    steps = np.asarray([_as_float(v) for v in info[t_key]], dtype=float)
    if len(values) == 0 or len(steps) == 0:
        return None, None
    return steps, values


def value_at_t(steps, values, target_t):
    return float(np.interp(target_t, steps, values))


def latest_metrics(run):
    out = {}
    for metric in ("test_battle_won_mean", "test_return_mean"):
        steps, values = metric_series(run, metric)
        if steps is None:
            out[metric + "_last_t"] = ""
            out[metric + "_last"] = ""
            out[metric + "_best"] = ""
            continue
        out[metric + "_last_t"] = float(steps[-1])
        out[metric + "_last"] = float(values[-1])
        out[metric + "_best"] = float(values.max())
    return out


def diagnose_pairs(runs, baseline, candidate, gap_threshold, t_max):
    by_name_seed = {}
    for run in runs:
        by_name_seed[(run["name"], str(run["seed"]))] = run

    rows = []
    seeds = sorted({str(run["seed"]) for run in runs}, key=lambda x: int(x) if x.isdigit() else x)
    for seed in seeds:
        base_run = by_name_seed.get((baseline, seed))
        cand_run = by_name_seed.get((candidate, seed))

        row = {
            "seed": seed,
            "baseline_run_id": base_run["run_id"] if base_run else "",
            "candidate_run_id": cand_run["run_id"] if cand_run else "",
            "baseline_status": base_run["status"] if base_run else "",
            "candidate_status": cand_run["status"] if cand_run else "",
        }

        if not base_run or not cand_run:
            row["decision"] = "missing_pair"
            rows.append(row)
            continue

        base_steps, base_wins = metric_series(base_run, "test_battle_won_mean")
        cand_steps, cand_wins = metric_series(cand_run, "test_battle_won_mean")
        if base_steps is None or cand_steps is None:
            row["decision"] = "missing_metric"
            rows.append(row)
            continue

        common_t = float(min(base_steps[-1], cand_steps[-1], t_max))
        base_win = value_at_t(base_steps, base_wins, common_t)
        cand_win = value_at_t(cand_steps, cand_wins, common_t)
        gap = base_win - cand_win

        row.update({
            "common_t": common_t,
            "baseline_win_at_common_t": base_win,
            "candidate_win_at_common_t": cand_win,
            "baseline_minus_candidate_win": gap,
        })

        base_ret_steps, base_rets = metric_series(base_run, "test_return_mean")
        cand_ret_steps, cand_rets = metric_series(cand_run, "test_return_mean")
        if base_ret_steps is not None and cand_ret_steps is not None:
            row["baseline_return_at_common_t"] = value_at_t(base_ret_steps, base_rets, common_t)
            row["candidate_return_at_common_t"] = value_at_t(cand_ret_steps, cand_rets, common_t)
            row["baseline_minus_candidate_return"] = (
                row["baseline_return_at_common_t"] - row["candidate_return_at_common_t"]
            )

        row.update({"baseline_" + k: v for k, v in latest_metrics(base_run).items()})
        row.update({"candidate_" + k: v for k, v in latest_metrics(cand_run).items()})

        if gap >= gap_threshold:
            decision = "baseline_much_better_run_ablation"
        elif common_t < t_max:
            decision = "continue_to_t_max"
        elif gap >= 0.1:
            decision = "candidate_failed_on_final"
        else:
            decision = "roughly_comparable"
        row["decision"] = decision
        rows.append(row)
    return rows


def write_csv(rows, output_path):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(output_path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    if not rows:
        print("No matching paired runs found.")
        return

    print("seed common_t baseline_win candidate_win gap decision")
    for row in rows:
        print("{} {} {} {} {} {}".format(
            row.get("seed", ""),
            int(row["common_t"]) if row.get("common_t") != "" and "common_t" in row else "",
            _fmt(row.get("baseline_win_at_common_t", "")),
            _fmt(row.get("candidate_win_at_common_t", "")),
            _fmt(row.get("baseline_minus_candidate_win", "")),
            row.get("decision", ""),
        ))


def _fmt(value):
    if value == "" or value is None:
        return ""
    return "{:.4f}".format(float(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose SMAC baseline vs candidate progress at common timesteps.")
    parser.add_argument("--sacred-dir", default="results/sacred")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--map-name", default="5m_vs_6m")
    parser.add_argument("--baseline", default="qmix")
    parser.add_argument("--candidate", default="qmix_attnres")
    parser.add_argument("--gap-threshold", type=float, default=0.25)
    parser.add_argument("--t-max", type=float, default=2050000)
    args = parser.parse_args(argv)

    runs = load_runs(args.sacred_dir, args.map_name, {args.baseline, args.candidate})
    rows = diagnose_pairs(runs, args.baseline, args.candidate, args.gap_threshold, args.t_max)
    if not rows:
        raise SystemExit("No matching runs found for map {}".format(args.map_name))

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.sacred_dir).parent / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "{}_{}_vs_{}_diagnosis.csv".format(args.map_name, args.baseline, args.candidate)
    write_csv(rows, output_path)
    print_summary(rows)
    print("Wrote diagnosis: {}".format(output_path))


if __name__ == "__main__":
    main()
