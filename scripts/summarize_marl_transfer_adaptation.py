#!/usr/bin/env python
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


PRIMARY_MAPS = ["5m_vs_6m", "3s5z"]
PRIMARY_CONFIGS = ["qmix", "qmix_attnres", "qmix_attnres_l2", "qmix_attnres_block", "qmix_depth_mlp"]
CROSS_MAP = "5m_vs_6m"
CROSS_PAIRS = [
    ("iql", "iql_attnres_l2"),
    ("vdn", "vdn_attnres_l2"),
    ("qmix", "qmix_attnres_l2"),
]
SEEDS = ["1", "2", "3"]


def _as_float(value):
    if isinstance(value, dict) and "value" in value:
        return float(value["value"])
    return float(value)


def _sort_key(path):
    return (0, int(path.name)) if path.name.isdigit() else (1, path.name)


def _parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _wall_hours(run_meta):
    start = _parse_time(run_meta.get("start_time"))
    stop = _parse_time(run_meta.get("stop_time"))
    if not start or not stop:
        return ""
    return (stop - start).total_seconds() / 3600.0


def metric_series(run, metric):
    info = run["info"]
    if metric not in info:
        return None, None
    values = np.asarray([_as_float(v) for v in info[metric]], dtype=float)
    if values.size == 0:
        return None, None
    t_key = "{}_T".format(metric)
    if t_key in info and len(info[t_key]) == len(values):
        steps = np.asarray([_as_float(v) for v in info[t_key]], dtype=float)
    else:
        steps = np.arange(values.size, dtype=float)
    return steps, values


def normalized_auc(steps, values):
    if steps is None or values is None:
        return ""
    if len(steps) < 2 or steps[-1] <= steps[0]:
        return float(values[-1])
    return float(np.trapz(values, steps) / (steps[-1] - steps[0]))


def first_reach_t(steps, values, threshold):
    if steps is None or values is None:
        return ""
    hits = np.where(values >= threshold)[0]
    if hits.size == 0:
        return ""
    return float(steps[int(hits[0])])


def load_best_runs(sacred_dir):
    runs = {}
    sacred_dir = Path(sacred_dir)
    for run_dir in sorted(sacred_dir.iterdir(), key=_sort_key):
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
        map_name = config.get("env_args", {}).get("map_name")
        name = config.get("name")
        seed = str(config.get("seed"))
        if not map_name or not name:
            continue

        item = {
            "run_id": run_dir.name,
            "map": map_name,
            "name": name,
            "seed": seed,
            "status": run_meta.get("status", ""),
            "wall_hours": _wall_hours(run_meta),
            "info": info,
        }
        key = (map_name, name, seed)
        old = runs.get(key)
        if old is None or run_rank(item) > run_rank(old):
            runs[key] = item
    return runs


def run_rank(run):
    steps, values = metric_series(run, "test_battle_won_mean")
    n = len(values) if values is not None else 0
    completed = 1 if run.get("status") == "COMPLETED" else 0
    last_t = float(steps[-1]) if steps is not None and len(steps) else 0.0
    return completed, n, last_t


def summarize_run(run):
    steps, wins = metric_series(run, "test_battle_won_mean")
    ret_steps, returns = metric_series(run, "test_return_mean")
    return {
        "run_id": run["run_id"],
        "status": run["status"],
        "n_eval": len(wins) if wins is not None else 0,
        "last_t": float(steps[-1]) if steps is not None and len(steps) else "",
        "final_win": float(wins[-1]) if wins is not None and len(wins) else "",
        "best_win": float(wins.max()) if wins is not None and len(wins) else "",
        "win_auc": normalized_auc(steps, wins),
        "final_return": float(returns[-1]) if returns is not None and len(returns) else "",
        "best_return": float(returns.max()) if returns is not None and len(returns) else "",
        "return_auc": normalized_auc(ret_steps, returns),
        "win50_t": first_reach_t(steps, wins, 0.5),
        "win70_t": first_reach_t(steps, wins, 0.7),
        "wall_hours": run["wall_hours"],
    }


def mean(values):
    values = [float(v) for v in values if v not in ("", None)]
    return float(np.mean(values)) if values else ""


def primary_table(runs):
    rows = []
    for map_name in PRIMARY_MAPS:
        for config in PRIMARY_CONFIGS:
            summaries = []
            missing = []
            for seed in SEEDS:
                run = runs.get((map_name, config, seed))
                if not run:
                    missing.append(seed)
                    continue
                summary = summarize_run(run)
                summary["seed"] = seed
                summaries.append(summary)
                if run["status"] != "COMPLETED":
                    missing.append("{}({})".format(seed, run["status"]))

            rows.append({
                "map": map_name,
                "config": config,
                "available_seeds": len(summaries),
                "complete_seeds": sum(1 for s in summaries if s["status"] == "COMPLETED"),
                "missing_or_partial": ";".join(missing),
                "final_win_mean": mean([s["final_win"] for s in summaries]),
                "best_win_mean": mean([s["best_win"] for s in summaries]),
                "win_auc_mean": mean([s["win_auc"] for s in summaries]),
                "final_return_mean": mean([s["final_return"] for s in summaries]),
                "wall_hours_mean": mean([s["wall_hours"] for s in summaries]),
            })
    return rows


def cross_algorithm_table(runs):
    rows = []
    for baseline, candidate in CROSS_PAIRS:
        for seed in SEEDS:
            base = runs.get((CROSS_MAP, baseline, seed))
            cand = runs.get((CROSS_MAP, candidate, seed))
            row = {
                "map": CROSS_MAP,
                "baseline": baseline,
                "candidate": candidate,
                "seed": seed,
                "baseline_run_id": base["run_id"] if base else "",
                "candidate_run_id": cand["run_id"] if cand else "",
                "baseline_status": base["status"] if base else "",
                "candidate_status": cand["status"] if cand else "",
            }
            if not base or not cand:
                row["decision"] = "missing_pair"
                rows.append(row)
                continue
            b = summarize_run(base)
            c = summarize_run(cand)
            row.update({
                "baseline_final_win": b["final_win"],
                "candidate_final_win": c["final_win"],
                "delta_final_win": _delta(c["final_win"], b["final_win"]),
                "baseline_best_win": b["best_win"],
                "candidate_best_win": c["best_win"],
                "delta_best_win": _delta(c["best_win"], b["best_win"]),
                "baseline_win_auc": b["win_auc"],
                "candidate_win_auc": c["win_auc"],
                "delta_win_auc": _delta(c["win_auc"], b["win_auc"]),
                "baseline_final_return": b["final_return"],
                "candidate_final_return": c["final_return"],
                "delta_final_return": _delta(c["final_return"], b["final_return"]),
                "baseline_wall_hours": b["wall_hours"],
                "candidate_wall_hours": c["wall_hours"],
                "wall_time_ratio": _ratio(c["wall_hours"], b["wall_hours"]),
            })
            row["decision"] = pair_decision(row)
            rows.append(row)
    return rows


def cross_algorithm_aggregate(pair_rows):
    rows = []
    groups = {}
    for row in pair_rows:
        if row.get("decision") == "missing_pair":
            continue
        key = (row["map"], row["baseline"], row["candidate"])
        groups.setdefault(key, []).append(row)
    for (map_name, baseline, candidate), group in sorted(groups.items()):
        rows.append({
            "map": map_name,
            "baseline": baseline,
            "candidate": candidate,
            "paired_seeds": len(group),
            "final_win_wins": sum(1 for r in group if float(r.get("delta_final_win", 0)) > 0),
            "auc_wins": sum(1 for r in group if float(r.get("delta_win_auc", 0)) > 0),
            "mean_delta_final_win": mean([r.get("delta_final_win") for r in group]),
            "mean_delta_best_win": mean([r.get("delta_best_win") for r in group]),
            "mean_delta_win_auc": mean([r.get("delta_win_auc") for r in group]),
            "mean_wall_time_ratio": mean([r.get("wall_time_ratio") for r in group]),
            "paper_reading": reading_for_group(group),
        })
    return rows


def missing_table(runs):
    rows = []
    expected = []
    for map_name in PRIMARY_MAPS:
        for config in PRIMARY_CONFIGS:
            for seed in SEEDS:
                expected.append((map_name, config, seed, "primary"))
    for baseline, candidate in CROSS_PAIRS:
        for seed in SEEDS:
            expected.append((CROSS_MAP, baseline, seed, "cross"))
            expected.append((CROSS_MAP, candidate, seed, "cross"))

    seen = set()
    for map_name, config, seed, group in expected:
        key = (map_name, config, seed)
        if (key, group) in seen:
            continue
        seen.add((key, group))
        run = runs.get(key)
        if not run:
            rows.append({"group": group, "map": map_name, "config": config, "seed": seed, "issue": "missing"})
            continue
        if run["status"] != "COMPLETED":
            summary = summarize_run(run)
            rows.append({
                "group": group,
                "map": map_name,
                "config": config,
                "seed": seed,
                "issue": run["status"],
                "run_id": run["run_id"],
                "n_eval": summary["n_eval"],
                "last_t": summary["last_t"],
                "final_win": summary["final_win"],
            })
    return rows


def _delta(a, b):
    if a in ("", None) or b in ("", None):
        return ""
    return float(a) - float(b)


def _ratio(a, b):
    if a in ("", None) or b in ("", None) or float(b) <= 0:
        return ""
    return float(a) / float(b)


def pair_decision(row):
    if row.get("candidate_status") != "COMPLETED" or row.get("baseline_status") != "COMPLETED":
        return "partial_pair"
    if float(row.get("delta_final_win", 0)) > 0 and float(row.get("delta_win_auc", 0)) >= 0:
        return "candidate_better"
    if float(row.get("delta_final_win", 0)) < 0 and float(row.get("delta_win_auc", 0)) < 0:
        return "candidate_worse"
    return "mixed"


def reading_for_group(group):
    complete = [r for r in group if r.get("decision") != "partial_pair"]
    usable = complete if complete else group
    final_wins = sum(1 for r in usable if float(r.get("delta_final_win", 0)) > 0)
    auc_wins = sum(1 for r in usable if float(r.get("delta_win_auc", 0)) > 0)
    if len(complete) < 3:
        return "needs_more_complete_seeds"
    if final_wins >= 2 and auc_wins >= 2:
        return "supports_lightweight_transfer"
    if final_wins == 0 and auc_wins == 0:
        return "argues_against_transfer"
    return "mixed_or_unstable"


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_compact(primary, cross_agg, missing):
    print("Primary QMIX table")
    for row in primary:
        print("{map} {config} complete={complete_seeds}/3 final={final_win_mean} auc={win_auc_mean} missing={missing_or_partial}".format(**row))
    print()
    print("Cross-algorithm transfer table")
    for row in cross_agg:
        print("{baseline}->{candidate} paired={paired_seeds} final_delta={mean_delta_final_win} auc_delta={mean_delta_win_auc} reading={paper_reading}".format(**row))
    print()
    if missing:
        print("Missing or partial slots")
        for row in missing:
            print(row)
    else:
        print("No missing or partial expected slots.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build paper-oriented tables for the LLM-to-MARL AttnRes adaptation study.")
    parser.add_argument("--sacred-dir", default="results/sacred")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    runs = load_best_runs(args.sacred_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.sacred_dir).parent / "diagnostics"

    primary = primary_table(runs)
    cross_pairs = cross_algorithm_table(runs)
    cross_agg = cross_algorithm_aggregate(cross_pairs)
    missing = missing_table(runs)

    write_csv(output_dir / "marl_transfer_primary_qmix_table.csv", primary)
    write_csv(output_dir / "marl_transfer_cross_algorithm_pairs.csv", cross_pairs)
    write_csv(output_dir / "marl_transfer_cross_algorithm_aggregate.csv", cross_agg)
    write_csv(output_dir / "marl_transfer_missing_or_partial.csv", missing)

    print_compact(primary, cross_agg, missing)
    print("Wrote MARL transfer adaptation summaries under: {}".format(output_dir))


if __name__ == "__main__":
    main()
