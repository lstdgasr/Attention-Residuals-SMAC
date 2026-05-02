#!/usr/bin/env python
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def _as_float(value):
    if isinstance(value, dict) and "value" in value:
        return float(value["value"])
    return float(value)


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


def _sort_key(path):
    return (0, int(path.name)) if path.name.isdigit() else (1, path.name)


def load_runs(sacred_dir, maps, algorithms):
    sacred_dir = Path(sacred_dir)
    runs = []
    for run_dir in sorted(sacred_dir.iterdir(), key=_sort_key):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        info_path = run_dir / "info.json"
        run_path = run_dir / "run.json"
        if not config_path.exists() or not info_path.exists():
            continue
        config = json.loads(config_path.read_text())
        map_name = config.get("env_args", {}).get("map_name")
        name = config.get("name")
        if map_name not in maps or name not in algorithms:
            continue
        run_meta = json.loads(run_path.read_text()) if run_path.exists() else {}
        runs.append({
            "run_id": run_dir.name,
            "name": name,
            "map": map_name,
            "seed": str(config.get("seed")),
            "status": run_meta.get("status", ""),
            "wall_hours": _wall_hours(run_meta),
            "info": json.loads(info_path.read_text()),
        })
    return runs


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


def clip_series(steps, values, end_t):
    if end_t <= steps[0]:
        return np.asarray([end_t]), np.asarray([float(values[0])])
    mask = steps <= end_t
    clipped_steps = steps[mask]
    clipped_values = values[mask]
    if clipped_steps.size == 0 or clipped_steps[-1] < end_t:
        clipped_steps = np.append(clipped_steps, end_t)
        clipped_values = np.append(clipped_values, np.interp(end_t, steps, values))
    return clipped_steps, clipped_values


def value_at(steps, values, target_t):
    return float(np.interp(target_t, steps, values))


def normalized_auc(steps, values, end_t):
    c_steps, c_values = clip_series(steps, values, end_t)
    span = float(c_steps[-1] - c_steps[0])
    if span <= 0:
        return float(c_values[-1])
    return float(np.trapezoid(c_values, c_steps) / span)


def first_reach_t(steps, values, threshold):
    hits = np.where(values >= threshold)[0]
    if hits.size == 0:
        return ""
    return float(steps[int(hits[0])])


def run_summary(run, t_max):
    row = {
        "run_id": run["run_id"],
        "map": run["map"],
        "name": run["name"],
        "seed": run["seed"],
        "status": run["status"],
        "wall_hours": run["wall_hours"],
    }
    for metric in ("test_battle_won_mean", "test_return_mean"):
        steps, values = metric_series(run, metric)
        if steps is None:
            continue
        end_t = min(float(steps[-1]), float(t_max))
        row["{}_last_t".format(metric)] = float(steps[-1])
        row["{}_final".format(metric)] = float(values[-1])
        row["{}_best".format(metric)] = float(values.max())
        row["{}_auc".format(metric)] = normalized_auc(steps, values, end_t)
    steps, wins = metric_series(run, "test_battle_won_mean")
    if steps is not None:
        row["win50_t"] = first_reach_t(steps, wins, 0.5)
        row["win70_t"] = first_reach_t(steps, wins, 0.7)
    return row


def paired_rows(runs, baseline, candidates, t_max):
    by_key = {(run["map"], run["name"], run["seed"]): run for run in runs}
    seeds = sorted({run["seed"] for run in runs}, key=lambda s: int(s) if s.isdigit() else s)
    maps = sorted({run["map"] for run in runs})
    rows = []

    for map_name in maps:
        for candidate in candidates:
            for seed in seeds:
                base = by_key.get((map_name, baseline, seed))
                cand = by_key.get((map_name, candidate, seed))
                row = {
                    "map": map_name,
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

                b_steps, b_wins = metric_series(base, "test_battle_won_mean")
                c_steps, c_wins = metric_series(cand, "test_battle_won_mean")
                if b_steps is None or c_steps is None:
                    row["decision"] = "missing_metric"
                    rows.append(row)
                    continue

                common_t = float(min(b_steps[-1], c_steps[-1], t_max))
                b_final = value_at(b_steps, b_wins, common_t)
                c_final = value_at(c_steps, c_wins, common_t)
                _, b_clip = clip_series(b_steps, b_wins, common_t)
                _, c_clip = clip_series(c_steps, c_wins, common_t)

                row.update({
                    "common_t": common_t,
                    "complete_pair": _is_complete_pair(base, cand, common_t, t_max),
                    "baseline_win_final": b_final,
                    "candidate_win_final": c_final,
                    "candidate_minus_baseline_win_final": c_final - b_final,
                    "baseline_win_best": float(b_clip.max()),
                    "candidate_win_best": float(c_clip.max()),
                    "candidate_minus_baseline_win_best": float(c_clip.max() - b_clip.max()),
                    "baseline_win_auc": normalized_auc(b_steps, b_wins, common_t),
                    "candidate_win_auc": normalized_auc(c_steps, c_wins, common_t),
                    "candidate_minus_baseline_win_auc": (
                        normalized_auc(c_steps, c_wins, common_t) - normalized_auc(b_steps, b_wins, common_t)
                    ),
                    "baseline_win50_t": first_reach_t(b_steps, b_wins, 0.5),
                    "candidate_win50_t": first_reach_t(c_steps, c_wins, 0.5),
                    "baseline_win70_t": first_reach_t(b_steps, b_wins, 0.7),
                    "candidate_win70_t": first_reach_t(c_steps, c_wins, 0.7),
                    "baseline_wall_hours": base["wall_hours"],
                    "candidate_wall_hours": cand["wall_hours"],
                    "wall_time_ratio": _ratio(cand["wall_hours"], base["wall_hours"]),
                })

                b_ret_steps, b_rets = metric_series(base, "test_return_mean")
                c_ret_steps, c_rets = metric_series(cand, "test_return_mean")
                if b_ret_steps is not None and c_ret_steps is not None:
                    row["baseline_return_final"] = value_at(b_ret_steps, b_rets, common_t)
                    row["candidate_return_final"] = value_at(c_ret_steps, c_rets, common_t)
                    row["candidate_minus_baseline_return_final"] = (
                        row["candidate_return_final"] - row["baseline_return_final"]
                    )

                if not row["complete_pair"]:
                    row["decision"] = "continue_to_t_max"
                elif row["candidate_minus_baseline_win_final"] > 0 and row["candidate_minus_baseline_win_auc"] >= 0:
                    row["decision"] = "candidate_better"
                elif row["candidate_minus_baseline_win_best"] > 0 and row["candidate_minus_baseline_win_auc"] >= 0:
                    row["decision"] = "candidate_promising"
                elif row["candidate_minus_baseline_win_final"] < -0.1 and row["candidate_minus_baseline_win_auc"] < 0:
                    row["decision"] = "candidate_worse"
                else:
                    row["decision"] = "mixed_or_tied"
                rows.append(row)
    return rows


def aggregate_rows(pairs, min_completed_seeds, wall_ratio_limit):
    grouped = {}
    for row in pairs:
        if row.get("decision") in ("missing_pair", "missing_metric"):
            continue
        key = (row["map"], row["candidate"])
        grouped.setdefault(key, []).append(row)

    rows = []
    for (map_name, candidate), group in sorted(grouped.items()):
        complete = [r for r in group if r.get("complete_pair") is True]
        usable = complete if complete else group
        final_wins = sum(1 for r in usable if float(r.get("candidate_minus_baseline_win_final", 0)) > 0)
        best_wins = sum(1 for r in usable if float(r.get("candidate_minus_baseline_win_best", 0)) > 0)
        auc_wins = sum(1 for r in usable if float(r.get("candidate_minus_baseline_win_auc", 0)) > 0)
        ratios = [float(r["wall_time_ratio"]) for r in usable if r.get("wall_time_ratio") not in ("", None)]
        mean_ratio = float(np.mean(ratios)) if ratios else ""
        mean_final_delta = float(np.mean([float(r.get("candidate_minus_baseline_win_final", 0)) for r in usable]))
        mean_auc_delta = float(np.mean([float(r.get("candidate_minus_baseline_win_auc", 0)) for r in usable]))

        required_wins = min(2, max(1, min_completed_seeds))

        if len(complete) < min_completed_seeds:
            decision = "collect_more_seeds"
        elif final_wins >= required_wins and (auc_wins >= required_wins or best_wins >= required_wins):
            decision = "continue_candidate"
        elif mean_ratio != "" and mean_ratio > wall_ratio_limit and final_wins < required_wins and auc_wins < required_wins:
            decision = "stop_loss_candidate"
        elif final_wins == 0 and auc_wins == 0:
            decision = "stop_loss_candidate"
        else:
            decision = "mixed_unstable"

        rows.append({
            "map": map_name,
            "candidate": candidate,
            "paired_seeds": len(group),
            "completed_seeds": len(complete),
            "final_wins": final_wins,
            "best_wins": best_wins,
            "auc_wins": auc_wins,
            "mean_final_win_delta": mean_final_delta,
            "mean_auc_delta": mean_auc_delta,
            "mean_wall_time_ratio": mean_ratio,
            "decision": decision,
        })
    return rows


def _ratio(num, den):
    if num in ("", None) or den in ("", None):
        return ""
    den = float(den)
    if den <= 0:
        return ""
    return float(num) / den


def _is_complete_pair(base, cand, common_t, t_max):
    if base.get("status") == "COMPLETED" and cand.get("status") == "COMPLETED":
        return True
    return common_t >= float(t_max) * 0.95


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


def print_decisions(rows):
    if not rows:
        print("No aggregate decisions available.")
        return
    print("map candidate completed final_wins auc_wins mean_final_delta mean_auc_delta wall_ratio decision")
    for row in rows:
        print("{} {} {} {} {} {:.4f} {:.4f} {} {}".format(
            row["map"],
            row["candidate"],
            row["completed_seeds"],
            row["final_wins"],
            row["auc_wins"],
            row["mean_final_win_delta"],
            row["mean_auc_delta"],
            "" if row["mean_wall_time_ratio"] == "" else "{:.2f}".format(row["mean_wall_time_ratio"]),
            row["decision"],
        ))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize AttnRes-SC2 diagnostic runs by paired seed.")
    parser.add_argument("--sacred-dir", default="results/sacred")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--maps", nargs="+", default=["5m_vs_6m", "3s5z"])
    parser.add_argument("--baseline", default="qmix")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["qmix_attnres", "qmix_attnres_l2", "qmix_attnres_block", "qmix_depth_mlp"],
    )
    parser.add_argument("--t-max", type=float, default=2050000)
    parser.add_argument("--min-completed-seeds", type=int, default=3)
    parser.add_argument("--wall-ratio-limit", type=float, default=1.5)
    args = parser.parse_args(argv)

    algorithms = set([args.baseline] + args.candidates)
    runs = load_runs(args.sacred_dir, set(args.maps), algorithms)
    if not runs:
        raise SystemExit("No matching runs found in {}".format(args.sacred_dir))

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.sacred_dir).parent / "diagnostics"
    run_rows = [run_summary(run, args.t_max) for run in runs]
    pair_rows = paired_rows(runs, args.baseline, args.candidates, args.t_max)
    aggregate = aggregate_rows(pair_rows, args.min_completed_seeds, args.wall_ratio_limit)

    write_csv(output_dir / "attnres_sc2_run_summary.csv", run_rows)
    write_csv(output_dir / "attnres_sc2_paired_summary.csv", pair_rows)
    write_csv(output_dir / "attnres_sc2_decisions.csv", aggregate)

    print_decisions(aggregate)
    print("Wrote summaries under: {}".format(output_dir))


if __name__ == "__main__":
    main()
