import json
import os
import sys
import tempfile
import unittest


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import summarize_attnres_sc2_diagnostics


class SummarizeAttnResSc2DiagnosticsTest(unittest.TestCase):
    def write_run(self, sacred_dir, run_id, name, seed, steps, wins, start, stop):
        run_dir = os.path.join(sacred_dir, str(run_id))
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({
                "name": name,
                "seed": seed,
                "env_args": {"map_name": "5m_vs_6m"},
            }, f)
        with open(os.path.join(run_dir, "run.json"), "w") as f:
            json.dump({
                "status": "COMPLETED",
                "start_time": start,
                "stop_time": stop,
            }, f)
        with open(os.path.join(run_dir, "info.json"), "w") as f:
            json.dump({
                "test_battle_won_mean_T": steps,
                "test_battle_won_mean": wins,
                "test_return_mean_T": steps,
                "test_return_mean": [v * 20 for v in wins],
            }, f)

    def test_summary_outputs_paired_and_decision_csvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sacred_dir = os.path.join(tmpdir, "sacred")
            output_dir = os.path.join(tmpdir, "diagnostics")
            os.makedirs(sacred_dir)

            self.write_run(
                sacred_dir, 1, "qmix", 1, [0, 100, 200], [0.0, 0.5, 0.6],
                "2026-04-26T00:00:00", "2026-04-26T01:00:00",
            )
            self.write_run(
                sacred_dir, 2, "qmix_attnres", 1, [0, 100, 200], [0.0, 0.6, 0.8],
                "2026-04-26T00:00:00", "2026-04-26T02:00:00",
            )

            summarize_attnres_sc2_diagnostics.main([
                "--sacred-dir", sacred_dir,
                "--output-dir", output_dir,
                "--maps", "5m_vs_6m",
                "--baseline", "qmix",
                "--candidates", "qmix_attnres",
                "--t-max", "200",
                "--min-completed-seeds", "1",
            ])

            paired_path = os.path.join(output_dir, "attnres_sc2_paired_summary.csv")
            decision_path = os.path.join(output_dir, "attnres_sc2_decisions.csv")
            self.assertTrue(os.path.exists(paired_path))
            self.assertTrue(os.path.exists(decision_path))
            with open(paired_path) as f:
                paired = f.read()
            with open(decision_path) as f:
                decisions = f.read()
            self.assertIn("candidate_better", paired)
            self.assertIn("continue_candidate", decisions)
            self.assertIn("2.0", paired)


if __name__ == "__main__":
    unittest.main()
