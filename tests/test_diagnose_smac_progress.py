import json
import os
import sys
import tempfile
import unittest


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import diagnose_smac_progress


class DiagnoseSmacProgressTest(unittest.TestCase):
    def write_run(self, sacred_dir, run_id, name, seed, status, steps, wins, returns):
        run_dir = os.path.join(sacred_dir, str(run_id))
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({
                "name": name,
                "seed": seed,
                "env_args": {"map_name": "5m_vs_6m"},
            }, f)
        with open(os.path.join(run_dir, "run.json"), "w") as f:
            json.dump({"status": status}, f)
        with open(os.path.join(run_dir, "info.json"), "w") as f:
            json.dump({
                "test_battle_won_mean_T": steps,
                "test_battle_won_mean": wins,
                "test_return_mean_T": steps,
                "test_return_mean": returns,
            }, f)

    def test_diagnosis_flags_large_baseline_gap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sacred_dir = os.path.join(tmpdir, "sacred")
            output_dir = os.path.join(tmpdir, "diagnostics")
            os.makedirs(sacred_dir)
            self.write_run(sacred_dir, 1, "qmix", 1, "RUNNING", [100, 200], [0.2, 0.8], [5, 15])
            self.write_run(sacred_dir, 2, "qmix_attnres", 1, "RUNNING", [100, 180], [0.1, 0.4], [4, 10])

            diagnose_smac_progress.main([
                "--sacred-dir", sacred_dir,
                "--output-dir", output_dir,
                "--map-name", "5m_vs_6m",
                "--gap-threshold", "0.25",
            ])

            path = os.path.join(output_dir, "5m_vs_6m_qmix_vs_qmix_attnres_diagnosis.csv")
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                text = f.read()
            self.assertIn("baseline_much_better_run_ablation", text)
            self.assertIn("0.28", text)


if __name__ == "__main__":
    unittest.main()
