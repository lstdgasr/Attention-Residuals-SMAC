import json
import os
import sys
import tempfile
import unittest


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import plot_smac_results


class PlotSmacResultsTest(unittest.TestCase):
    def write_run(self, sacred_dir, run_id, name, seed, wins, returns):
        run_dir = os.path.join(sacred_dir, str(run_id))
        os.makedirs(run_dir)
        config = {
            "name": name,
            "seed": seed,
            "env_args": {"map_name": "2s3z"},
        }
        info = {
            "test_battle_won_mean_T": [0, 10, 20],
            "test_battle_won_mean": wins,
            "test_return_mean_T": [0, 10, 20],
            "test_return_mean": returns,
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f)
        with open(os.path.join(run_dir, "info.json"), "w") as f:
            json.dump(info, f)

    def test_plot_script_writes_figures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sacred_dir = os.path.join(tmpdir, "sacred")
            output_dir = os.path.join(tmpdir, "figures")
            os.makedirs(sacred_dir)
            self.write_run(sacred_dir, 1, "qmix", 1, [0.0, 0.5, 0.75], [1.0, 2.0, 3.0])
            self.write_run(sacred_dir, 2, "qmix_attnres", 1, [0.0, 0.6, 0.9], [1.0, 2.5, 3.5])

            plot_smac_results.main([
                "--sacred-dir", sacred_dir,
                "--output-dir", output_dir,
                "--map-name", "2s3z",
            ])

            expected = [
                "2s3z_test_battle_won_mean_curve.png",
                "2s3z_test_battle_won_mean_final_bar.png",
                "2s3z_test_return_mean_curve.png",
                "2s3z_test_return_mean_final_bar.png",
                "2s3z_qmix_vs_attnres_summary.csv",
            ]
            for filename in expected:
                self.assertTrue(os.path.exists(os.path.join(output_dir, filename)))

            with open(os.path.join(output_dir, "2s3z_qmix_vs_attnres_summary.csv")) as f:
                summary = f.read()
            self.assertIn("t_to_0_5_mean", summary)
            self.assertIn("qmix_attnres", summary)


if __name__ == "__main__":
    unittest.main()
