"""
Unit & Integration Tests for rsl_model (`train_compare.py` & `predict_rsl.py`).
Run with: python -m unittest test_predict_rsl.py
"""

import os
import tempfile
import unittest
from train_compare import generate_demo_dataset, run_benchmark
from predict_rsl import predict_rsl, predict_rsl_batch


class TestRSLPipeline(unittest.TestCase):
    def test_end_to_end_training_and_inference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = generate_demo_dataset(n_lots=30, rows_per_lot=90, seed=7)
            summary = run_benchmark(df, output_dir=tmpdir)

            self.assertIn("K1_Linear", summary["models"])
            self.assertIn("K2_Kinetic", summary["models"])
            self.assertIn("R1_Ridge", summary["models"])
            self.assertIn("ML_Gradient_Boosting", summary["models"])

            artifact_path = os.path.join(tmpdir, "rsl_gb_artifact.joblib")
            self.assertTrue(os.path.exists(artifact_path))

            sample_payload = {
                "su_id": "SU_0001_010",
                "lot_id": "LOT_0001",
                "equiv_age_at_4c_days": 5.02,
                "age_h": 100.0,
                "meas_mean_c": 4.4,
                "meas_max_c": 5.2,
                "meas_dh_above_4": 8.0,
                "checkpoint": "CP3_HUB",
                "store": "STORE_B",
            }
            res = predict_rsl(sample_payload, artifact_path=artifact_path)
            self.assertGreaterEqual(res["rsl_pred_days"], 0.0)
            self.assertLessEqual(res["rsl_lower_p10_days"], res["rsl_pred_days"])
            self.assertGreaterEqual(res["rsl_upper_p90_days"], res["rsl_pred_days"])
            self.assertIn(res["risk_status"], {"SAFE", "WARNING", "CRITICAL"})

            batch_res = predict_rsl_batch(
                [sample_payload, {"equiv_age_at_4c_days": 14.5}],
                artifact_path=artifact_path,
            )
            self.assertEqual(len(batch_res), 2)
            self.assertEqual(batch_res[1]["risk_status"], "CRITICAL")


if __name__ == "__main__":
    unittest.main()
