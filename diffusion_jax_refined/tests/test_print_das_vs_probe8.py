from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PrintDasVsProbe8Tests(unittest.TestCase):
    def test_printer_uses_expected_algorithms_and_signs(self):
        text = (
            ROOT / "3dshapes" / "script" / "print_das_vs_probe8_both_l2.py"
        ).read_text()
        self.assertIn("--das-lambda", text)
        self.assertIn("default=200.0", text)
        self.assertIn(
            '"traj_tracin_predicted_noise_jvp_final_linear_mean_probe8_query_train_l2"',
            text,
        )
        self.assertIn("pred_kept_sign_m1", text)
        self.assertIn("pred_kept_sign_p1", text)
        self.assertIn("Probe8 wins", text)


if __name__ == "__main__":
    unittest.main()
