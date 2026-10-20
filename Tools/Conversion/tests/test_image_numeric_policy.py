import unittest

import numpy as np

from Tools.Conversion.adapters.qwen_image21.evaluation import canonical_hash
from Tools.Conversion.adapters.qwen_image21.numeric_policy import numeric_metrics, numeric_policy


class ImageNumericPolicyTests(unittest.TestCase):
    def test_component_precision_is_independent_of_float32_capture_dtype(self):
        reference = np.ones((4, 8), np.float32)
        actual = reference + np.float32(0.0005)
        half = numeric_metrics(reference, actual, compute_dtype="float16", boundary="component")
        vae = numeric_metrics(reference, actual, compute_dtype="float32", boundary="component")
        self.assertTrue(half["metrics"]["passed"])
        self.assertFalse(vae["metrics"]["passed"])
        self.assertEqual(half["policy_sha256"], canonical_hash(numeric_policy()))
        self.assertEqual(set(half["metric_code_sha256"]), {"numeric_policy.py", "contracts.py", "evaluation.py"})
        with self.assertRaises(ValueError):
            numeric_metrics(reference, actual, compute_dtype="automatic", boundary="component")

    def test_final_output_budgets_do_not_relax_with_compute_precision(self):
        reference = np.ones((1, 1024, 64), np.float32)
        actual = reference + np.float32(0.006)
        for precision in ("float32", "float16", "bfloat16"):
            with self.subTest(precision=precision):
                self.assertFalse(numeric_metrics(
                    reference, actual, compute_dtype=precision, boundary="final-latent")["metrics"]["passed"])
        rgba = np.zeros((1, 4, 1, 512, 512), np.float32)
        self.assertTrue(numeric_metrics(
            rgba, rgba + np.float32(0.009), compute_dtype="float16", boundary="final-rgba")["metrics"]["passed"])
        self.assertFalse(numeric_metrics(
            rgba, rgba + np.float32(0.011), compute_dtype="float16", boundary="final-rgba")["metrics"]["passed"])

    def test_shape_dtype_and_nonfinite_results_cannot_pass(self):
        reference = np.ones((1, 1024, 64), np.float32)
        actual = reference.copy()
        actual[0, 0, 0] = np.nan
        self.assertFalse(numeric_metrics(
            reference, actual, compute_dtype="float16", boundary="fixed-step-target")["metrics"]["passed"])
        for value, boundary in ((reference[:, :100], "fixed-step-target"),
                                (reference, "final-rgba"), (reference.astype(np.float16), "final-latent")):
            with self.subTest(boundary=boundary), self.assertRaises(ValueError):
                numeric_metrics(value, value, compute_dtype="float16", boundary=boundary)

    def test_frozen_policy_preserves_existing_final_and_float32_gates(self):
        policy = numeric_policy()
        self.assertEqual(canonical_hash(policy), "6e3c72b9321b6cdc105f8ddf38aaea9c151ab912e2bb9c36dd57654b31d4c793")
        self.assertEqual(policy["component"]["float32"], {"normalized_max": 1e-3, "normalized_rmse": 1e-4})
        self.assertEqual(policy["component"]["float16"], {"normalized_max": 4 * 2**-10, "normalized_rmse": 2**-10})
        self.assertEqual(policy["component"]["bfloat16"], {"normalized_max": 4 * 2**-7, "normalized_rmse": 2**-7})
        self.assertEqual(policy["rollout"], {"latent_normalized_rmse": 5e-3, "rgba_absolute_rmse": 1e-2})
