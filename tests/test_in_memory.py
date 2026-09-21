import copy
import unittest

import numpy as np
import torch as tc

from deeperhistreg.dhr_pipeline.in_memory import (
    build_registration_parameters,
    displacement_field_qc,
    warp_array_with_displacement,
)


class TestInMemoryHelpers(unittest.TestCase):
    def test_build_params_applies_overrides_without_mutating_input(self):
        overrides = {
            "loading_params": {
                "source_resample_ratio": 0.5,
                "target_resample_ratio": 0.5,
            },
            "nonrigid_registration_params": {"registration_size": 1024},
        }
        original = copy.deepcopy(overrides)
        params = build_registration_parameters(
            preset="default_nonrigid_fast",
            device="cpu",
            overrides=overrides,
        )
        self.assertEqual(overrides, original)
        self.assertEqual(params["device"], "cpu")
        self.assertEqual(params["loading_params"]["source_resample_ratio"], 0.5)
        self.assertEqual(params["loading_params"]["target_resample_ratio"], 0.5)
        self.assertEqual(
            params["nonrigid_registration_params"]["registration_size"], 1024
        )
        self.assertFalse(params["save_final_images"])
        self.assertFalse(params["save_final_displacement_field"])
        self.assertFalse(params["preprocessing_params"]["save_results"])

    def test_mismatched_resample_ratios_fail(self):
        with self.assertRaisesRegex(ValueError, "matching source/target"):
            build_registration_parameters(
                overrides={
                    "loading_params": {
                        "source_resample_ratio": 0.5,
                        "target_resample_ratio": 0.25,
                    }
                }
            )

    def test_identity_displacement_qc(self):
        field = tc.zeros((1, 32, 48, 2), dtype=tc.float32)
        qc = displacement_field_qc(field)
        self.assertAlmostEqual(qc["displacement_max_px"], 0.0, places=6)
        self.assertAlmostEqual(qc["jacobian_median"], 1.0, places=6)
        self.assertAlmostEqual(qc["folding_fraction"], 0.0, places=6)

    def test_constant_translation_has_unit_jacobian(self):
        h, w = 32, 64
        field = tc.zeros((1, h, w, 2), dtype=tc.float32)
        field[..., 0] = 2.0 * 4.0 / w
        field[..., 1] = 2.0 * -2.0 / h
        qc = displacement_field_qc(field)
        expected = float(np.sqrt(4.0 ** 2 + 2.0 ** 2))
        self.assertAlmostEqual(qc["displacement_median_px"], expected, places=4)
        self.assertAlmostEqual(qc["jacobian_median"], 1.0, places=5)
        self.assertAlmostEqual(qc["folding_fraction"], 0.0, places=6)

    def test_zero_field_warp_preserves_image_and_mask(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[8:40, 10:50, 0] = 180
        mask = np.ones((48, 64), dtype=np.uint8)
        mask[:, :4] = 0
        field = tc.zeros((1, 24, 32, 2), dtype=tc.float32)

        warped, valid, qc = warp_array_with_displacement(
            image,
            field,
            source_valid_mask=mask,
        )
        diff = np.abs(warped.astype(np.int16) - image.astype(np.int16))
        self.assertLessEqual(int(diff.max()), 1)
        self.assertLess(float(diff.mean()), 0.05)
        self.assertTrue(np.array_equal(valid, mask.astype(bool)))
        self.assertAlmostEqual(qc["valid_fraction"], float(mask.mean()), places=6)

    def test_translation_reduces_valid_support_at_boundary(self):
        h, w = 32, 64
        image = np.zeros((h, w, 3), dtype=np.uint8)
        field = tc.zeros((1, h, w, 2), dtype=tc.float32)
        field[..., 0] = 2.0 * 4.0 / w

        _, valid, qc = warp_array_with_displacement(image, field)
        self.assertAlmostEqual(qc["valid_fraction"], (w - 4) / w, places=6)
        self.assertEqual(int(valid[:, -4:].sum()), 0)

    def test_invalid_rgb_dtype_fails_loudly(self):
        image = np.zeros((32, 32, 3), dtype=np.float32)
        field = tc.zeros((1, 32, 32, 2), dtype=tc.float32)
        with self.assertRaisesRegex(TypeError, "dtype uint8"):
            warp_array_with_displacement(image, field)


if __name__ == "__main__":
    unittest.main()
