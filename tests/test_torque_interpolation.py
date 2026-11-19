import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller_comparison import compare_controllers
from torque_interpolation import (
    HEEL,
    MID,
    PROFILE_KEYS,
    TOE,
    blend_profiles,
    cop_interpolation_weights,
    select_discrete_profile,
    smoothstep,
)
from torque_profiles import load_profile_coefficients, poly5_profile

HEEL_MM = 0.0
MID_MM = 261.1 / 2.0
TOE_MM = 261.1


class TorqueInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.x_grid = np.linspace(0.0, 100.0, 200)
        coefficients = load_profile_coefficients()
        self.profiles = {
            HEEL: poly5_profile(self.x_grid, coefficients[HEEL], 75.0, 0.25),
            MID: poly5_profile(self.x_grid, coefficients[MID], 75.0, 0.25),
            TOE: poly5_profile(self.x_grid, coefficients[TOE], 75.0, 0.25),
        }

    def assert_valid_weights(self, weights):
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=12)
        for key in PROFILE_KEYS:
            self.assertGreaterEqual(weights[key], 0.0)

    def blended(self, cop_mm):
        weights = cop_interpolation_weights(cop_mm, HEEL_MM, MID_MM, TOE_MM)
        self.assert_valid_weights(weights)
        return weights, blend_profiles(self.profiles, weights)

    def test_smoothstep_endpoints_and_midpoint(self):
        self.assertEqual(smoothstep(-1.0), 0.0)
        self.assertEqual(smoothstep(0.0), 0.0)
        self.assertAlmostEqual(smoothstep(0.5), 0.5)
        self.assertEqual(smoothstep(1.0), 1.0)
        self.assertEqual(smoothstep(2.0), 1.0)

    def test_heel_reproduces_original_profile(self):
        weights, blended = self.blended(HEEL_MM)
        self.assertAlmostEqual(weights[HEEL], 1.0)
        np.testing.assert_allclose(blended, self.profiles[HEEL], rtol=0.0, atol=1e-7)

    def test_midfoot_reproduces_original_profile(self):
        weights, blended = self.blended(MID_MM)
        self.assertAlmostEqual(weights[MID], 1.0)
        np.testing.assert_allclose(blended, self.profiles[MID], rtol=0.0, atol=1e-7)

    def test_toe_reproduces_original_profile(self):
        weights, blended = self.blended(TOE_MM)
        self.assertAlmostEqual(weights[TOE], 1.0)
        np.testing.assert_allclose(blended, self.profiles[TOE], rtol=0.0, atol=1e-7)

    def test_between_heel_and_midfoot(self):
        weights, blended = self.blended((HEEL_MM + MID_MM) / 2.0)
        self.assertAlmostEqual(weights[HEEL], 0.5)
        self.assertAlmostEqual(weights[MID], 0.5)
        self.assertEqual(weights[TOE], 0.0)
        expected = 0.5 * self.profiles[HEEL] + 0.5 * self.profiles[MID]
        np.testing.assert_allclose(blended, expected, rtol=0.0, atol=1e-7)

    def test_between_midfoot_and_toe(self):
        weights, blended = self.blended((MID_MM + TOE_MM) / 2.0)
        self.assertEqual(weights[HEEL], 0.0)
        self.assertAlmostEqual(weights[MID], 0.5)
        self.assertAlmostEqual(weights[TOE], 0.5)
        expected = 0.5 * self.profiles[MID] + 0.5 * self.profiles[TOE]
        np.testing.assert_allclose(blended, expected, rtol=0.0, atol=1e-7)

    def test_out_of_range_and_non_finite_cop(self):
        cases = (
            (-100.0, HEEL),
            (1000.0, TOE),
            (float("-inf"), HEEL),
            (float("inf"), TOE),
            (float("nan"), MID),
        )
        for cop_mm, expected_key in cases:
            with self.subTest(cop_mm=cop_mm):
                weights, blended = self.blended(cop_mm)
                self.assertAlmostEqual(weights[expected_key], 1.0)
                np.testing.assert_allclose(
                    blended, self.profiles[expected_key], rtol=0.0, atol=1e-7
                )

    def test_invalid_reference_positions(self):
        with self.assertRaises(ValueError):
            cop_interpolation_weights(100.0, 0.0, 0.0, 200.0)
        with self.assertRaises(ValueError):
            cop_interpolation_weights(100.0, 0.0, 200.0, 100.0)

    def test_discrete_baseline_boundary(self):
        self.assertEqual(select_discrete_profile(MID_MM, MID_MM), HEEL)
        self.assertEqual(select_discrete_profile(MID_MM + 1e-6, MID_MM), TOE)

    def test_rapidly_fluctuating_cop(self):
        cops = np.array([
            HEEL_MM, MID_MM - 30.0, MID_MM + 30.0, MID_MM - 5.0,
            MID_MM + 5.0, MID_MM - 1.0, MID_MM + 1.0, TOE_MM,
            MID_MM + 1.0, MID_MM - 1.0, MID_MM + 30.0, MID_MM - 30.0,
        ])
        for cop_mm in cops:
            with self.subTest(cop_mm=cop_mm):
                _, blended = self.blended(cop_mm)
                self.assertTrue(np.all(np.isfinite(blended)))
                self.assertLessEqual(float(blended.max()), float(max(profile.max() for profile in self.profiles.values())) + 1e-7)

    def test_controller_comparison_metrics(self):
        report = compare_controllers()
        for controller in ("continuous", "discrete"):
            metrics = report[controller]
            for value in metrics.values():
                self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(metrics["max_torque_nm"], 0.0)
            self.assertGreater(metrics["max_torque_rate_nm_per_s"], 0.0)

        self.assertLessEqual(
            report["continuous"]["max_curve_jump_nm"],
            report["discrete"]["max_curve_jump_nm"],
        )
        self.assertLess(
            report["continuous"]["max_switch_jump_nm"],
            report["discrete"]["max_switch_jump_nm"],
        )
        self.assertLess(
            report["continuous"]["max_switch_rate_nm_per_s"],
            report["discrete"]["max_switch_rate_nm_per_s"],
        )
        self.assertLess(
            report["continuous"]["max_cop_sensitivity_nm_per_mm"],
            report["discrete"]["max_cop_sensitivity_nm_per_mm"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
