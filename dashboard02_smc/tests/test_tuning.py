"""Gain hints: formulas, and the closed-loop promise they make."""

import unittest

import numpy as np

from dashboard02_smc import config as cfg
from dashboard02_smc import engine, tuning
from dashboard02_smc import latex as tex
from dashboard02_smc.tests.helpers import constant, sinusoid


def form(**changes):
    data = dict(cfg.default_form(), ref_laps=1.0)
    data.update(changes)
    return data


def hints(**changes):
    return tuning.gain_hints(cfg.form_to_config(form(**changes)))


class FormulaTests(unittest.TestCase):
    def test_matched_uncertainty(self):
        h = hints(**sinusoid("u", 10.0, 0.05, bias=5.0))["surge"]
        self.assertAlmostEqual(h.d_ext, 15.0 / cfg.PARAMS["z1"])
        self.assertGreater(h.d_traj, 0.0)
        self.assertAlmostEqual(h.d_bar, h.d_ext + h.d_traj)
        y = hints(**constant(d_r=-8.0))["yaw"]
        d = hints(dist_kind="directional")
        self.assertAlmostEqual(d["surge"].d_ext, 25.0 / cfg.PARAMS["z1"])
        self.assertEqual(d["yaw"].d_ext, 0.0)
        self.assertAlmostEqual(y.d_ext, 8.0 / cfg.PARAMS["z3"])

    def test_sliding_condition_and_alternatives(self):
        h = hints(**constant(d_r=15.0), yaw_k_s_r=0.05,
                  yaw_lambda_r=0.5)["yaw"]
        self.assertFalse(h.sliding_ok)
        self.assertAlmostEqual(h.k_min, h.d_bar - 0.5 * 0.08)
        # Each alternative restores the condition exactly.
        self.assertAlmostEqual(0.05 + h.lam_min * 0.08, h.d_bar)
        self.assertAlmostEqual(0.05 + 0.5 * h.phi_min_sliding, h.d_bar)
        self.assertAlmostEqual(h.k_suggested, 1.25 * h.d_bar)
        self.assertAlmostEqual(h.tolerable_disturbance,
                               cfg.PARAMS["z3"] * (0.05 + 0.04 - h.d_traj))

    def test_chattering_window(self):
        h = hints(yaw_phi_r=0.005, sim_sample_rate_hz=20.0)["yaw"]
        self.assertAlmostEqual(h.k_max_chattering, 0.005 * (20.0 - 2.0))
        self.assertFalse(h.chattering_ok)               # k_s = 0.8 chatters
        self.assertTrue(h.window_exists)                 # small k_s works
        self.assertAlmostEqual(h.phi_min_window, h.k_suggested / 18.0)
        big = hints(yaw_phi_r=0.005, sim_sample_rate_hz=20.0,
                    **constant(d_r=30.0))["yaw"]
        self.assertFalse(big.window_exists)
        self.assertIsNone(hints(sim_mode=cfg.CONTINUOUS)["yaw"].k_max_chattering)

    def test_trajectory_part_grows_with_the_reference_frequency(self):
        slow = hints(ref_mode="amp_freq", ref_frequency_hz=1 / 140.0)
        fast = hints(ref_mode="amp_freq", ref_frequency_hz=1 / 70.0)
        for channel in ("surge", "yaw"):
            ratio = fast[channel].d_traj / slow[channel].d_traj
            self.assertGreater(ratio, 3.0, channel)      # ~ omega^2
            self.assertLess(ratio, 5.0, channel)

    def test_suggested_value_rounds_up(self):
        self.assertEqual(tuning.suggested_value(0.1423), 0.15)
        self.assertEqual(tuning.suggested_value(0.15), 0.15)
        self.assertEqual(tuning.suggested_value(0.0252), 0.026)
        self.assertEqual(tuning.suggested_value(0.0), 0.0)

    def test_actuator_check(self):
        default = tuning.actuator_check(cfg.default_config())
        self.assertLess(default.utilization_pct, 80.0)
        heavy = tuning.actuator_check(cfg.form_to_config(form(
            **constant(d_r=30.0))))
        self.assertGreater(heavy.utilization_pct, 100.0)
        self.assertTrue(any("Thrusters" in w for w in cfg.validate_config(
            cfg.form_to_config(form(**constant(d_r=30.0)))).warnings))

    def test_hint_text(self):
        h = hints(**constant(d_r=15.0), yaw_k_s_r=0.05,
                  yaw_lambda_r=0.5)["yaw"]
        lines = tex.gain_hint_lines(h, tuning.suggested_value(h.k_suggested))
        self.assertTrue(lines[0].startswith("$$"))
        self.assertIn("sliding condition violated", lines[2])
        self.assertEqual(tex.gain_status(h)[0], "violated")
        ok = hints()["yaw"]
        self.assertEqual(tex.gain_status(ok)[0], "robust")


class ClosedLoopTests(unittest.TestCase):
    """With the suggested k_s the sliding variable stays in the layer; with
    a k_s below the minimum it leaves it (constant disturbances make the
    bound nearly tight)."""

    def run_case(self, channel, k_s, **dist):
        key = "surge_k_s_u" if channel == "surge" else "yaw_k_s_r"
        config = cfg.form_to_config(form(surge_lambda_u=0.5, yaw_lambda_r=0.5,
                                         **{key: k_s}, **dist))
        data = engine.run_from_config(config).data
        s = data["s_u" if channel == "surge" else "s_r"][data.t > 20.0]
        return float(np.abs(s).max() / 0.08)          # max |s| / phi

    def check(self, channel, **dist):
        key = "surge_k_s_u" if channel == "surge" else "yaw_k_s_r"
        h = tuning.gain_hints(cfg.form_to_config(form(
            surge_lambda_u=0.5, yaw_lambda_r=0.5, **{key: 0.05}, **dist)))[channel]
        self.assertGreater(h.k_min, 0.05)
        self.assertGreater(self.run_case(channel, 0.05, **dist), 1.0)
        suggested = tuning.suggested_value(h.k_suggested)
        self.assertLess(self.run_case(channel, suggested, **dist), 0.9)

    def test_surge_constant_disturbance(self):
        self.check("surge", **constant(d_u=12.0))

    def test_yaw_sinusoidal_disturbance(self):
        self.check("yaw", **sinusoid("r", 12.0, 0.05))

    def test_when_the_thrusters_saturate_sliding_is_lost_whatever_the_gain(self):
        """The actuator check (>= 100 %) explains the remaining failures: the
        samples outside the layer are exactly the saturated ones."""
        dist = sinusoid("r", 15.0, 0.05)
        base = form(surge_lambda_u=0.5, yaw_lambda_r=0.5, yaw_k_s_r=0.05, **dist)
        config = cfg.form_to_config(base)
        self.assertGreaterEqual(tuning.actuator_check(config).utilization_pct,
                                99.5)
        k_s = tuning.suggested_value(
            tuning.gain_hints(config)["yaw"].k_suggested)
        data = engine.run_from_config(cfg.form_to_config(
            dict(base, yaw_k_s_r=k_s))).data
        forces = data[["F1", "F2", "F3", "F4"]].to_numpy()
        saturated = (np.isclose(forces, 36.38, atol=1e-6)
                     | np.isclose(forces, -28.44, atol=1e-6)).any(axis=1)
        outside = (np.abs(data.s_r) > 0.08).to_numpy()
        self.assertTrue(outside.any())
        self.assertTrue(np.all(saturated[outside]))


if __name__ == "__main__":
    unittest.main()
