"""Performance indices on synthetic signals with known values."""

import math
import unittest

import numpy as np

from dashboard02_smc import config as cfg
from dashboard02_smc import engine
from dashboard02_smc import metrics as met


class ElementaryIndexTests(unittest.TestCase):
    def test_error_integrals(self):
        t = np.linspace(0.0, 10.0, 10001)
        iae, ise, itae = met.error_integrals(t, -2.0 * np.ones_like(t))
        self.assertAlmostEqual(iae, 20.0)
        self.assertAlmostEqual(ise, 40.0)
        self.assertAlmostEqual(itae, 2.0 * 10.0**2 / 2.0)

    def test_reaching_time_and_time_in_layer(self):
        t = np.arange(0.0, 10.0, 0.01)
        s = np.where(t < 2.0, 1.0, 0.01)
        s[(t > 6.0) & (t < 7.0)] = 0.5          # leaves the layer for 1 s
        t_reach, inside = met.reaching(t, s, phi=0.1)
        self.assertAlmostEqual(t_reach, 2.0)
        self.assertAlmostEqual(inside, 100.0 * 7.0 / 8.0, delta=0.3)
        t_reach, inside = met.reaching(t, np.ones_like(t), phi=0.1)
        self.assertTrue(math.isnan(t_reach))
        self.assertEqual(inside, 0.0)

    def test_zero_crossing_rate(self):
        t = np.arange(0.0, 20.0, 0.001)
        rate = met.zero_crossing_rate(np.sin(2 * np.pi * 0.5 * t + 0.1), 20.0)
        self.assertAlmostEqual(rate, 1.0, delta=0.06)   # 2 crossings per period

    def test_high_frequency_component_separates_chattering(self):
        t = np.arange(0.0, 60.0, 0.02)
        slow = 30.0 * np.sin(2 * np.pi * t / 70.0)
        chatter = np.where(np.arange(t.size) % 2, 5.0, -5.0)
        self.assertLess(met._rms(met.high_frequency_component(t, slow)), 0.05)
        hf = met.high_frequency_component(t, slow + chatter)
        self.assertAlmostEqual(met._rms(hf), 5.0, delta=0.1)

    def test_total_variation_rate(self):
        self.assertEqual(met.total_variation_rate([0, 1, 0, 1], 3.0), 1.0)

    def test_verdict(self):
        self.assertEqual(met.verdict(1.0, 2.0, "lower"), "better")
        self.assertEqual(met.verdict(1.0, 2.0, "higher"), "worse")
        self.assertEqual(met.verdict(1.0, 1.001, "lower"), "similar")
        self.assertEqual(met.verdict(float("nan"), 2.0, "lower"), "worse")
        self.assertEqual(met.verdict(2.0, float("nan"), "lower"), "better")


class RunIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        form = dict(cfg.default_form(), ref_laps=40.0 / 140.0,
                    sim_initial_mode="offset", init_dy_m=2.0,
                    init_dpsi_deg=30.0)
        cls.result = engine.run_from_config(cfg.form_to_config(form))

    def test_every_catalogued_index_is_computed(self):
        computed = set(self.result.metrics) - {"t_final_s", "samples"}
        self.assertEqual(computed, set(met.INDEX_INFO))
        for key, info in met.INDEX_INFO.items():
            self.assertIn(info[0], met.GROUPS, key)
            self.assertIn(info[4], ("lower", "higher"), key)

    def test_position_indices(self):
        m, data = self.result.metrics, self.result.data
        self.assertAlmostEqual(m["e_p_max"], data.e_p.max())
        self.assertAlmostEqual(m["e_p_rms"], np.sqrt(np.mean(data.e_p**2)))
        self.assertAlmostEqual(m["e_p_final"], data.e_p.iloc[-1])

    def test_reaching_phase_is_detected(self):
        m = self.result.metrics
        self.assertGreater(m["t_reach_r"], 0.5)    # 30 deg initial heading error
        self.assertGreater(m["in_layer_r_pct"], 95.0)

    def test_chattering_indices_ignore_the_reaching_transient(self):
        m = self.result.metrics
        # The initial transient saturates the thrusters, but the controller
        # does not chatter: the steady-state window indices stay small.
        self.assertLess(m["Tr_hf_rms"], 0.5)
        self.assertLess(m["sat_r_tv"], 0.1)

    def test_index_table(self):
        table = met.index_table(self.result.metrics, self.result.metrics)
        self.assertEqual(len(table), len(met.INDEX_INFO))
        self.assertTrue(set(table["Verdict"]) <= {"similar"})
        table = met.index_table(self.result.metrics)
        self.assertNotIn("Previous", table.columns)


if __name__ == "__main__":
    unittest.main()
