"""Configuration model: defaults, round trips and validation."""

import dataclasses
import math
import unittest

import numpy as np

from dashboard import config as cfg

import controllers
import run_simulation


class DefaultsTests(unittest.TestCase):
    def test_defaults_follow_the_simulator(self):
        config = cfg.default_config()
        sim, ref, ctrl = config.simulation, config.reference, config.controller
        self.assertEqual(ref.amplitude_m, controllers.FIGURE_EIGHT_AMPLITUDE)
        self.assertEqual(ref.period_s, controllers.FIGURE_EIGHT_PERIOD)
        self.assertEqual(
            sim.t_final_s,
            controllers.FIGURE_EIGHT_PERIOD * controllers.FIGURE_EIGHT_LAPS,
        )
        self.assertEqual(sim.dt_s, run_simulation.DT)
        np.testing.assert_array_equal(
            sim.initial_state(), controllers.figure_eight_initial_state()
        )
        self.assertIsInstance(ctrl, cfg.SMCConfig)
        self.assertEqual(ctrl.lookahead_L, controllers.TRACKING_LOOKAHEAD)
        self.assertEqual(config.disturbance.kind, run_simulation.DISTURBANCE_MODE)

    def test_smc_defaults_equal_smc_figure_eight_signature(self):
        smc = cfg.default_config().controller
        for field in dataclasses.fields(smc):
            self.assertEqual(
                getattr(smc, field.name), cfg._SMC_DEFAULTS[field.name], field.name
            )

    def test_default_configuration_is_valid(self):
        report = cfg.validate_config(cfg.default_config())
        self.assertEqual(report.errors, [])
        self.assertEqual(report.warnings, [])

    def test_every_disturbance_kind_builds_a_valid_config(self):
        for kind in cfg.DISTURBANCE_TYPES:
            check = cfg.check_form(dict(cfg.default_form(), dist_kind=kind))
            self.assertTrue(check.report.ok, (kind, check.report.errors))
            self.assertEqual(check.config.disturbance.kind, kind)


class RoundTripTests(unittest.TestCase):
    def _configs(self):
        base = cfg.default_form()
        yield cfg.default_config()
        yield cfg.form_to_config(dict(
            base, ctrl_kind="feedback", fb_k_u=70.0, dist_kind="sinusoidal",
            dist_sin_amp_v=9.0, sim_duration_mode="manual", sim_t_final_s=33.3,
        ))
        yield cfg.form_to_config(dict(
            base, sim_initial_mode="custom", sim_psi0_deg=12.345,
            sim_x0_m=-2.0, dist_kind="combined", dist_comb_wind=False,
            ref_timing_mode="peak_speed", ref_peak_speed_mps=1.1,
        ))

    def test_json_round_trip_is_exact(self):
        for config in self._configs():
            self.assertEqual(cfg.SimulationConfig.from_json(config.to_json()),
                             config)

    def test_form_round_trip_reproduces_the_experiment(self):
        for config in self._configs():
            restored = cfg.form_to_config(cfg.config_to_form(config))
            self.assertTrue(cfg.same_experiment(config, restored),
                            cfg.config_diff(config, restored))

    def test_peak_speed_timing_converts_to_period(self):
        form = dict(cfg.default_form(), ref_timing_mode="peak_speed",
                    ref_peak_speed_mps=1.0)
        config = cfg.form_to_config(form)
        self.assertAlmostEqual(config.reference.peak_speed_mps, 1.0, places=12)
        self.assertAlmostEqual(
            config.reference.period_s, 2 * math.pi * math.sqrt(2) * 15.0 / 1.0
        )

    def test_duration_from_laps(self):
        config = cfg.form_to_config(dict(cfg.default_form(), sim_laps=0.5,
                                         ref_period_s=100.0))
        self.assertEqual(config.simulation.t_final_s, 50.0)
        self.assertAlmostEqual(config.laps, 0.5)

    def test_config_is_immutable(self):
        config = cfg.default_config()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.controller.k_pos = 1.0


class ValidationTests(unittest.TestCase):
    def errors(self, **changes):
        return cfg.check_form(dict(cfg.default_form(), **changes)).report.errors

    def assertRejected(self, text, **changes):
        errors = self.errors(**changes)
        self.assertTrue(any(text in e for e in errors), errors)

    def test_boundary_layer_must_be_positive(self):
        self.assertRejected(r"$\phi_u$ must be > 0", smc_phi_u=0.0)
        self.assertRejected(r"$\phi_r$ must be > 0", smc_phi_r=-0.1)

    def test_gains_must_be_non_negative(self):
        self.assertRejected(r"$\lambda_u$ must be $\geq$ 0", smc_lambda_u=-1.0)
        self.assertRejected(r"$k_{s,r}$ must be $\geq$ 0", smc_k_s_r=-1.0)
        self.assertRejected(r"$k_u$ must be $\geq$ 0", ctrl_kind="feedback",
                            fb_k_u=-5.0)

    def test_lookahead_must_be_positive(self):
        self.assertRejected("lookahead distance $L$ must be > 0",
                            smc_lookahead_L=0.0)

    def test_time_settings(self):
        self.assertRejected(r"$\Delta t$ must be > 0", sim_dt_s=0.0)
        self.assertRejected("smaller than the total",
                            sim_duration_mode="manual", sim_t_final_s=1.0,
                            sim_dt_s=2.0)
        self.assertRejected("must be > 0 s", sim_duration_mode="manual",
                            sim_t_final_s=0.0)
        self.assertRejected("exceeds the dashboard limit",
                            sim_duration_mode="manual", sim_t_final_s=1e5)
        self.assertRejected(r"increase $\Delta t$", sim_duration_mode="manual",
                            sim_t_final_s=5000.0, sim_dt_s=0.001)
        self.assertRejected("Number of laps $N_{laps}$ must be > 0", sim_laps=0.0)

    def test_reference_settings(self):
        self.assertRejected("amplitude $A$ must be > 0", ref_amplitude_m=0.0)
        self.assertRejected("period $T$ must be > 0", ref_period_s=0.0,
                            sim_duration_mode="manual")
        self.assertRejected("Peak reference speed $v_{peak}$ must be > 0",
                            ref_timing_mode="peak_speed", ref_peak_speed_mps=0.0)

    def test_disturbance_settings(self):
        self.assertRejected(r"frequencies $f_i$ must be $\geq$ 0",
                            dist_kind="sinusoidal",
                            dist_sin_freq_r=-0.1)
        self.assertRejected(r"magnitude $F$ must be $\geq$ 0",
                            dist_kind="directional",
                            dist_dir_magnitude_N=-1.0)

    def test_non_finite_values_are_rejected(self):
        config = cfg.default_config()
        bad = dataclasses.replace(
            config,
            controller=dataclasses.replace(config.controller, k_pos=float("nan")),
        )
        report = cfg.validate_config(bad)
        self.assertTrue(any("finite" in e for e in report.errors), report.errors)

    def test_warnings_do_not_block(self):
        check = cfg.check_form(dict(cfg.default_form(), ref_period_s=40.0,
                                    sim_dt_s=0.03))
        self.assertTrue(check.report.ok)
        joined = " ".join(check.report.warnings)
        self.assertIn("exceeds the maximum steady surge speed", joined)
        self.assertIn(r"not a multiple of $\Delta t$", joined)


class DiffTests(unittest.TestCase):
    def test_diff_lists_changed_parameters_only(self):
        a = cfg.default_config()
        b = cfg.form_to_config(dict(cfg.default_form(), smc_lambda_r=2.5))
        self.assertEqual(cfg.config_diff(a, b),
                         [("controller.lambda_r", 2.0, 2.5)])
        self.assertEqual(cfg.config_diff(a, a), [])

    def test_diff_across_kinds(self):
        a = cfg.default_config()
        b = cfg.form_to_config(dict(cfg.default_form(), dist_kind="constant"))
        keys = [k for k, _, _ in cfg.config_diff(a, b)]
        self.assertEqual(keys[0], "disturbance.kind")
        self.assertIn("disturbance.dTr", keys)


if __name__ == "__main__":
    unittest.main()
