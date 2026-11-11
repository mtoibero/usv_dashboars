"""Configuration: defaults, reference parametrizations, validation, presets."""

import dataclasses
import math
import unittest

import numpy as np

from dashboard02_smc import config as cfg
from dashboard02_smc.tests.helpers import constant, sinusoid

import controllers


def form(**changes):
    data = cfg.default_form()
    data.update(changes)
    return data


class DefaultTests(unittest.TestCase):
    def test_defaults_are_the_simulator_defaults(self):
        config = cfg.default_config()
        ref = config.reference
        self.assertEqual(ref.amplitude_m, controllers.FIGURE_EIGHT_AMPLITUDE)
        self.assertEqual(ref.period_s, controllers.FIGURE_EIGHT_PERIOD)
        self.assertEqual(ref.laps, controllers.FIGURE_EIGHT_LAPS)
        smc = config.smc_kwargs()
        for name, value in cfg.SMC_DEFAULTS.items():
            if name in smc and name not in ("amplitude", "period"):
                self.assertEqual(smc[name], value, name)
        np.testing.assert_array_equal(
            config.initial_state(), controllers.figure_eight_initial_state())
        self.assertEqual(config.simulation.mode, cfg.DIGITAL)
        self.assertEqual(config.simulation.output_dt_s, cfg.SIMULATOR_DT_S)
        self.assertFalse(config.disturbance.active)
        self.assertEqual(cfg.validate_config(config).errors, [])
        self.assertEqual(cfg.validate_config(config).warnings, [])

    def test_simulator_dt_constant_matches_run_simulation(self):
        import run_simulation
        self.assertEqual(cfg.SIMULATOR_DT_S, run_simulation.DT)


class ReferenceTests(unittest.TestCase):
    def test_all_modes_describe_the_same_default_trajectory(self):
        configs = [cfg.form_to_config(form(ref_mode=mode))
                   for mode in cfg.REF_MODES]
        for config in configs[1:]:
            self.assertTrue(cfg.same_experiment(configs[0], config))

    def test_frequency_and_speed_define_the_amplitude(self):
        config = cfg.form_to_config(form(ref_mode="freq_speed",
                                         ref_frequency_hz=0.01,
                                         ref_peak_speed_mps=1.2))
        ref = config.reference
        self.assertAlmostEqual(ref.period_s, 100.0)
        self.assertAlmostEqual(ref.amplitude_m,
                               1.2 / (math.sqrt(2) * 2 * math.pi * 0.01))
        self.assertAlmostEqual(ref.peak_speed_mps, 1.2)
        self.assertAlmostEqual(ref.t_final_s, 200.0)

    def test_peak_speed_is_the_maximum_of_the_reference(self):
        ref = cfg.default_config().reference
        t = np.linspace(0.0, ref.period_s, 20001)
        _, _, xd, yd, _, _ = controllers.figure_eight_reference(
            t, ref.amplitude_m, ref.period_s)
        self.assertAlmostEqual(np.max(np.hypot(xd, yd)), ref.peak_speed_mps,
                               places=9)

    def test_other_modes(self):
        a = cfg.form_to_config(form(ref_mode="amp_freq", ref_amplitude_m=10.0,
                                    ref_frequency_hz=0.02)).reference
        self.assertEqual((a.amplitude_m, a.period_s), (10.0, 50.0))
        b = cfg.form_to_config(form(ref_mode="amp_speed", ref_amplitude_m=10.0,
                                    ref_peak_speed_mps=a.peak_speed_mps))
        self.assertAlmostEqual(b.reference.period_s, 50.0)

    def test_zero_inputs_are_reported(self):
        check = cfg.check_form(form(ref_frequency_hz=0.0))
        self.assertIsNone(check.config)
        self.assertTrue(any("frequency" in e for e in check.report.errors))


class ValidationTests(unittest.TestCase):
    def errors(self, **changes):
        return cfg.check_form(form(**changes)).report.errors

    def warnings(self, **changes):
        return cfg.check_form(form(**changes)).report.warnings

    def test_boundary_layer_must_be_positive(self):
        self.assertTrue(any(r"\phi_u" in e for e in self.errors(surge_phi_u=0)))
        self.assertTrue(any(r"\phi_r" in e for e in self.errors(yaw_phi_r=-1)))

    def test_gains_must_be_non_negative(self):
        for key, symbol in (("surge_lambda_u", r"\lambda_u"),
                            ("surge_k_s_u", "k_{s,u}"),
                            ("surge_k_pos", "k_{pos}"),
                            ("yaw_lambda_r", r"\lambda_r"),
                            ("yaw_k_s_r", "k_{s,r}"),
                            ("yaw_k_psi", r"k_{\psi}")):
            self.assertTrue(any(symbol in e for e in self.errors(**{key: -0.1})),
                            key)
            self.assertEqual(self.errors(**{key: 0.0}), [], key)

    def test_non_finite_values_are_rejected(self):
        self.assertTrue(self.errors(surge_lambda_u=float("nan")))
        self.assertTrue(self.errors(**constant(d_r=float("inf"))))
        self.assertTrue(self.errors(dist_kind="directional",
                                    dist_dir_magnitude_N=-1.0))
        self.assertTrue(self.errors(dist_kind="sinusoidal",
                                    dist_sin_freq_u=-0.1))

    def test_custom_duration(self):
        config = cfg.form_to_config(form(sim_duration_mode="manual",
                                         sim_t_final_s=100.0))
        self.assertAlmostEqual(config.t_final_s, 100.0)
        back = cfg.config_to_form(config)
        self.assertEqual(back["sim_duration_mode"], "manual")
        self.assertAlmostEqual(back["sim_t_final_s"], 100.0)
        self.assertTrue(cfg.check_form(form(sim_duration_mode="manual",
                                            sim_t_final_s=0.0)).report.errors)

    def test_simulation_limits(self):
        self.assertTrue(self.errors(ref_laps=100.0))           # > 3600 s
        self.assertTrue(self.errors(sim_sample_rate_hz=0.5))
        self.assertTrue(self.errors(sim_sample_rate_hz=5000.0))
        self.assertTrue(self.errors(sim_mode=cfg.CONTINUOUS, sim_output_dt_s=1.0))
        self.assertEqual(self.errors(sim_mode=cfg.CONTINUOUS), [])
        self.assertTrue(self.warnings(ref_laps=10.0))          # 1400 s

    def test_physical_warnings(self):
        self.assertTrue(any("thruster saturation" in w
                            for w in self.warnings(ref_peak_speed_mps=2.0)))
        self.assertTrue(any("k_{s,r}" in w for w in self.warnings(
            **constant(d_r=200.0))))
        self.assertTrue(any("thin boundary layer" in w for w in self.warnings(
            sim_mode=cfg.CONTINUOUS, yaw_phi_r=0.001)))


class SerializationTests(unittest.TestCase):
    def test_json_round_trip(self):
        config = cfg.form_to_config(cfg.preset_form("weak_switching"))
        self.assertEqual(cfg.SimulationConfig.from_json(config.to_json()), config)

    def test_form_round_trip_for_every_preset_and_mode(self):
        for name in cfg.PRESETS:
            for mode in cfg.REF_MODES:
                base = dict(cfg.default_form(), ref_mode=mode)
                config = cfg.form_to_config(cfg.preset_form(name, base))
                back = cfg.form_to_config(cfg.config_to_form(config, base))
                self.assertEqual(cfg.config_diff(config, back), [],
                                 (name, mode))

    def test_disturbance_models_and_defaults(self):
        """Same models and defaults as the first dashboard, i.e. the modes of
        run_simulation.DISTURBANCE_OPTIONS."""
        self.assertEqual(set(cfg.DIST_KINDS), {
            "none", "constant", "sinusoidal", "directional", "wind",
            "combined"})
        const = cfg.form_to_config(form(dist_kind="constant")).disturbance
        self.assertEqual(const.constant, (0.0, 0.0, 15.0))
        sine = cfg.form_to_config(form(dist_kind="sinusoidal")).disturbance
        self.assertEqual(sine.sin_amplitude, (3.0, 4.0, 2.0))
        self.assertEqual(sine.sin_bias, (0.0, 0.0, 0.0))
        comb = cfg.form_to_config(form(dist_kind="combined")).disturbance
        self.assertTrue(comb.include_wind)
        self.assertEqual((comb.dir_magnitude_N, comb.dir_direction_deg),
                         (7.0, 64.0))

    def test_disturbance_bounds(self):
        def bounds(**changes):
            return cfg.form_to_config(form(**changes)).disturbance.bounds()
        self.assertEqual(bounds(**constant(-5, 2, 8)), (5.0, 2.0, 8.0))
        self.assertEqual(bounds(**sinusoid("r", 4.0, 0.1, bias=-1.0)),
                         (0.0, 0.0, 5.0))
        self.assertEqual(bounds(dist_kind="directional"), (25.0, 25.0, 0.0))
        self.assertEqual(bounds(dist_kind="wind"), cfg.WIND_BOUNDS)
        self.assertEqual(bounds(dist_kind="combined"),
                         (2 + 7 + 6, 7 + 7 + 3, 2.5 + 1.5))
        self.assertEqual(bounds(), (0.0, 0.0, 0.0))

    def test_hidden_disturbance_values_do_not_change_the_experiment(self):
        a = cfg.form_to_config(form(dist_kind="constant", dist_sin_amp_u=99.0,
                                    dist_dir_magnitude_N=3.0))
        b = cfg.form_to_config(form(dist_kind="constant", dist_sin_amp_u=1.0))
        self.assertEqual(a, b)
        self.assertEqual(a.disturbance.sin_amplitude, (0.0, 0.0, 0.0))
        # Offsets only count when the initial state is "offset".
        c = cfg.form_to_config(form(init_dy_m=5.0))
        self.assertEqual(c, cfg.default_config())
        d = cfg.form_to_config(form(init_dy_m=5.0, sim_initial_mode="offset"))
        self.assertEqual(d.simulation.dy0_m, 5.0)

    def test_diff_reports_changes_only(self):
        a = cfg.default_config()
        b = cfg.form_to_config(form(yaw_phi_r=0.02))
        self.assertEqual(cfg.config_diff(a, b), [("yaw.phi_r", 0.08, 0.02)])
        self.assertTrue(cfg.same_experiment(a, cfg.default_config()))

    def test_every_flat_key_has_display_metadata(self):
        for key in cfg.default_config().flat():
            self.assertIn(key, cfg.PARAM_INFO)


class PresetAndDesignTests(unittest.TestCase):
    def test_presets_are_valid(self):
        for name in cfg.PRESETS:
            check = cfg.check_form(cfg.preset_form(name))
            self.assertTrue(check.report.ok, (name, check.report.errors))

    def test_discrete_gain(self):
        config = cfg.form_to_config(form(yaw_phi_r=0.01, sim_sample_rate_hz=20))
        gamma = [c for c in cfg.design_checks(config)
                 if c["kind"] == "discrete" and c["channel"] == "yaw"][0]
        self.assertAlmostEqual(gamma["gamma"], (2.0 + 0.8 / 0.01) / 20.0)
        self.assertFalse(gamma["ok"])
        default = [c for c in cfg.design_checks(cfg.default_config())
                   if c["kind"] == "discrete"]
        self.assertTrue(all(c["ok"] for c in default))

    def test_continuous_mode_has_no_discrete_check(self):
        config = cfg.form_to_config(form(sim_mode=cfg.CONTINUOUS))
        self.assertFalse([c for c in cfg.design_checks(config)
                          if c["kind"] == "discrete"])

    def test_sliding_check_uses_disturbance_and_trajectory(self):
        from dashboard02_smc import tuning
        config = cfg.form_to_config(form(
            **sinusoid("u", 10.0, 0.05, bias=5.0)))
        check = [c for c in cfg.design_checks(config)
                 if c["kind"] == "sliding" and c["channel"] == "surge"][0]
        d_bar = 15.0 / cfg.PARAMS["z1"] + tuning.load_for(config).d_traj_u
        self.assertIn(f"{d_bar:.3g}", check["latex"])
        self.assertTrue(check["ok"])
        weak = cfg.form_to_config(form(surge_k_s_u=0.01, surge_lambda_u=0.1,
                                       **constant(d_u=10.0)))
        self.assertTrue(any("sliding condition violated" in w
                            for w in cfg.validate_config(weak).warnings))

    def test_config_is_immutable(self):
        config = cfg.default_config()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.surge.lambda_u = 3.0


if __name__ == "__main__":
    unittest.main()
