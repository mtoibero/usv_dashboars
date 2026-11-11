"""Engine: connection to the real simulator and consistency with it."""

import unittest
from unittest import mock

import numpy as np
from scipy.integrate import solve_ivp

from dashboard02_smc import config as cfg
from dashboard02_smc import engine
from dashboard02_smc.tests.helpers import constant, offset, sinusoid

import controllers
import disturbances
import run_simulation
import usv_model


def short_config(t_final=20.0, **changes):
    """Config of a short run (laps chosen so that T_sim = t_final)."""
    form = dict(cfg.default_form(), ref_laps=t_final / 140.0)
    form.update(changes)
    return cfg.form_to_config(form)


# Outputs of the ORIGINAL smc_figure_eight (before ``signals`` was added),
# computed with the unmodified controllers.py of the repository baseline.
GOLDEN = [
    (0.0, [0, 0, 0.785, 0.9, 0.0, 0.0], [[38.25403556864722, 0.0, 4.633388998294995], [38.25403556864722, 0.0, 4.633388998294996], [6.105755908359569, 6.105755908359569, 13.02126187596404, 13.02126187596404]]),
    (13.7, [3.1, -1.2, 1.1, 0.6, 0.05, -0.1], [[774.2325405873839, 0.0, -432.2940277414703], [15.880000000000003, 0.0, -43.42940000000001], [36.38, 36.38, -28.44, -28.44]]),
    (57.3, [-8.0, 4.5, -2.9, 1.2, -0.1, 0.3], [[804.1245570212646, 0.0, 2331.641582907925], [15.880000000000003, 0.0, 43.42940000000001], [-28.44, -28.44, 36.38, 36.38]]),
    (91.0, [5.0, 2.0, 3.1, 0.2, 0.0, 0.05], [[1536.2320365858468, 0.0, 452.0383835342395], [145.52, 0.0, 0.0], [36.38, 36.38, 36.38, 36.38]]),
    (30.0, [1.0, 1.0, 0.3, 0.7, 0.02, 0.01, 29.5, 0.8], [[890.9574283863379, 0.0, -709.209522325282], [15.880000000000003, 0.0, -43.42940000000001], [36.38, 36.38, -28.44, -28.44]]),
]


class ControllerRefactorTests(unittest.TestCase):
    def controller(self):
        return controllers.smc_figure_eight(
            k_pos=0.5, k_psi=2.5, lambda_u=1.5, lambda_r=2.5, k_s_u=0.4,
            k_s_r=0.9, phi_u=0.05, phi_r=0.06, lookahead_L=1.2,
            amplitude=12.0, period=120.0)

    def test_outputs_are_bit_identical_to_the_original_controller(self):
        controller = self.controller()
        for t, state, expected in GOLDEN:
            output = controller.output(t, np.array(state, dtype=float))
            for got, want in zip(output, expected):
                self.assertEqual([float(v) for v in got], want)

    def test_signals_contain_the_same_forces(self):
        controller = self.controller()
        for t, state, expected in GOLDEN:
            signals = controller.signals(t, np.array(state, dtype=float))
            self.assertEqual(list(signals["tau_required"]), expected[0])
            self.assertEqual(list(signals["tau_delivered"]), expected[1])
            s_u = signals["s_u"]
            self.assertEqual(s_u, state[3] - signals["speed_cmd"])


class ParameterPropagationTests(unittest.TestCase):
    def test_all_gains_reach_smc_figure_eight(self):
        config = short_config(surge_lambda_u=1.7, surge_k_s_u=0.6,
                              surge_phi_u=0.03, surge_k_pos=0.45,
                              yaw_lambda_r=2.2, yaw_k_s_r=1.1, yaw_phi_r=0.05,
                              yaw_k_psi=3.5, sim_lookahead_L_m=1.4,
                              ref_mode="amp_freq", ref_amplitude_m=12.0,
                              ref_frequency_hz=0.01)
        with mock.patch.object(controllers, "smc_figure_eight",
                               wraps=controllers.smc_figure_eight) as factory:
            engine.build_controller(config)
        self.assertEqual(factory.call_args.kwargs, dict(
            k_pos=0.45, k_psi=3.5, lambda_u=1.7, lambda_r=2.2, k_s_u=0.6,
            k_s_r=1.1, phi_u=0.03, phi_r=0.05, lookahead_L=1.4,
            amplitude=12.0, period=100.0))

    def test_disturbances_use_the_existing_functions(self):
        config = short_config(**sinusoid("r", 3.0, 0.2, phase=np.pi / 2,
                                         bias=1.0))
        with mock.patch.object(disturbances, "sinusoidal_disturbance",
                               wraps=disturbances.sinusoidal_disturbance) as f:
            fun = engine.build_disturbance(config.disturbance)
        kwargs = f.call_args.kwargs
        self.assertEqual(kwargs["bias"], (0.0, 0.0, 1.0))
        self.assertEqual(kwargs["amplitude"], (0.0, 0.0, 3.0))
        for t in (0.0, 1.3, 7.9):
            expected = [0.0, 0.0, 1.0 + 3.0 * np.sin(2 * np.pi * 0.2 * t
                                                     + np.pi / 2)]
            np.testing.assert_allclose(fun(t, np.zeros(6)), expected,
                                       rtol=0, atol=1e-12)
        self.assertIs(engine.build_disturbance(
            cfg.default_config().disturbance), disturbances.no_disturbance)

    def test_default_models_match_run_simulation(self):
        """Each model with its default parameters is the corresponding mode
        of run_simulation.DISTURBANCE_OPTIONS (as in the first dashboard)."""
        rng = np.random.default_rng(1)
        for kind in ("none", "constant", "sinusoidal", "directional", "wind",
                     "combined"):
            config = short_config(dist_kind=kind)
            ours = engine.build_disturbance(config.disturbance)
            theirs = run_simulation.DISTURBANCE_OPTIONS[kind]
            for _ in range(20):
                t = rng.uniform(0.0, 300.0)
                state = rng.normal(size=6)
                np.testing.assert_allclose(ours(t, state), theirs(t, state),
                                           rtol=0, atol=1e-12, err_msg=kind)

    def test_bounds_contain_the_disturbance(self):
        rng = np.random.default_rng(2)
        for kind in ("constant", "sinusoidal", "directional", "wind",
                     "combined"):
            dist = short_config(dist_kind=kind).disturbance
            fun = engine.build_disturbance(dist)
            values = np.array([fun(t, np.r_[0, 0, psi, 0, 0, 0])
                               for t, psi in zip(rng.uniform(0, 500, 4000),
                                                 rng.uniform(-4, 4, 4000))])
            peak = np.abs(values).max(axis=0)
            bounds = np.array(dist.bounds())
            self.assertTrue(np.all(peak <= bounds + 1e-9), kind)
            self.assertTrue(np.all(peak >= 0.8 * bounds), kind)


class ContinuousModeTests(unittest.TestCase):
    def test_matches_a_direct_call_of_the_simulator(self):
        config = short_config(20.0, sim_mode=cfg.CONTINUOUS,
                              **constant(d_r=4.0), **offset(dy=1.0))
        result = engine.run_from_config(config)

        controller = controllers.smc_figure_eight(**config.smc_kwargs())
        dist = disturbances.sinusoidal_disturbance(
            amplitude=(0, 0, 0), frequency=(0, 0, 0), phase=(0, 0, 0),
            bias=(0.0, 0.0, 4.0))
        state0 = controllers.figure_eight_initial_state(15.0, 140.0)
        state0[1] += 1.0
        t, state = usv_model.simulate(state0, controller, 20.0,
                                      dt=cfg.SIMULATOR_DT_S,
                                      disturbance_fun=dist)
        np.testing.assert_array_equal(result.data["t"], t)
        np.testing.assert_array_equal(
            result.data[["x", "y", "psi", "u", "v", "r"]].to_numpy(), state)
        _req, delivered, forces = run_simulation.controller_output_history(
            controller, t, state)
        np.testing.assert_array_equal(result.data["Tu"], delivered[:, 0])
        np.testing.assert_array_equal(result.data["Tr"], delivered[:, 2])
        np.testing.assert_array_equal(
            result.data[["F1", "F2", "F3", "F4"]].to_numpy(), forces)


class DigitalModeTests(unittest.TestCase):
    def setUp(self):
        self.config = short_config(10.0, sim_sample_rate_hz=20.0,
                                   **offset(dpsi_deg=20.0),
                                   **sinusoid("u", 8.0, 0.3))
        self.result = engine.run_from_config(self.config)

    def test_samples_are_the_controller_instants(self):
        t = self.result.data["t"].to_numpy()
        np.testing.assert_allclose(t, np.arange(201) * 0.05, atol=1e-12)

    def test_control_is_the_law_evaluated_on_the_sampled_state(self):
        data = self.result.data
        controller = engine.build_controller(self.config)
        for k in (0, 7, 150):
            state = data.loc[k, ["x", "y", "psi", "u", "v", "r"]].to_numpy(float)
            tau = controller(data.loc[k, "t"], state)
            self.assertEqual(tau[0], data.loc[k, "Tu"])
            self.assertEqual(tau[2], data.loc[k, "Tr"])

    def test_forces_are_held_between_samples(self):
        """Re-integrate one interval with the held force: same next state."""
        data = self.result.data
        dist = engine.build_disturbance(self.config.disturbance)
        k = 40
        x0 = data.loc[k, ["x", "y", "psi", "u", "v", "r"]].to_numpy(float)
        tau = np.array([data.loc[k, "Tu"], 0.0, data.loc[k, "Tr"]])
        t0, t1 = data.loc[k, "t"], data.loc[k + 1, "t"]
        sol = solve_ivp(
            lambda t, x: usv_model.vessel_rhs(t, x, lambda *_: tau.copy(), dist),
            (t0, t1), x0, rtol=1e-8, atol=1e-10, first_step=t1 - t0)
        x1 = data.loc[k + 1, ["x", "y", "psi", "u", "v", "r"]].to_numpy(float)
        np.testing.assert_allclose(sol.y[:, -1], x1, rtol=0, atol=1e-12)

    def test_fast_sampling_converges_to_the_continuous_controller(self):
        base = offset(dy=1.0)
        cont = engine.run_from_config(short_config(
            10.0, sim_mode=cfg.CONTINUOUS, **base))
        fast = engine.run_from_config(short_config(
            10.0, sim_sample_rate_hz=200.0, **base))
        cont_xy = cont.data[["t", "x", "y"]].assign(t=cont.data.t.round(6))
        fast_xy = fast.data[["t", "x", "y"]].assign(t=fast.data.t.round(6))
        joined = cont_xy.merge(fast_xy, on="t", suffixes=("", "_d"))
        self.assertGreater(len(joined), 100)
        err = np.hypot(joined.x - joined.x_d, joined.y - joined.y_d)
        self.assertLess(err.max(), 5e-3)

    def test_disturbance_columns(self):
        data = self.result.data
        expected = 8.0 * np.sin(2 * np.pi * 0.3 * data["t"])
        np.testing.assert_allclose(data["dTu"], expected, atol=1e-12)
        self.assertTrue((data["dTr"] == 0).all())


class PhasePlaneTests(unittest.TestCase):
    def test_sliding_variables_are_exactly_e_dot_plus_k_e(self):
        config = short_config(20.0, **offset(-1.0, 2.0, -40.0), surge_k_pos=0.6,
                              yaw_k_psi=2.0)
        data = engine.run_from_config(config).data
        np.testing.assert_allclose(
            data.s_u, data.e_u_dot + 0.6 * data.e_u, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            data.s_r, data.e_psi_dot + 2.0 * data.e_psi, rtol=0, atol=1e-12)
        # Definitions of the controller: s_u = u - u_cmd, s_r = r - r_cmd.
        np.testing.assert_allclose(data.s_u, data.u - data.u_cmd, atol=1e-12)
        np.testing.assert_allclose(data.s_r, data.r - data.r_cmd, atol=1e-12)
        self.assertTrue(data.sat_u.between(-1, 1).all())
        np.testing.assert_allclose(
            data.sat_r, np.clip(data.s_r / config.yaw.phi_r, -1, 1))

    def test_position_error_is_the_norm_of_its_components(self):
        data = engine.run_from_config(short_config(10.0, **offset(dy=2.0))).data
        np.testing.assert_allclose(
            data.e_p, np.hypot(data.e_u, data.e_cross), rtol=1e-12)


class RobustnessTests(unittest.TestCase):
    def test_invalid_config_never_reaches_the_simulator(self):
        config = short_config(surge_phi_u=0.01)
        bad = cfg.SimulationConfig(
            config.reference,
            cfg.SurgeSMCConfig(k_pos=0.35, lambda_u=1.2, k_s_u=0.35, phi_u=0.0),
            config.yaw, config.disturbance, config.simulation)
        with mock.patch.object(usv_model, "simulate") as sim, \
                mock.patch.object(engine, "solve_ivp") as ivp:
            with self.assertRaises(cfg.ConfigValidationError):
                engine.run_from_config(bad)
            sim.assert_not_called()
            ivp.assert_not_called()

    def test_wall_clock_limit(self):
        with self.assertRaises(engine.SimulationTimeoutError):
            engine.run_from_config(short_config(60.0), max_wall_time_s=1e-9)

    def test_results_are_tied_to_their_config(self):
        config = short_config(5.0)
        a, b = engine.run_from_config(config), engine.run_from_config(config)
        self.assertIs(a.config, config)
        self.assertNotEqual(a.run_id, b.run_id)
        np.testing.assert_array_equal(a.data.to_numpy(), b.data.to_numpy())


class TheoryTests(unittest.TestCase):
    """The design hints shown in the sidebar agree with the simulator."""

    def test_constant_disturbance_shift_matches_the_prediction(self):
        laps = dict(ref_laps=1.0)
        base = engine.run_from_config(cfg.form_to_config(
            dict(cfg.default_form(), **laps)))
        config = cfg.form_to_config(dict(cfg.preset_form("constant_dist"),
                                         **laps))
        disturbed = engine.run_from_config(config)
        predicted = {c["channel"]: c for c in cfg.design_checks(config)
                     if c["kind"] == "steady"}
        mask = base.data.t > 40.0
        for channel, s_name, e_name, phi, lam, k_s, k, z, d in (
            ("surge", "s_u", "e_u", 0.08, 1.2, 0.35, 0.35, cfg.PARAMS["z1"], 15),
            ("yaw", "s_r", "e_psi", 0.08, 2.0, 0.8, 3.0, cfg.PARAMS["z3"], 8),
        ):
            s_ss = (d / z) / (lam + k_s / phi)
            shift = (disturbed.data[s_name] - base.data[s_name])[mask].mean()
            self.assertAlmostEqual(shift, s_ss, delta=0.1 * s_ss, msg=channel)
            e_shift = (disturbed.data[e_name] - base.data[e_name])[mask].mean()
            self.assertAlmostEqual(e_shift, s_ss / k, delta=0.1 * s_ss / k,
                                   msg=channel)
            self.assertIn(f"{s_ss:.3g}", predicted[channel]["latex"])

    def test_chattering_appears_when_the_discrete_gain_exceeds_one(self):
        indices = {}
        for phi in (0.03, 0.012):          # Gamma_r = 0.57 and 1.37 at 50 Hz
            config = short_config(40.0, yaw_phi_r=phi)
            indices[phi] = engine.run_from_config(config).metrics
        self.assertLess(indices[0.03]["sat_r_tv"], 0.1)
        self.assertGreater(indices[0.012]["sat_r_tv"], 5.0)
        self.assertGreater(indices[0.012]["Tr_hf_rms"],
                           50 * indices[0.03]["Tr_hf_rms"])
        self.assertGreater(indices[0.012]["s_r_zc"], 5.0)


if __name__ == "__main__":
    unittest.main()
