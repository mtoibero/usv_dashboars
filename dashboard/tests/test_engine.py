"""Engine: connection to the real simulator and consistency with it."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from dashboard import config as cfg
from dashboard import engine

import controllers
import disturbances
import run_simulation
import usv_model


def short_form(t_final=20.0, **changes):
    form = dict(cfg.default_form(), sim_duration_mode="manual",
                sim_t_final_s=t_final, sim_dt_s=0.05)
    form.update(changes)
    return form


def short_config(t_final=20.0, **changes):
    return cfg.form_to_config(short_form(t_final, **changes))


class ParameterPropagationTests(unittest.TestCase):
    def test_smc_parameters_reach_smc_figure_eight(self):
        config = short_config(smc_lambda_r=3.3, smc_k_s_u=0.7, smc_phi_r=0.2,
                              smc_lookahead_L=1.5, ref_amplitude_m=12.0)
        with mock.patch.object(controllers, "smc_figure_eight",
                               wraps=controllers.smc_figure_eight) as factory:
            engine.build_controller(config)
        kwargs = factory.call_args.kwargs
        self.assertEqual(kwargs["lambda_r"], 3.3)
        self.assertEqual(kwargs["k_s_u"], 0.7)
        self.assertEqual(kwargs["phi_r"], 0.2)
        self.assertEqual(kwargs["lookahead_L"], 1.5)
        self.assertEqual(kwargs["amplitude"], 12.0)
        self.assertEqual(kwargs["period"], controllers.FIGURE_EIGHT_PERIOD)

    def test_feedback_parameters_reach_feedback_figure_eight(self):
        config = short_config(ctrl_kind="feedback", fb_k_r=99.0)
        with mock.patch.object(controllers, "feedback_figure_eight",
                               wraps=controllers.feedback_figure_eight) as factory:
            engine.build_controller(config)
        self.assertEqual(factory.call_args.kwargs["k_r"], 99.0)

    def test_changing_a_gain_changes_the_simulation(self):
        a = engine.run_from_config(short_config())
        b = engine.run_from_config(short_config(smc_k_pos=0.8))
        self.assertEqual(b.config.controller.k_pos, 0.8)
        self.assertFalse(np.allclose(a.data["x"], b.data["x"]))

    def test_trajectory_parameters_are_used(self):
        result = engine.run_from_config(short_config(
            t_final=50.0, ref_amplitude_m=8.0, ref_period_s=100.0))
        self.assertAlmostEqual(result.data["x_ref"].max(), 8.0, places=3)
        self.assertAlmostEqual(result.data["x"].iloc[0], 0.0)
        self.assertLess(result.metrics["lookahead_err_max_m"], 1.0)


class ConsistencyWithSimulatorTests(unittest.TestCase):
    def test_matches_direct_call_of_the_python_simulator(self):
        config = short_config(t_final=30.0)
        result = engine.run_from_config(config)
        t, state = usv_model.simulate(
            state0=controllers.figure_eight_initial_state(),
            tau_fun=controllers.smc_figure_eight(),
            t_final=30.0, dt=0.05,
            disturbance_fun=disturbances.no_disturbance,
        )
        np.testing.assert_array_equal(result.data["t"].to_numpy(), t)
        np.testing.assert_array_equal(
            result.data[["x", "y", "psi", "u", "v", "r"]].to_numpy(), state)

    def test_feedback_defaults_equal_tau_figure_eight(self):
        config = short_config(ctrl_kind="feedback", dist_kind="wind")
        result = engine.run_from_config(config)
        _, state = usv_model.simulate(
            state0=controllers.figure_eight_initial_state(),
            tau_fun=controllers.tau_figure_eight, t_final=20.0, dt=0.05,
            disturbance_fun=disturbances.wind_like_disturbance,
        )
        np.testing.assert_array_equal(
            result.data[["x", "y", "psi", "u", "v", "r"]].to_numpy(), state)

    def test_disturbance_defaults_reproduce_run_simulation_options(self):
        rng = np.random.default_rng(0)
        for kind in ("none", "constant", "sinusoidal", "directional", "wind",
                     "combined"):
            built = engine.build_disturbance(
                cfg.form_to_config(dict(cfg.default_form(),
                                        dist_kind=kind)).disturbance)
            reference = run_simulation.DISTURBANCE_OPTIONS[kind]
            for t in np.linspace(0.0, 300.0, 25):
                state = rng.normal(size=6)
                np.testing.assert_array_equal(
                    built(t, state), reference(t, state), err_msg=kind)

    def _run_cli(self, disturbance_mode, laps):
        """Run the unchanged run_simulation.main() with the dashboard scope
        (no obstacles, CBF or governor) and return its CSV."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "cli.csv"
            patches = dict(
                CSV_PATH=csv_path, FIGURES_DIR=Path(tmp) / "figures",
                ENABLE_OBSTACLES=False, ENABLE_CBF_AVOIDANCE=False,
                ENABLE_CBF_DIAGNOSTICS=False, ENABLE_REFERENCE_GOVERNOR=False,
                SAVE_ANIMATION=False, DISTURBANCE_MODE=disturbance_mode,
                FIGURE_EIGHT_LAPS=laps,
                plot_results=lambda *args, **kwargs: {},
            )
            with contextlib.ExitStack() as stack:
                for name, value in patches.items():
                    stack.enter_context(
                        mock.patch.object(run_simulation, name, value))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                run_simulation.main()
            return pd.read_csv(csv_path)

    def test_matches_the_command_line_simulator(self):
        laps = 0.25
        for mode in ("none", "constant", "sinusoidal", "combined"):
            with self.subTest(disturbance=mode):
                cli = self._run_cli(mode, laps)
                form = dict(cfg.default_form(), sim_laps=laps, dist_kind=mode)
                result = engine.run_from_config(cfg.form_to_config(form))
                self.assertEqual(len(cli), len(result.data))
                for column in engine.CLI_CSV_COLUMNS:
                    np.testing.assert_allclose(
                        result.data[column].to_numpy(), cli[column].to_numpy(),
                        rtol=0, atol=1e-12, err_msg=f"{mode}: {column}")


class ResultIntegrityTests(unittest.TestCase):
    def test_repeated_runs_are_fresh_and_deterministic(self):
        config = short_config()
        a = engine.run_from_config(config)
        b = engine.run_from_config(config)
        self.assertNotEqual(a.run_id, b.run_id)
        self.assertIsNot(a.data, b.data)
        pd.testing.assert_frame_equal(a.data, b.data)
        self.assertIs(a.config, config)

    def test_invalid_config_never_reaches_the_simulator(self):
        bad = short_config(smc_phi_u=0.0)
        with mock.patch.object(usv_model, "simulate") as simulate:
            with self.assertRaises(cfg.ConfigValidationError) as ctx:
                engine.run_from_config(bad)
        simulate.assert_not_called()
        self.assertIn("phi_u", str(ctx.exception))

    def test_wall_clock_guard(self):
        with self.assertRaises(engine.SimulationTimeoutError):
            engine.run_from_config(short_config(), max_wall_time_s=0.0)

    def test_solver_failure_is_reported(self):
        with mock.patch.object(usv_model, "simulate",
                               side_effect=RuntimeError("step size too small")):
            with self.assertRaises(engine.SimulationError) as ctx:
                engine.run_from_config(short_config())
        self.assertIn("step size too small", str(ctx.exception))

    def test_csv_columns_follow_save_csv(self):
        config = short_config(t_final=5.0)
        result = engine.run_from_config(config)
        exported = pd.read_csv(io.BytesIO(result.to_csv_bytes()))
        self.assertEqual(list(exported.columns[:len(engine.CLI_CSV_COLUMNS)]),
                         engine.CLI_CSV_COLUMNS)
        controller = engine.build_controller(config)
        t = result.data["t"].to_numpy()
        state = result.data[["x", "y", "psi", "u", "v", "r"]].to_numpy()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "save_csv.csv"
            run_simulation.save_csv(path, t, state, controller=controller)
            saved = pd.read_csv(path)
        self.assertEqual(list(saved.columns), engine.CLI_CSV_COLUMNS)
        np.testing.assert_allclose(
            exported[engine.CLI_CSV_COLUMNS].to_numpy(), saved.to_numpy(),
            rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
