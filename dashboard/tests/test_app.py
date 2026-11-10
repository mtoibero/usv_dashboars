"""End-to-end tests of the Streamlit app with streamlit.testing.AppTest.

The real simulator is used, with short simulated times to keep tests fast.
"""

import json
import unittest
from pathlib import Path
from unittest import mock

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover - streamlit not installed
    AppTest = None

from dashboard import config as cfg
from dashboard import engine

APP = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT = 120


@unittest.skipIf(AppTest is None, "streamlit is not installed")
class DashboardAppTests(unittest.TestCase):
    def start(self, t_final=10.0):
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        at.run()
        self.assertFalse(at.exception, at.exception)
        # Short runs: custom duration.
        at.radio(key="sim_duration_mode").set_value("manual").run()
        at.number_input(key="sim_t_final_s").set_value(t_final).run()
        return at

    def run_button(self, at):
        return next(b for b in at.button if b.label == "Run simulation")

    def click_run(self, at):
        self.run_button(at).click().run()
        self.assertFalse(at.exception, at.exception)
        return at

    def history(self, at):
        return at.session_state["history"]

    def test_app_starts_without_results(self):
        at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
        self.assertFalse(at.exception)
        self.assertFalse(self.run_button(at).disabled)
        self.assertEqual(len(self.history(at)), 0)
        self.assertTrue(any("press **Run simulation**" in i.value for i in at.info))
        self.assertEqual(at.session_state["smc_lambda_u"], 1.2)

    def test_widget_changes_do_not_run_simulations(self):
        at = self.start()
        with mock.patch.object(engine, "run_from_config") as run:
            at.number_input(key="smc_lambda_r").set_value(2.5).run()
            at.selectbox(key="dist_kind").set_value("constant").run()
            run.assert_not_called()
        self.assertEqual(len(self.history(at)), 0)

    def test_controller_parameter_reaches_the_simulation(self):
        at = self.start()
        at.number_input(key="smc_lambda_r").set_value(3.25).run()
        at.number_input(key="smc_phi_r").set_value(0.15).run()
        with mock.patch.object(engine, "run_from_config",
                               wraps=engine.run_from_config) as run:
            self.click_run(at)
        passed = run.call_args.args[0]
        self.assertEqual(passed.controller.lambda_r, 3.25)
        self.assertEqual(passed.controller.phi_r, 0.15)
        latest = self.history(at).latest
        self.assertEqual(latest.config.controller.lambda_r, 3.25)
        self.assertEqual(latest.config.simulation.t_final_s, 10.0)
        self.assertTrue(any("correspond to the current sidebar" in s.value
                            for s in at.success))
        self.assertEqual(len(at.metric), 5)

    def test_invalid_configuration_blocks_the_run(self):
        at = self.start()
        at.number_input(key="smc_phi_u").set_value(0.0).run()
        self.assertTrue(any(r"$\phi_u$ must be > 0" in e.value for e in at.error))
        self.assertTrue(self.run_button(at).disabled)
        # Even if a run request arrives, nothing is simulated.
        with mock.patch.object(engine, "run_from_config") as run:
            at.session_state["pending_run"] = {"label": ""}
            at.run()
            run.assert_not_called()
        self.assertEqual(len(self.history(at)), 0)
        self.assertTrue(any("Configuration is invalid" in e.value
                            for e in at.error))

    def test_repeated_runs_never_reuse_outdated_results(self):
        at = self.start()
        self.click_run(at)
        first = self.history(at).latest

        # Change a parameter without running: results are flagged as stale.
        at.number_input(key="smc_k_pos").set_value(0.7).run()
        self.assertEqual(self.history(at).latest.number, first.number)
        self.assertTrue(any("differ from the configuration of Run #1" in w.value
                            for w in at.warning))

        self.click_run(at)
        second = self.history(at).latest
        self.assertEqual(second.number, 2)
        self.assertEqual(second.config.controller.k_pos, 0.7)
        self.assertEqual(first.config.controller.k_pos, 0.35)
        self.assertNotEqual(second.result.run_id, first.result.run_id)
        self.assertNotEqual(second.result.metrics["lookahead_err_rms_m"],
                            first.result.metrics["lookahead_err_rms_m"])
        self.assertFalse(at.warning)
        # The previous run is retained for comparison.
        self.assertIs(self.history(at).previous, first)
        self.assertTrue(any(row.label == "Run A (baseline)"
                            for row in at.selectbox))

    def test_failed_run_keeps_old_results_clearly_labelled(self):
        at = self.start()
        self.click_run(at)
        at.number_input(key="smc_k_psi").set_value(4.0).run()
        with mock.patch.object(engine, "run_from_config",
                               side_effect=engine.SimulationError("diverged")):
            self.click_run(at)
        self.assertEqual(len(self.history(at)), 1)
        self.assertTrue(any("last run attempt failed" in e.value and
                            "diverged" in e.value and "Run #1" in e.value
                            for e in at.error))
        self.assertTrue(any("differ from the configuration of Run #1" in w.value
                            for w in at.warning))

    def test_loading_a_stored_configuration_restores_the_sidebar(self):
        at = self.start()
        self.click_run(at)
        at.number_input(key="smc_lambda_u").set_value(2.0).run()
        at.selectbox(key="dist_kind").set_value("constant").run()
        self.assertTrue(at.warning)
        load = next(b for b in at.button
                    if b.label == "Load its settings into the sidebar")
        load.click().run()
        self.assertEqual(at.session_state["smc_lambda_u"], 1.2)
        self.assertEqual(at.session_state["dist_kind"], "none")
        self.assertFalse(at.warning)

    def test_switching_display_modes_does_not_change_the_experiment(self):
        at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
        default = cfg.default_config()
        at.radio(key="ref_timing_mode").set_value("peak_speed").run()
        at.radio(key="sim_duration_mode").set_value("manual").run()
        at.radio(key="sim_initial_mode").set_value("custom").run()
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertFalse(at.warning)
        form = {key: at.session_state[key] for key in cfg.FORM_KEYS}
        current = cfg.form_to_config(form)
        self.assertTrue(cfg.same_experiment(default, current),
                        cfg.config_diff(default, current))

    def xy_traces(self, at):
        spec = json.loads(at.get("plotly_chart")[0].proto.spec)
        return [(t.get("name"), t.get("line", {}).get("color"))
                for t in spec["data"]]

    def test_xy_plot_overlays_only_the_previous_run(self):
        at = self.start(t_final=5.0)
        self.click_run(at)
        names = [n for n, _ in self.xy_traces(at)]
        self.assertFalse(any(n.startswith("Previous") for n in names))
        self.assertFalse(at.checkbox)

        for k_pos in (0.5, 0.7):
            at.number_input(key="smc_k_pos").set_value(k_pos).run()
            self.click_run(at)
        traces = self.xy_traces(at)
        previous = [t for t in traces if t[0].startswith("Previous")]
        self.assertEqual(previous, [("Previous: Run #2", "#c8c8c8")])
        self.assertEqual(traces[0][0], "Previous: Run #2")
        self.assertIn("Reference path", [n for n, _ in traces])
        # The history mechanism is unchanged: all runs are still stored.
        self.assertEqual([r.number for r in self.history(at).records], [1, 2, 3])
        # Parameters and metrics of the previous run are shown.
        self.assertTrue(any("Previous run: Run #2" in m.value
                            for m in at.markdown))
        tables = [df.value for df in at.dataframe]
        self.assertTrue(any("Previous #2" in t.columns and "Current #3" in t.columns
                            and "Max lookahead-point error [m]" in set(t["Metric"])
                            for t in tables if "Metric" in t.columns))
        self.assertTrue(any("Parameters changed (#2 → #3)" in m.value
                            for m in at.markdown))
        self.assertTrue(any(("controller.k_pos" in set(t["Parameter"]))
                            for t in tables if "Parameter" in t.columns
                            and "Previous" in t.columns))

        at.checkbox(key="xy_show_previous").uncheck().run()
        self.assertFalse(any(n.startswith("Previous")
                             for n, _ in self.xy_traces(at)))

    def test_hidden_widgets_keep_their_values(self):
        at = self.start()
        at.selectbox(key="dist_kind").set_value("constant").run()
        at.number_input(key="dist_const_dTr").set_value(7.5).run()
        at.selectbox(key="dist_kind").set_value("none").run()
        at.selectbox(key="dist_kind").set_value("constant").run()
        self.assertEqual(at.number_input(key="dist_const_dTr").value, 7.5)


if __name__ == "__main__":
    unittest.main()
