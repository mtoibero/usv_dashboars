"""End-to-end tests of the Streamlit app (streamlit.testing.AppTest).

The real simulator is used with short runs (0.1 laps = 14 s).
"""

import json
import unittest
from pathlib import Path
from unittest import mock

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover - streamlit not installed
    AppTest = None

from dashboard02_smc import config as cfg
from dashboard02_smc import engine
from dashboard02_smc import figures
from dashboard02_smc.tests.helpers import constant

APP = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT = 180
P = "smc2_"


@unittest.skipIf(AppTest is None, "streamlit is not installed")
class AppTests(unittest.TestCase):
    def start(self, laps=0.1, sliders=False):
        """App with short runs; number boxes (default) or sliders."""
        at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
        self.assertFalse(at.exception, at.exception)
        at.number_input(key=P + "ref_laps").set_value(laps).run()
        if sliders:
            at.toggle(key=P + "use_sliders").set_value(True).run()
        return at

    def set_form(self, at, **values):
        """Set widget values through the session state (as callbacks do)."""
        for name, value in values.items():
            at.session_state[P + name] = value
        at.run()
        self.assertFalse(at.exception, at.exception)

    def run_button(self, at):
        return next(b for b in at.button if b.label == "Run simulation")

    def click_run(self, at):
        self.run_button(at).click().run()
        self.assertFalse(at.exception, at.exception)
        return at

    def history(self, at):
        return at.session_state[P + "history"]

    def chart_traces(self, at, index=0):
        spec = json.loads(at.get("plotly_chart")[index].proto.spec)
        return [(t.get("name") or "", t.get("line", {}).get("color"))
                for t in spec["data"]]

    # -- start-up ------------------------------------------------------------
    def test_starts_without_results_and_with_the_equations(self):
        at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
        self.assertFalse(at.exception)
        self.assertEqual(len(self.history(at)), 0)
        self.assertFalse(self.run_button(at).disabled)
        self.assertTrue(any("press **Run simulation**" in i.value
                            for i in at.info))
        latex = [m.value for m in at.markdown if r"\begin{aligned}" in m.value]
        self.assertTrue(any(r"\mathrm{sat}(s_u/\phi_u)" in v for v in latex))
        self.assertTrue(any(r"\mathrm{sat}(s_r/\phi_r)" in v for v in latex))
        self.assertEqual(at.session_state[P + "surge_lambda_u"], 1.2)
        # Number boxes by default (as the first dashboard); sliders on demand.
        self.assertEqual(len(at.slider), 0)
        n_boxes = len(at.number_input)
        self.assertGreater(n_boxes, 10)
        at.toggle(key=P + "use_sliders").set_value(True).run()
        self.assertEqual(len(at.slider), n_boxes)
        self.assertEqual(len(at.number_input), 0)
        # Sidebar structure of the first dashboard.
        labels = [e.label for e in at.sidebar.expander]
        for name in ("1 - Simulation settings", "2 - Reference trajectory",
                     "3 - Controller", "4 - Disturbance"):
            self.assertIn(name, labels)

    def test_all_widget_keys_are_namespaced(self):
        at = self.start(sliders=True)
        self.click_run(at)
        widgets = (list(at.slider) + list(at.number_input) + list(at.radio)
                   + list(at.selectbox) + list(at.toggle) + list(at.checkbox)
                   + list(at.text_input))
        self.assertTrue(widgets)
        for widget in widgets:
            self.assertTrue(widget.key and widget.key.startswith(P),
                            widget.key)

    # -- only the button runs simulations ---------------------------------------
    def test_parameter_changes_never_run_simulations(self):
        at = self.start()
        with mock.patch.object(engine, "run_from_config") as run:
            at.number_input(key=P + "yaw_phi_r").set_value(0.02).run()
            at.number_input(key=P + "surge_k_s_u").set_value(0.5).run()
            at.selectbox(key=P + "dist_kind").set_value("sinusoidal").run()
            at.selectbox(key=P + "dist_kind").set_value("combined").run()
            at.radio(key=P + "sim_duration_mode").set_value("manual").run()
            at.radio(key=P + "ref_mode").set_value("amp_speed").run()
            at.radio(key=P + "sim_mode").set_value(cfg.CONTINUOUS).run()
            at.toggle(key=P + "use_sliders").set_value(True).run()
            at.selectbox(key=P + "preset").set_value("chattering").run()
            next(b for b in at.button if b.label == "Apply preset").click().run()
            run.assert_not_called()
        self.assertEqual(len(self.history(at)), 0)

    def test_display_controls_never_run_simulations(self):
        at = self.start()
        self.click_run(at)
        at.number_input(key=P + "surge_lambda_u").set_value(2.0).run()
        self.click_run(at)
        with mock.patch.object(engine, "run_from_config") as run:
            at.slider(key=P + "window").set_value((2.0, 10.0)).run()
            at.selectbox(key=P + "cmp_number").set_value(0).run()
            at.checkbox(key=P + "show_cmd").uncheck().run()
            run.assert_not_called()
        self.assertEqual(len(self.history(at)), 2)

    # -- the parameters reach the simulation ----------------------------------
    def test_all_controller_parameters_reach_the_simulation(self):
        at = self.start()
        values = {"surge_lambda_u": 1.7, "surge_k_s_u": 0.45,
                  "surge_phi_u": 0.06, "surge_k_pos": 0.4,
                  "yaw_lambda_r": 2.4, "yaw_k_s_r": 0.9, "yaw_phi_r": 0.07,
                  "yaw_k_psi": 2.5}
        for name, value in values.items():
            at.number_input(key=P + name).set_value(value).run()
        at.selectbox(key=P + "dist_kind").set_value("sinusoidal").run()
        at.number_input(key=P + "dist_sin_amp_u").set_value(12.0).run()
        at.number_input(key=P + "dist_sin_amp_r").set_value(6.0).run()
        at.number_input(key=P + "ref_frequency_hz").set_value(0.01).run()
        at.number_input(key=P + "ref_peak_speed_mps").set_value(0.8).run()
        with mock.patch.object(engine, "run_from_config",
                               wraps=engine.run_from_config) as run:
            self.click_run(at)
        config = run.call_args.args[0]
        self.assertEqual(config.surge, cfg.SurgeSMCConfig(
            k_pos=0.4, lambda_u=1.7, k_s_u=0.45, phi_u=0.06))
        self.assertEqual(config.yaw, cfg.YawSMCConfig(
            k_psi=2.5, lambda_r=2.4, k_s_r=0.9, phi_r=0.07))
        self.assertEqual(config.disturbance.kind, "sinusoidal")
        self.assertEqual(config.disturbance.sin_amplitude, (12.0, 4.0, 6.0))
        self.assertAlmostEqual(config.reference.period_s, 100.0)
        self.assertAlmostEqual(config.reference.peak_speed_mps, 0.8)
        self.assertIs(self.history(at).latest.config, config)
        self.assertTrue(any("correspond to the current sidebar" in s.value
                            for s in at.success))

    # -- current run bright, previous run gray --------------------------------
    def test_previous_run_is_overlaid_in_gray_with_its_parameters(self):
        at = self.start()
        self.click_run(at)
        self.assertFalse(any("Previous" in n for n, _ in self.chart_traces(at)))
        at.number_input(key=P + "yaw_phi_r").set_value(0.03).run()
        self.click_run(at)
        for index in range(len(at.get("plotly_chart")) - 1):
            traces = self.chart_traces(at, index)
            names = [n for n, _ in traces]
            if not any("Run #" in n for n in names):
                continue
            prev = [c for n, c in traces if "Previous · Run #1" in n]
            cur = [c for n, c in traces if "Current · Run #2" in n]
            self.assertTrue(prev and cur, index)
            self.assertEqual(set(prev), {figures.PREVIOUS_COLOR})
            self.assertEqual(set(cur), {figures.CURRENT_COLOR})
        names = [n for index in range(len(at.get("plotly_chart")))
                 for n, _ in self.chart_traces(at, index)]
        self.assertTrue(any("Current" in n and "φ<sub>r</sub>=0.03" in n
                            for n in names))
        self.assertTrue(any("Previous" in n and "φ<sub>r</sub>=0.08" in n
                            for n in names))
        # LaTeX run cards, with the changed parameter highlighted.
        cards = [m.value for m in at.markdown if r"\text{Surge: }" in m.value]
        self.assertEqual(len(cards), 2)
        self.assertIn(r"\boldsymbol{\phi_r = 0.03}", cards[0])
        # Comparison can be switched off.
        at.selectbox(key=P + "cmp_number").set_value(0).run()
        self.assertFalse(any("Previous" in n for n, _ in self.chart_traces(at)))

    def test_xy_plot_comes_before_the_comparison_of_the_runs(self):
        """After a run the XY trajectory is the first thing on the page; the
        parameter cards and key indices are shown below the plots."""
        at = self.start()
        self.click_run(at)
        at.number_input(key=P + "yaw_phi_r").set_value(0.03).run()
        self.click_run(at)
        order = []
        for node in at.main:
            if node.type == "plotly_chart":
                order.append("chart")
            elif node.type == "markdown" and r"\text{Surge: }" in node.value:
                order.append("card")
            elif node.type == "metric":
                order.append("metric")
        self.assertEqual(order[0], "chart")
        first_chart = json.loads(at.get("plotly_chart")[0].proto.spec)
        self.assertEqual(first_chart["layout"]["xaxis"]["title"]["text"],
                         "x [m]")                      # the XY trajectory
        last_chart = max(i for i, o in enumerate(order) if o == "chart")
        self.assertTrue(all(o != "chart" for o in order[last_chart + 1:]))
        self.assertGreater(order.index("card"), last_chart)
        self.assertGreater(order.index("metric"), order.index("card"))
        self.assertTrue(any("Changes: Run #2 vs Run #1" in m.value
                            for m in at.markdown))

    def test_main_tab_has_the_xy_plot_the_time_window_and_the_phase_planes(self):
        at = self.start()
        self.click_run(at)
        tabs = at.tabs
        self.assertEqual(tabs[0].label, "Trajectory & sliding surfaces")
        self.assertNotIn("Phase plane (s-surface)", [t.label for t in tabs])
        main = tabs[0]
        self.assertEqual([s.key for s in main.slider], [P + "window"])
        charts = main.get("plotly_chart")
        self.assertEqual(len(charts), 3)
        specs = [json.loads(c.proto.spec) for c in charts]
        self.assertEqual(specs[0]["layout"]["xaxis"]["title"]["text"], "x [m]")
        self.assertIn("e<sub>u</sub>", specs[1]["layout"]["xaxis"]["title"]["text"])
        self.assertIn("e<sub>ψ</sub>", specs[2]["layout"]["xaxis"]["title"]["text"])
        # The window reaches every plot of the main tab.
        at.slider(key=P + "window").set_value((2.0, 6.0)).run()
        specs = [json.loads(c.proto.spec) for c in at.tabs[0].get("plotly_chart")]
        texts = [a.get("text") for a in specs[0]["layout"].get("annotations", [])]
        self.assertIn("t = 2.0 s", texts)
        self.assertIn("t = 6.0 s", texts)
        self.assertTrue(any("Time window" in c.value for c in at.caption))

    def test_results_are_never_reused_when_parameters_change(self):
        at = self.start()
        self.click_run(at)
        first = self.history(at).latest
        at.number_input(key=P + "surge_k_pos").set_value(0.7).run()
        self.assertEqual(self.history(at).latest.number, 1)
        self.assertTrue(any("differ from Run #1" in w.value for w in at.warning))
        self.click_run(at)
        second = self.history(at).latest
        self.assertEqual(second.config.surge.k_pos, 0.7)
        self.assertEqual(first.config.surge.k_pos, 0.35)
        self.assertNotEqual(second.result.run_id, first.result.run_id)
        self.assertFalse(any("differ from" in w.value for w in at.warning))
        self.assertIs(self.history(at).previous, first)

    # -- invalid input and failures --------------------------------------------
    def test_invalid_configuration_blocks_the_run(self):
        at = self.start()
        at.number_input(key=P + "surge_phi_u").set_value(0.0).run()
        self.assertTrue(any(r"$\phi_u$" in e.value for e in at.error))
        self.assertTrue(self.run_button(at).disabled)
        with mock.patch.object(engine, "run_from_config") as run:
            at.session_state[P + "pending_run"] = {"label": ""}
            at.run()
            run.assert_not_called()
        self.assertEqual(len(self.history(at)), 0)
        self.assertTrue(any("Configuration is invalid" in e.value
                            for e in at.error))

    def test_failed_run_keeps_old_results_clearly_labelled(self):
        at = self.start()
        self.click_run(at)
        at.number_input(key=P + "yaw_k_psi").set_value(4.0).run()
        with mock.patch.object(engine, "run_from_config",
                               side_effect=engine.SimulationError("diverged")):
            self.click_run(at)
        self.assertEqual(len(self.history(at)), 1)
        self.assertTrue(any("last run failed" in e.value and "diverged"
                            in e.value and "Run #1" in e.value
                            for e in at.error))
        self.assertTrue(any("differ from Run #1" in w.value for w in at.warning))

    # -- input conveniences ------------------------------------------------------
    def test_input_style_and_reference_mode_keep_the_experiment(self):
        at = self.start(sliders=True)
        at.slider(key=P + "yaw_phi_r").set_value(0.05).run()
        before = cfg.form_to_config(
            {k: at.session_state[P + k] for k in cfg.FORM_KEYS})
        at.toggle(key=P + "use_sliders").set_value(False).run()
        self.assertEqual(at.number_input(key=P + "yaw_phi_r").value, 0.05)
        for mode in ("amp_freq", "amp_speed", "freq_speed"):
            at.radio(key=P + "ref_mode").set_value(mode).run()
            after = cfg.form_to_config(
                {k: at.session_state[P + k] for k in cfg.FORM_KEYS})
            self.assertEqual(cfg.config_diff(before, after), [], mode)
        self.assertFalse(at.error)

    def test_changing_frequency_keeps_the_derived_fields_consistent(self):
        at = self.start()
        at.number_input(key=P + "ref_frequency_hz").set_value(0.01).run()
        amplitude = at.session_state[P + "ref_amplitude_m"]
        speed = at.session_state[P + "ref_peak_speed_mps"]
        self.assertAlmostEqual(cfg.peak_speed(amplitude, 100.0), speed)

    def test_loading_a_stored_run_restores_the_sidebar(self):
        at = self.start()
        self.click_run(at)
        at.number_input(key=P + "surge_lambda_u").set_value(3.0).run()
        at.selectbox(key=P + "dist_kind").set_value("constant").run()
        self.assertTrue(at.warning)
        next(b for b in at.button
             if b.label == "Load its settings into the sidebar").click().run()
        self.assertEqual(at.session_state[P + "surge_lambda_u"], 1.2)
        self.assertEqual(at.session_state[P + "dist_kind"], "none")
        self.assertFalse(at.warning)

    def test_gain_hint_follows_the_disturbance_and_sets_the_gain(self):
        at = self.start()
        at.number_input(key=P + "yaw_k_s_r").set_value(0.05).run()
        at.number_input(key=P + "yaw_lambda_r").set_value(0.5).run()
        hint_titles = lambda: [m.value for m in at.markdown
                               if "Gain hint for the yaw SMC" in m.value]
        self.assertTrue(hint_titles()[0].startswith("✅"))
        with mock.patch.object(engine, "run_from_config") as run:
            self.set_form(at, **constant(d_r=12.0))
            self.assertTrue(hint_titles()[0].startswith("❌"))
            self.assertTrue(any("sliding condition violated" in w.value
                                for w in at.warning))
            button = next(b for b in at.button
                          if b.key == P + "use_ks_yaw_disturbance")
            button.click().run()
            run.assert_not_called()
        self.assertAlmostEqual(at.session_state[P + "yaw_k_s_r"], 0.13)
        self.assertTrue(hint_titles()[0].startswith("✅"))
        self.assertFalse(any(b.key == P + "use_ks_yaw_disturbance"
                             for b in at.button))
        self.assertFalse(any("sliding condition violated" in w.value
                             for w in at.warning))

    def test_preset_sets_the_sidebar(self):
        at = self.start()
        at.selectbox(key=P + "preset").set_value("reaching").run()
        next(b for b in at.button if b.label == "Apply preset").click().run()
        self.assertEqual(at.session_state[P + "init_dy_m"], 3.0)
        self.assertEqual(at.session_state[P + "init_dpsi_deg"], 30.0)
        self.assertEqual(at.session_state[P + "sim_initial_mode"], "offset")
        next(b for b in at.button
             if b.label == "Reset all settings to defaults").click().run()
        self.assertEqual(at.session_state[P + "sim_initial_mode"], "reference")
        self.assertEqual(at.session_state[P + "init_dy_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
