"""LaTeX snippets and the run history."""

import unittest

from dashboard02_smc import config as cfg
from dashboard02_smc import latex as tex
from dashboard02_smc.session import RunHistory, staleness


class LatexTests(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(tex.num(0.35), "0.35")
        self.assertEqual(tex.num(0.0), "0")
        self.assertEqual(tex.num(2.5e-4), r"2.5\times 10^{-4}")
        self.assertEqual(tex.num(float("nan")), r"\text{n/a}")

    def test_disturbance_formula(self):
        d = cfg.DisturbanceConfig("sinusoidal", sin_amplitude=(0, 0, 5.0),
                                  sin_frequency_hz=(0, 0, 0.1),
                                  sin_phase_rad=(0, 0, -0.5),
                                  sin_bias=(0, 0, 2.0))
        lines = tex.disturbance_lines(d)
        self.assertEqual(len(lines), 1)
        self.assertIn(r"2 + 5\sin(2\pi\cdot 0.1\,t - 0.5)", lines[0])
        self.assertIn(r"\mathrm{N\,m}", lines[0])
        self.assertIn(r"F = 25", tex.disturbance_lines(cfg.form_to_config(
            dict(cfg.default_form(), dist_kind="directional")).disturbance)[0])
        self.assertEqual(tex.disturbance_lines(cfg.DisturbanceConfig()),
                         [r"\boldsymbol\delta = \boldsymbol 0"])

    def test_run_card_highlights_changed_parameters(self):
        a = cfg.default_config()
        b = cfg.form_to_config(dict(cfg.default_form(), yaw_phi_r=0.02))
        changed = [k for k, _o, _n in cfg.config_diff(a, b)]
        card = tex.run_card(b, changed)
        self.assertIn(r"\textcolor{" + tex.CHANGED_TEX_COLOR +
                      r"}{\boldsymbol{\phi_r = 0.02}}", card)
        self.assertIn(r"\lambda_u = 1.2", card)
        self.assertTrue(card.startswith("$$") and card.endswith("$$"))
        self.assertEqual(card.count(r"\begin{aligned}"), 1)

    def test_change_list(self):
        a = cfg.default_config()
        b = cfg.form_to_config(dict(cfg.default_form(), surge_k_s_u=0.5,
                                    sim_mode=cfg.CONTINUOUS))
        text = tex.change_list(cfg.config_diff(a, b))
        self.assertIn(r"$k_{s,u}$: $0.35\ \mathrm{m/s^2} \rightarrow "
                      r"0.5\ \mathrm{m/s^2}$", text)
        self.assertIn(r"\text{continuous}", text)
        self.assertNotIn("f_s", text)      # 0 only marks "continuous"


def _balanced(text):
    depth = 0
    for i, char in enumerate(text):
        if char == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif char == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def _unescaped_percent(text):
    return any(c == "%" and (i == 0 or text[i - 1] != "\\")
               for i, c in enumerate(text))


class LatexValidityTests(unittest.TestCase):
    """Every generated LaTeX string is well formed for KaTeX.

    ('%' starts a comment in LaTeX and silently breaks the rest of a block.)
    """

    def strings(self):
        from dashboard02_smc import metrics as met
        configs = [cfg.form_to_config(cfg.preset_form(name))
                   for name in cfg.PRESETS]
        configs.append(cfg.form_to_config(dict(
            cfg.default_form(), sim_mode=cfg.CONTINUOUS,
            dist_kind="sinusoidal", dist_sin_phase_v=-0.5)))
        for kind in cfg.DIST_KINDS:
            configs.append(cfg.form_to_config(dict(cfg.default_form(),
                                                   dist_kind=kind)))
        out = []
        for config in configs:
            changed = [k for k, _o, _n in
                       cfg.config_diff(cfg.default_config(), config)]
            out.append(tex.run_card(config, changed))
            out.append(tex.change_list(cfg.config_diff(cfg.default_config(),
                                                       config)))
            out.extend(c["latex"] for c in cfg.design_checks(config))
            from dashboard02_smc import tuning
            for hint in tuning.gain_hints(config).values():
                out.extend(tex.gain_hint_lines(
                    hint, tuning.suggested_value(hint.k_suggested)))
            out.extend(tex.actuator_lines(tuning.actuator_check(config)))
        for key, (_g, sym, _l, unit, _b, defn) in met.INDEX_INFO.items():
            out.append(rf"{sym} &= {defn} && [{tex.unit(unit)}]")
        return out

    def test_braces_are_balanced_and_percent_is_escaped(self):
        import re
        for text in self.strings():
            self.assertTrue(_balanced(text), text)
            # '%' matters only inside math; in plain Markdown it is harmless.
            display = re.findall(r"\$\$(.*?)\$\$", text, flags=re.S)
            rest = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
            self.assertEqual(rest.count("$") % 2, 0, f"unbalanced $: {text}")
            inline = re.findall(r"\$(.+?)\$", rest, flags=re.S)
            for part in display + inline:
                self.assertFalse(_unescaped_percent(part), text)
            if "\\begin{aligned}" in text:
                self.assertTrue(display, text)

    def test_the_check_sees_math_in_both_forms(self):
        import re
        bad = ["$$\na = 5%\n$$", "x $b = 5%$ y"]
        for text in bad:
            display = re.findall(r"\$\$(.*?)\$\$", text, flags=re.S)
            rest = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
            parts = display + re.findall(r"\$(.+?)\$", rest, flags=re.S)
            self.assertTrue(any(_unescaped_percent(p) for p in parts), text)

    def test_units(self):
        self.assertEqual(tex.unit("%"), r"\mathrm{\%}")
        self.assertEqual(tex.unit("m/s²"), r"\mathrm{m/s^2}")
        self.assertEqual(tex.unit("N m"), r"\mathrm{N\,m}")
        self.assertEqual(tex.unit("-"), "")


class HistoryTests(unittest.TestCase):
    class Result:
        def __init__(self, config):
            self.config = config

    def test_bounded_history_and_previous(self):
        history = RunHistory(max_runs=3)
        for _ in range(5):
            history.add(self.Result(cfg.default_config()))
        self.assertEqual([r.number for r in history.records], [3, 4, 5])
        self.assertEqual(history.previous.number, 4)
        history.clear()
        self.assertIsNone(history.latest)
        self.assertEqual(history.add(self.Result(None)).number, 6)

    def test_staleness(self):
        history = RunHistory()
        record = history.add(self.Result(cfg.default_config()), "label")
        self.assertEqual(record.title, "Run #1 – label")
        self.assertEqual(staleness(record, cfg.default_config()), (False, []))
        changed = cfg.form_to_config(dict(cfg.default_form(), yaw_k_psi=4.0))
        stale, diff = staleness(record, changed)
        self.assertTrue(stale)
        self.assertEqual(diff, [("yaw.k_psi", 3.0, 4.0)])
        self.assertEqual(staleness(record, None), (True, []))


if __name__ == "__main__":
    unittest.main()
