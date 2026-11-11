"""Figures: current run bright on top of the comparison run in gray."""

import unittest

import numpy as np

from dashboard02_smc import config as cfg
from dashboard02_smc import engine, figures
from dashboard02_smc.session import RunHistory

from usv_model import TR_LIMITS, TU_LIMITS


def run(**changes):
    form = dict(cfg.default_form(), ref_laps=15.0 / 140.0)
    form.update(changes)
    return engine.run_from_config(cfg.form_to_config(form))


class OverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        history = RunHistory()
        history.add(run())
        history.add(run(yaw_phi_r=0.02, surge_lambda_u=2.5, init_dy_m=1.0,
                        sim_initial_mode="offset"),
                    label="thin yaw layer")
        cls.previous, cls.current = history.previous, history.latest

    def builders(self):
        c, p = self.current, self.previous
        return {
            "xy": figures.xy_figure(c, p),
            "errors": figures.errors_figure(c, p),
            "controls": figures.controls_figure(c, p),
            "phase_u": figures.phase_figure(c, p, "surge"),
            "phase_r": figures.phase_figure(c, p, "yaw"),
            "dist": figures.disturbance_figure(c, p),
        }

    def test_previous_is_gray_underneath_and_current_bright(self):
        for name, fig in self.builders().items():
            names = [t.name or "" for t in fig.data]
            prev = [i for i, n in enumerate(names) if "Previous · Run #1" in n]
            cur = [i for i, n in enumerate(names) if "Current · Run #2" in n]
            self.assertTrue(prev and cur, name)
            self.assertLess(max(prev), min(cur), name)    # drawn first
            for i in prev:
                self.assertEqual(fig.data[i].line.color, figures.PREVIOUS_COLOR)
            for i in cur:
                self.assertEqual(fig.data[i].line.color, figures.CURRENT_COLOR)

    def test_legends_show_the_parameters_of_each_run(self):
        fig = figures.errors_figure(self.current, self.previous)
        legend = [t.name for t in fig.data if t.showlegend is not False
                  and t.name]
        cur = next(n for n in legend if "Current" in n)
        prev = next(n for n in legend if "Previous" in n)
        self.assertIn("φ<sub>r</sub>=0.02", cur)
        self.assertIn("λ<sub>u</sub>=2.5", cur)
        self.assertIn("thin yaw layer", cur)
        self.assertIn("φ<sub>r</sub>=0.08", prev)
        self.assertIn("λ<sub>u</sub>=1.2", prev)
        phase = figures.phase_figure(self.current, self.previous, "yaw")
        self.assertTrue(any("k<sub>ψ</sub>=3" in (t.name or "")
                            for t in phase.data))

    def test_runs_share_the_same_axes(self):
        fig = figures.errors_figure(self.current, self.previous)
        for name in ("e_p", "s_r"):
            axes = {(t.xaxis, t.yaxis) for t in fig.data
                    if t.name and "Run #" in t.name and len(t.x) > 5}
            self.assertTrue(axes)
        by_run = {}
        for t in fig.data:
            if t.name and "Run #" in t.name:
                by_run.setdefault("cur" if "Current" in t.name else "prev",
                                  []).append((t.xaxis, t.yaxis))
        self.assertEqual(by_run["cur"], by_run["prev"])

    def test_phase_plot_band_is_the_boundary_layer(self):
        fig = figures.phase_figure(self.current, self.previous, "yaw")
        band = next(t for t in fig.data
                    if t.name.startswith("Current: boundary layer"))
        e, e_dot = np.array(band.x), np.array(band.y)
        s = e_dot + 3.0 * e
        np.testing.assert_allclose(np.abs(s), 0.02, atol=1e-12)
        prev_band = next(t for t in fig.data
                         if t.name.startswith("Previous: boundary layer"))
        s_prev = np.array(prev_band.y) + 3.0 * np.array(prev_band.x)
        np.testing.assert_allclose(np.abs(s_prev), 0.08, atol=1e-12)

    def test_force_axes_are_fixed_to_the_thruster_limits(self):
        fig = figures.controls_figure(self.current, self.previous)
        lo, hi = fig.layout.yaxis.range
        self.assertLess(lo, TU_LIMITS[0])
        self.assertGreater(hi, TU_LIMITS[1])
        lo, hi = fig.layout.yaxis2.range
        self.assertLess(lo, TR_LIMITS[0])
        self.assertGreater(hi, TR_LIMITS[1])

    def test_without_comparison_only_the_current_run_is_drawn(self):
        fig = figures.xy_figure(self.current, None)
        self.assertFalse(any("Previous" in (t.name or "") for t in fig.data))

    def test_time_window(self):
        fig = figures.errors_figure(self.current, None, t_window=(5.0, 10.0))
        trace = next(t for t in fig.data if t.name and "Current" in t.name)
        self.assertGreaterEqual(min(trace.x), 5.0)
        self.assertLessEqual(max(trace.x), 10.0)

    def test_indices_figure(self):
        fig = figures.indices_figure(self.current, self.previous)
        self.assertEqual(len(fig.data), len(figures.met.KEY_INDICES))
        self.assertEqual(list(fig.data[0].marker.color),
                         [figures.PREVIOUS_COLOR, figures.CURRENT_COLOR])


class XYDisturbanceAndWindowTests(unittest.TestCase):
    """Disturbance information and time window in the XY trajectory."""

    @classmethod
    def setUpClass(cls):
        history = RunHistory()
        cls.records = {}
        for kind, extra in (("none", {}), ("directional", {}),
                            ("constant", dict(dist_const_dTu=8.0,
                                              dist_const_dTv=5.0)),
                            ("combined", {})):
            cls.records[kind] = history.add(run(dist_kind=kind, **extra))

    def arrows(self, fig):
        return [t for t in fig.data
                if (t.name or "").startswith("Disturbance force on the hull")]

    def test_no_disturbance_no_disturbance_marks(self):
        fig = figures.xy_figure(self.records["none"])
        self.assertFalse(self.arrows(fig))
        self.assertFalse(fig.layout.annotations)

    def test_fixed_direction_force_is_one_arrow_with_its_orientation(self):
        fig = figures.xy_figure(self.records["directional"])
        self.assertFalse(self.arrows(fig))      # same vector everywhere
        arrow = next(a for a in fig.layout.annotations if a.showarrow)
        angle = np.degrees(np.arctan2(arrow.y - arrow.ay, arrow.x - arrow.ax))
        self.assertAlmostEqual(angle, 45.0, places=6)
        texts = " ".join(a.text for a in fig.layout.annotations)
        self.assertIn("F = 25 N", texts)
        self.assertIn("β = 45°", texts)
        # The arrow lies in the band above the trajectory.
        data = self.records["directional"].result.data
        self.assertGreater(min(arrow.y, arrow.ay), data["y"].max())

    def test_force_arrows_follow_the_simulated_force(self):
        record = self.records["constant"]
        fig = figures.xy_figure(record)
        (trace,) = self.arrows(fig)
        x = np.array(trace.x, dtype=float)
        y = np.array(trace.y, dtype=float)
        data = record.result.data
        fx, fy = figures._inertial_force(data)
        # The body-fixed force rotates with the vessel in the XY plane.
        heads = np.isfinite(x)
        starts = np.where(heads & np.roll(~heads, 1))[0]
        for i in starts[:5]:
            k = int(np.argmin(np.hypot(data["x"] - x[i], data["y"] - y[i])))
            direction = np.arctan2(y[i + 1] - y[i], x[i + 1] - x[i])
            self.assertAlmostEqual(np.cos(direction - np.arctan2(fy[k], fx[k])),
                                   1.0, places=6)
        self.assertIn("rotates with the vessel",
                      " ".join(a.text for a in fig.layout.annotations))
        self.assertIn("δT<sub>r</sub> = 15 N m ↺",
                      " ".join(a.text for a in fig.layout.annotations))

    def test_combined_shows_both(self):
        fig = figures.xy_figure(self.records["combined"])
        self.assertEqual(len(self.arrows(fig)), 1)
        self.assertEqual(sum(1 for a in fig.layout.annotations
                             if a.showarrow), 1)

    def test_time_window_in_the_main_plot(self):
        record = self.records["none"]
        fig = figures.xy_figure(record, t_window=(3.0, 9.0))
        names = [t.name for t in fig.data]
        self.assertIn("Current run outside the time window", names)
        texts = [a.text for a in fig.layout.annotations]
        self.assertIn("t = 3.0 s", texts)
        self.assertIn("t = 9.0 s", texts)
        current = next(t for t in fig.data if "Current · Run" in (t.name or ""))
        self.assertGreaterEqual(min(current.customdata), 3.0)
        self.assertLessEqual(max(current.customdata), 9.0)
        full = figures.xy_figure(record)
        self.assertNotIn("Current run outside the time window",
                         [t.name for t in full.data])


class EnvelopeTests(unittest.TestCase):
    def test_envelope_keeps_extremes_of_fast_oscillations(self):
        t = np.arange(100000) * 0.01
        y = np.where(np.arange(t.size) % 2, 1.0, -1.0) * (1 + t / 1000)
        te, ye = figures.envelope(t, y)
        self.assertLessEqual(len(te), 2 * figures.MAX_BUCKETS + 10)
        self.assertEqual(ye.max(), y.max())
        self.assertEqual(ye.min(), y.min())
        self.assertTrue(np.all(np.diff(te) > 0))
        self.assertEqual((te[0], te[-1]), (t[0], t[-1]))

    def test_short_signals_are_unchanged(self):
        t = np.arange(10.0)
        te, ye = figures.envelope(t, t**2)
        np.testing.assert_array_equal(ye, t**2)


if __name__ == "__main__":
    unittest.main()
