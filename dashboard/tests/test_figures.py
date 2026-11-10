"""XY figure: overlay of the previous run."""

import unittest
from types import SimpleNamespace

import numpy as np

try:
    from dashboard import figures
except ImportError:  # pragma: no cover - plotly not installed
    figures = None

from dashboard import config as cfg
from dashboard import engine


def run(**changes):
    form = dict(cfg.default_form(), sim_duration_mode="manual",
                sim_t_final_s=10.0, sim_dt_s=0.05)
    form.update(changes)
    return engine.run_from_config(cfg.form_to_config(form))


@unittest.skipIf(figures is None, "plotly is not installed")
class XYOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = run(smc_k_pos=0.6)
        cls.previous = SimpleNamespace(result=run(), title="Run #1 - baseline")

    def names(self, fig):
        return [trace.name for trace in fig.data]

    def test_without_previous_run(self):
        fig = figures.xy_trajectory(self.current)
        self.assertFalse(any(n.startswith("Previous") for n in self.names(fig)))

    def test_previous_run_is_drawn_in_gray_below_reference_and_current(self):
        fig = figures.xy_trajectory(self.current, previous=self.previous)
        names = self.names(fig)
        self.assertEqual(names[0], "Previous: Run #1 - baseline")
        self.assertEqual(fig.data[0].line.color, figures.PREVIOUS_COLOR)
        # Reference and current run are drawn on top, in their own colours.
        reference = names.index("Reference path")
        current = names.index("USV hull centre <b>p</b>")
        self.assertLess(0, reference)
        self.assertLess(reference, current)
        self.assertEqual(fig.data[reference].line.color, figures.REF_COLOR)
        self.assertEqual(fig.data[reference].line.dash, "dash")
        self.assertEqual(fig.data[current].line.color, figures.RUN_COLORS[0])
        # Only one previous trajectory.
        self.assertEqual(sum(n.startswith("Previous:") for n in names), 1)

    def test_previous_trajectory_data_and_shared_axes(self):
        fig = figures.xy_trajectory(self.current, previous=self.previous)
        data = self.previous.result.data
        np.testing.assert_allclose(fig.data[0].x[[0, -1]],
                                   data["x"].to_numpy()[[0, -1]])
        np.testing.assert_allclose(fig.data[0].y[[0, -1]],
                                   data["y"].to_numpy()[[0, -1]])
        self.assertTrue(all(trace.xaxis in (None, "x") and
                            trace.yaxis in (None, "y") for trace in fig.data))
        self.assertEqual(fig.layout.yaxis.scaleanchor, "x")
        self.assertEqual(fig.layout.yaxis.scaleratio, 1)

    def test_previous_reference_shown_only_if_different(self):
        same = figures.xy_trajectory(self.current, previous=self.previous)
        self.assertNotIn("Previous reference path", self.names(same))
        other = SimpleNamespace(result=run(ref_amplitude_m=12.0), title="Run #0")
        different = figures.xy_trajectory(self.current, previous=other)
        self.assertIn("Previous reference path", self.names(different))


if __name__ == "__main__":
    unittest.main()
