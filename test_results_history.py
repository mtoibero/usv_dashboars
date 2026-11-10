import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_results import plot_thruster_forces, plot_torque_saturation
from run_simulation import controller_output_history


class CountingController:
    def __init__(self):
        self.output_calls = 0

    def output(self, t, state):
        self.output_calls += 1
        return (
            np.array([t, 0.0, t]),
            np.array([t, 0.0, t]),
            np.array([t, t, t, t]),
        )

    def required_tau(self, t, state):
        raise AssertionError("Cached combined output should be used.")

    def delivered_tau(self, t, state):
        raise AssertionError("Cached combined output should be used.")

    def thruster_forces(self, t, state):
        raise AssertionError("Cached combined output should be used.")


class ResultsHistoryTests(unittest.TestCase):
    def setUp(self):
        self.t = np.array([0.0, 0.02, 0.04])
        self.state = np.zeros((3, 6))
        self.controller = CountingController()

    def tearDown(self):
        plt.close("all")

    def test_controller_history_uses_single_combined_call_per_sample(self):
        history = controller_output_history(self.controller, self.t, self.state)

        self.assertEqual(self.controller.output_calls, len(self.t))
        np.testing.assert_allclose(history[2][:, 0], self.t)

    def test_plots_accept_cached_history_without_reinvoking_controller(self):
        tau_required, tau_delivered, forces = controller_output_history(
            self.controller,
            self.t,
            self.state,
        )

        plot_torque_saturation(
            self.t,
            self.state,
            self.controller,
            tau_required=tau_required,
            tau_delivered=tau_delivered,
        )
        plot_thruster_forces(
            self.t,
            self.state,
            self.controller,
            forces=forces,
        )

        self.assertEqual(self.controller.output_calls, len(self.t))


if __name__ == "__main__":
    unittest.main()
