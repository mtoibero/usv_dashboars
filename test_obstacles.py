import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from obstacle_config import OBSTACLES, configured_obstacles
from plot_results import draw_obstacles


class ObstacleConfigurationTests(unittest.TestCase):
    def test_enabled_obstacles_use_valid_default_geometry(self):
        obstacles = configured_obstacles()

        self.assertEqual(obstacles, OBSTACLES)
        self.assertTrue(obstacles)
        self.assertTrue(all(obstacle.radius > 0.0 for obstacle in obstacles))

    def test_obstacles_can_be_disabled(self):
        self.assertEqual(configured_obstacles(enabled=False), ())

    def test_obstacles_draw_one_circular_boundary_per_buoy(self):
        fig, ax = plt.subplots()

        draw_obstacles(ax, configured_obstacles())

        self.assertEqual(len(ax.patches), 2)
        self.assertTrue(
            all(
                patch.get_edgecolor()[0:3] == (1.0, 0.0, 1.0)
                for patch in ax.patches
            )
        )
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()