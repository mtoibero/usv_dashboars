import unittest

import numpy as np

from usv_model import (
    THRUSTER_FWD_MAX,
    THRUSTER_REV_MAX,
    thrust_to_normalized_cmd,
    thrust_to_pwm_us,
    thruster_forces_to_pwm_us,
)


class ThrusterPwmConversionTests(unittest.TestCase):
    def test_reference_thrust_values_map_to_pwm_endpoints(self):
        forces = np.array([-THRUSTER_REV_MAX, 0.0, THRUSTER_FWD_MAX])

        np.testing.assert_allclose(
            thrust_to_pwm_us(forces),
            [1100.0, 1500.0, 1900.0],
        )

    def test_reference_thrust_values_map_to_normalized_commands(self):
        forces = np.array([-THRUSTER_REV_MAX, 0.0, THRUSTER_FWD_MAX])

        np.testing.assert_allclose(
            thrust_to_normalized_cmd(forces),
            [-1.0, 0.0, 1.0],
        )

    def test_four_thruster_conversion_is_vectorized(self):
        forces = np.array([THRUSTER_FWD_MAX, 0.0, -THRUSTER_REV_MAX, 0.0])

        np.testing.assert_allclose(
            thruster_forces_to_pwm_us(forces),
            [1900.0, 1500.0, 1100.0, 1500.0],
        )


if __name__ == "__main__":
    unittest.main()
