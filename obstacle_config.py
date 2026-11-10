from dataclasses import dataclass


@dataclass(frozen=True)
class CircularObstacle:
    """Circular obstacle represented in the inertial x-y plane."""

    x: float
    y: float
    radius: float
    name: str = ""


OBSTACLES = (
    CircularObstacle(x=10.0, y=-6.5, radius=0.75, name="buoy_1"),
    CircularObstacle(x=-13.0, y=-7.5, radius=0.25, name="buoy_2"),
)


def configured_obstacles(enabled=True, obstacles=OBSTACLES):
    """Returns the configured obstacle set, or no obstacles if disabled."""
    if enabled:
        return tuple(obstacles)

    return ()
