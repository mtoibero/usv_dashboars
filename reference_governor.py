import numpy as np

from cbf_filters import CBF_MARGIN, obstacle_safe_radius
from controllers import TRACKING_LOOKAHEAD, lookahead_point_from_state


GOVERNOR_ENABLED = True         # baja la velocidad de la referencia cdo hay un obstáculo
GOVERNOR_GAMMA_MIN = 0.50
GOVERNOR_H_SLOW = 20.0
GOVERNOR_H_STOP = 2.0
GOVERNOR_FILTER_TAU_MIN = 1.0
GOVERNOR_FILTER_TAU_MAX = 8.0


def barrier_margin_min(state, obstacles, lookahead_L=TRACKING_LOOKAHEAD, margin=CBF_MARGIN):
    """Devuelve el minimo h = ||p_s - p_o||^2 - R^2 entre obstaculos."""
    obstacles = () if obstacles is None else tuple(obstacles)
    if not obstacles:
        return np.inf

    p_s, _ = lookahead_point_from_state(state, lookahead_L)
    h_values = []

    for obstacle in obstacles:
        center = np.array([obstacle.x, obstacle.y], dtype=float)
        safe_radius = obstacle_safe_radius(obstacle, margin)
        relative_position = np.asarray(p_s, dtype=float) - center
        h_values.append(np.dot(relative_position, relative_position) - safe_radius**2)

    return float(np.min(h_values))


def gamma_from_barrier_margin(
    h_min,
    gamma_min=GOVERNOR_GAMMA_MIN,
    h_slow=GOVERNOR_H_SLOW,
    h_stop=GOVERNOR_H_STOP,
):
    """
    Convierte margen de barrera en velocidad virtual.

    h >= h_slow -> gamma = 1
    h <= h_stop -> gamma = gamma_min
    """
    if not np.isfinite(h_min):
        return 1.0

    if h_slow <= h_stop:
        return 1.0

    blend = (h_min - h_stop) / (h_slow - h_stop)
    blend = np.clip(blend, 0.0, 1.0)

    return float(gamma_min + (1.0 - gamma_min) * blend)


def governor_intensity_from_margin(h_min, h_slow=GOVERNOR_H_SLOW, h_stop=GOVERNOR_H_STOP):
    """Devuelve 0 lejos del obstaculo y 1 dentro de la zona mas critica."""
    if not np.isfinite(h_min) or h_slow <= h_stop:
        return 0.0

    return float(np.clip((h_slow - h_min) / (h_slow - h_stop), 0.0, 1.0))


def gamma_filter_time_constant(
    h_min,
    gamma_current,
    gamma_target,
    gamma_min=GOVERNOR_GAMMA_MIN,
    h_slow=GOVERNOR_H_SLOW,
    h_stop=GOVERNOR_H_STOP,
    tau_min=GOVERNOR_FILTER_TAU_MIN,
    tau_max=GOVERNOR_FILTER_TAU_MAX,
):
    """
    Constante de tiempo del pasabajos de gamma.

    A mayor urgencia CBF o mayor salto pedido de gamma, mas grande es tau y
    mas suave resulta el cambio de velocidad de referencia.
    """
    if tau_max <= tau_min:
        return float(max(tau_min, 1e-6))

    intensity = governor_intensity_from_margin(
        h_min,
        h_slow=h_slow,
        h_stop=h_stop,
    )
    change = abs(gamma_target - gamma_current) / max(1.0 - gamma_min, 1e-6)
    blend = np.clip(max(intensity, change), 0.0, 1.0)

    return float(tau_min + (tau_max - tau_min) * blend)


def virtual_time_governor(
    obstacles,
    enabled=GOVERNOR_ENABLED,
    lookahead_L=TRACKING_LOOKAHEAD,
    margin=CBF_MARGIN,
    gamma_min=GOVERNOR_GAMMA_MIN,
    h_slow=GOVERNOR_H_SLOW,
    h_stop=GOVERNOR_H_STOP,
    tau_min=GOVERNOR_FILTER_TAU_MIN,
    tau_max=GOVERNOR_FILTER_TAU_MAX,
):
    """Crea el comando del gobernador para sigma_dot = gamma filtrada."""
    obstacles = () if obstacles is None else tuple(obstacles)

    def governor(t, state):
        if not enabled:
            return {
                "gamma_target": 1.0,
                "tau_gamma": max(tau_min, 1e-6),
                "h_min": np.inf,
            }

        h_min = barrier_margin_min(
            state,
            obstacles,
            lookahead_L=lookahead_L,
            margin=margin,
        )
        gamma_target = gamma_from_barrier_margin(
            h_min,
            gamma_min=gamma_min,
            h_slow=h_slow,
            h_stop=h_stop,
        )
        gamma_current = state[7] if len(state) >= 8 else gamma_target
        tau_gamma = gamma_filter_time_constant(
            h_min,
            gamma_current,
            gamma_target,
            gamma_min=gamma_min,
            h_slow=h_slow,
            h_stop=h_stop,
            tau_min=tau_min,
            tau_max=tau_max,
        )

        return {
            "gamma_target": gamma_target,
            "tau_gamma": tau_gamma,
            "h_min": h_min,
        }

    return governor


def governor_history(
    state,
    obstacles,
    enabled=GOVERNOR_ENABLED,
    lookahead_L=TRACKING_LOOKAHEAD,
    margin=CBF_MARGIN,
    gamma_min=GOVERNOR_GAMMA_MIN,
    h_slow=GOVERNOR_H_SLOW,
    h_stop=GOVERNOR_H_STOP,
    tau_min=GOVERNOR_FILTER_TAU_MIN,
    tau_max=GOVERNOR_FILTER_TAU_MAX,
):
    """Calcula h_min y gamma para una historia de estados."""
    governor = virtual_time_governor(
        obstacles,
        enabled=enabled,
        lookahead_L=lookahead_L,
        margin=margin,
        gamma_min=gamma_min,
        h_slow=h_slow,
        h_stop=h_stop,
        tau_min=tau_min,
        tau_max=tau_max,
    )
    h_min = np.array([
        barrier_margin_min(
            state_i,
            obstacles,
            lookahead_L=lookahead_L,
            margin=margin,
        )
        for state_i in state
    ])
    governor_output = [governor(0.0, state_i) for state_i in state]
    gamma_target = np.array([output["gamma_target"] for output in governor_output])
    tau_gamma = np.array([output["tau_gamma"] for output in governor_output])
    gamma = state[:, 7] if state.shape[1] >= 8 else gamma_target

    return {
        "h_min": h_min,
        "gamma": gamma,
        "gamma_target": gamma_target,
        "tau_gamma": tau_gamma,
        "sigma": state[:, 6] if state.shape[1] >= 7 else None,
    }
