import numpy as np

from usv_model import PARAMS, TR_LIMITS, TU_LIMITS, allocate_thrusters


FIGURE_EIGHT_AMPLITUDE  = 15.0
FIGURE_EIGHT_PERIOD     = 140.0      #Lower to increase Reference Velocity 
FIGURE_EIGHT_LAPS       = 2.0
TRACKING_LOOKAHEAD      = 1.0        # 1 metro hacia adelante
REFERENCE_EPS           = 1e-6       # es para no dividir por cero


def wrap_angle(angle):
    """Limita un angulo al intervalo [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def reference_time_from_state(t, state):
    """Devuelve sigma si el estado aumentado lo incluye; si no, usa t."""
    state = np.asarray(state, dtype=float)
    if state.shape[-1] >= 7:
        return state[..., 6]

    return t


def reference_rate_from_state(state):
    """Devuelve gamma si fue agregado para la evaluacion del controlador."""
    state = np.asarray(state, dtype=float)
    if state.shape[-1] >= 8:
        return state[..., 7]

    return 1.0


def manual_tau(Tu=0.0, Tv=0.0, Tr=0.0):
    """Devuelve un controlador de fuerzas/torques constantes."""
    def controller(t, state):
        return np.array([Tu, Tv, Tr])

    return controller


def tau_example(t, state):
    """Entrada de prueba por tramos."""
    if t < 10.0:
        return np.array([80.0, 0.0, 0.0])
    if t < 25.0:
        return np.array([60.0, 0.0, 35.0])
    return np.array([0.0, 0.0, 0.0])


def figure_eight_reference(
    t,
    amplitude=FIGURE_EIGHT_AMPLITUDE,
    period=FIGURE_EIGHT_PERIOD,
):
    """
    Referencia tipo 8 cerrada.

    x = A sin(wt)
    y = A sin(wt) cos(wt)
    """
    omega = 2.0 * np.pi / period
    phase = omega * t

    x_ref = amplitude * np.sin(phase)
    y_ref = amplitude * np.sin(phase) * np.cos(phase)

    x_dot_ref = amplitude * omega * np.cos(phase)
    y_dot_ref = amplitude * omega * np.cos(2.0 * phase)

    x_ddot_ref = -amplitude * omega**2 * np.sin(phase)
    y_ddot_ref = -2.0 * amplitude * omega**2 * np.sin(2.0 * phase)

    return x_ref, y_ref, x_dot_ref, y_dot_ref, x_ddot_ref, y_ddot_ref


def lookahead_point_from_state(state, lookahead_L=TRACKING_LOOKAHEAD):
    """Calcula el punto adelantado ps y su velocidad inercial ps_dot."""
    state = np.asarray(state, dtype=float)
    x = state[..., 0]
    y = state[..., 1]
    psi = state[..., 2]
    u = state[..., 3]
    v = state[..., 4]
    r = state[..., 5]

    c = np.cos(psi)
    s = np.sin(psi)

    x_s = x + lookahead_L * c
    y_s = y + lookahead_L * s
    x_s_dot = c * u - s * v - lookahead_L * r * s
    y_s_dot = s * u + c * v + lookahead_L * r * c

    return (
        np.stack([x_s, y_s], axis=-1),
        np.stack([x_s_dot, y_s_dot], axis=-1),
    )


def figure_eight_lookahead_reference(
    t,
    lookahead_L=TRACKING_LOOKAHEAD,
    time_scale=1.0,
    eps=REFERENCE_EPS,
    amplitude=FIGURE_EIGHT_AMPLITUDE,
    period=FIGURE_EIGHT_PERIOD,
):
    """Referencia del punto adelantado ps_ref y su velocidad inercial."""
    x_ref, y_ref, x_dot_ref, y_dot_ref, x_ddot_ref, y_ddot_ref = (
        figure_eight_reference(t, amplitude, period)
    )

    time_scale = np.asarray(time_scale, dtype=float)
    x_dot_ref = time_scale * x_dot_ref
    y_dot_ref = time_scale * y_dot_ref
    x_ddot_ref = time_scale**2 * x_ddot_ref
    y_ddot_ref = time_scale**2 * y_ddot_ref

    psi_ref = np.arctan2(y_dot_ref, x_dot_ref)
    c_ref = np.cos(psi_ref)
    s_ref = np.sin(psi_ref)

    speed_squared = x_dot_ref**2 + y_dot_ref**2 + eps
    psi_dot_ref = (
        x_dot_ref * y_ddot_ref
        - y_dot_ref * x_ddot_ref
    ) / speed_squared

    x_s_ref = x_ref + lookahead_L * c_ref
    y_s_ref = y_ref + lookahead_L * s_ref
    x_s_dot_ref = x_dot_ref - lookahead_L * psi_dot_ref * s_ref
    y_s_dot_ref = y_dot_ref + lookahead_L * psi_dot_ref * c_ref

    return (
        np.stack([x_s_ref, y_s_ref], axis=-1),
        np.stack([x_s_dot_ref, y_s_dot_ref], axis=-1),
        psi_ref,
        psi_dot_ref,
    )


def figure_eight_initial_state(
    amplitude=FIGURE_EIGHT_AMPLITUDE,
    period=FIGURE_EIGHT_PERIOD,
):
    """Estado inicial compatible con el inicio de la referencia tipo 8."""
    x_ref, y_ref, x_dot_ref, y_dot_ref, _, _ = figure_eight_reference(
        0.0, amplitude, period
    )
    psi_ref = np.arctan2(y_dot_ref, x_dot_ref)
    u_ref = np.hypot(x_dot_ref, y_dot_ref)

    return np.array([
        x_ref, y_ref, psi_ref,
        u_ref, 0.0, 0.0,
    ])


def _feedback_figure_eight_output(
    t,
    state,
    k_pos=0.35,
    k_u=90.0,
    k_psi=3.0,
    k_r=120.0,
    amplitude=FIGURE_EIGHT_AMPLITUDE,
    period=FIGURE_EIGHT_PERIOD,
):
    """Calcula (tau requerido, tau real, fuerzas) del controlador simple."""
    vessel_state = np.asarray(state, dtype=float)[0:6]
    x, y, psi, u, _, r = vessel_state
    reference_time = reference_time_from_state(t, state)
    x_ref, y_ref, x_dot_ref, y_dot_ref, x_ddot_ref, y_ddot_ref = (
        figure_eight_reference(reference_time, amplitude, period)
    )

    position_error = np.array([x_ref - x, y_ref - y])
    velocity_ref = np.array([x_dot_ref, y_dot_ref])
    acceleration_ref = np.array([x_ddot_ref, y_ddot_ref])

    velocity_cmd = velocity_ref + k_pos * position_error
    speed_cmd = np.linalg.norm(velocity_cmd)
    psi_cmd = np.arctan2(velocity_cmd[1], velocity_cmd[0])

    speed_ref = max(np.linalg.norm(velocity_ref), 1e-6)
    psi_dot_ref = (
        velocity_ref[0] * acceleration_ref[1]
        - velocity_ref[1] * acceleration_ref[0]
    ) / speed_ref**2

    psi_error = wrap_angle(psi_cmd - psi)

    r_cmd = psi_dot_ref + k_psi * psi_error

    Tu = PARAMS["z4"] * u + PARAMS["z5"] * abs(u) * u + k_u * (speed_cmd - u)
    Tr = PARAMS["z8"] * r + PARAMS["z9"] * abs(r) * r + k_r * (r_cmd - r)

    tau_required = np.array([Tu, 0.0, Tr])
    tau_real, thruster_forces = allocate_thrusters(Tu, Tr)

    return tau_required, tau_real, thruster_forces


def tau_figure_eight(t, state):
    """
    Controlador simple para que el bote siga una trayectoria cerrada tipo 8.

    Usa fuerza longitudinal Tu y torque de guinada Tr. Tv se mantiene en cero.
    Ganancias: k_pos = 0.35, k_u = 90, k_psi = 3, k_r = 120.
    """
    return _feedback_figure_eight_output(t, state)[1]


def feedback_figure_eight(
    k_pos=0.35,
    k_u=90.0,
    k_psi=3.0,
    k_r=120.0,
    amplitude=FIGURE_EIGHT_AMPLITUDE,
    period=FIGURE_EIGHT_PERIOD,
):
    """
    Version parametrizable de tau_figure_eight con la misma interfaz que
    smc_figure_eight (output, required_tau, delivered_tau, thruster_forces).

    Con los valores por defecto reproduce exactamente tau_figure_eight.
    """

    def compute_tau(t, state):
        return _feedback_figure_eight_output(
            t, state, k_pos, k_u, k_psi, k_r, amplitude, period
        )

    def controller(t, state):
        return compute_tau(t, state)[1]

    def required_tau(t, state):
        return compute_tau(t, state)[0]

    def delivered_tau(t, state):
        return compute_tau(t, state)[1]

    def thruster_forces(t, state):
        return compute_tau(t, state)[2]

    controller.output = compute_tau
    controller.required_tau = required_tau
    controller.delivered_tau = delivered_tau
    controller.thruster_forces = thruster_forces

    return controller

def sat(x):
    """
    Saturacion suave por componentes en [-1, 1].
    """
    return np.clip(x, -1.0, 1.0)


def smc_figure_eight(
    k_pos=0.35,
    k_psi=3.0,
    lambda_u=1.2,
    lambda_r=2.0,
    k_s_u=0.35,
    k_s_r=0.80,
    phi_u=0.08,
    phi_r=0.08,
    Tu_limits=TU_LIMITS,
    Tr_limits=TR_LIMITS,
    lookahead_L=TRACKING_LOOKAHEAD,
    amplitude=FIGURE_EIGHT_AMPLITUDE,
    period=FIGURE_EIGHT_PERIOD,
):
    """
    Controlador de modo deslizante para seguimiento de la referencia tipo 8.

    La estrategia es:
    1) convertir error del punto adelantado en una velocidad deseada;
    2) obtener u_cmd, psi_cmd y r_cmd;
    3) aplicar SMC sobre u y r.

    s_u = u - u_cmd
    s_r = r - r_cmd

    La entrada lateral Tv se deja en cero porque el bote se asume con propulsion diferencial.
    """

    def compute_tau(t, state):
        vessel_state = np.asarray(state, dtype=float)[0:6]
        x, y, psi, u, v, r = vessel_state
        reference_time = reference_time_from_state(t, state)
        reference_rate = reference_rate_from_state(state)

        x_ref, y_ref, x_dot_ref, y_dot_ref, x_ddot_ref, y_ddot_ref = (
            figure_eight_reference(reference_time, amplitude, period)
        )
        x_dot_ref = reference_rate * x_dot_ref
        y_dot_ref = reference_rate * y_dot_ref
        x_ddot_ref = reference_rate**2 * x_ddot_ref
        y_ddot_ref = reference_rate**2 * y_ddot_ref

        a_ref = np.array([x_ddot_ref, y_ddot_ref])
        p_s, p_s_dot = lookahead_point_from_state(vessel_state, lookahead_L)
        p_s_ref, p_s_dot_ref, _, _ = figure_eight_lookahead_reference(
            reference_time,
            lookahead_L,
            time_scale=reference_rate,
            amplitude=amplitude,
            period=period,
        )

        # ------------------------------------------------------------
        # Campo de velocidades deseado para el punto adelantado.
        # ------------------------------------------------------------
        e_p = p_s_ref - p_s
        e_v = p_s_dot_ref - p_s_dot

        v_cmd = p_s_dot_ref + k_pos * e_p
        v_cmd_dot = a_ref + k_pos * e_v

        speed_cmd = np.linalg.norm(v_cmd)
        speed_cmd = max(speed_cmd, 1e-6)

        speed_dot_cmd = np.dot(v_cmd, v_cmd_dot) / speed_cmd

        psi_cmd = np.arctan2(v_cmd[1], v_cmd[0])

        psi_dot_cmd = (
            v_cmd[0] * v_cmd_dot[1]
            - v_cmd[1] * v_cmd_dot[0]
        ) / speed_cmd**2

        e_psi = wrap_angle(psi_cmd - psi)

        r_cmd = psi_dot_cmd + k_psi * e_psi

        # ------------------------------------------------------------
        # Superficies de deslizamiento
        # ------------------------------------------------------------
        s_u = u - speed_cmd
        s_r = r - r_cmd

        # ------------------------------------------------------------
        # Ley SMC en aceleracion deseada
        #
        # u_dot_des = speed_dot_cmd - lambda_u*s_u - k_s_u*sat(s_u/phi_u)
        # r_dot_des =              - lambda_r*s_r - k_s_r*sat(s_r/phi_r)
        #
        # Luego se usa la dinamica identificada para calcular Tu y Tr.
        # ------------------------------------------------------------
        u_dot_des = (
            speed_dot_cmd
            - lambda_u * s_u
            - k_s_u * sat(s_u / phi_u)
        )

        r_dot_des = (
            - lambda_r * s_r
            - k_s_r * sat(s_r / phi_r)
        )

        # ------------------------------------------------------------
        # Inversion aproximada de la dinamica identificada
        #
        # u_dot = (Tu - z4*u - z5*|u|u)/z1 + v*r
        # r_dot = (Tr - z8*r - z9*|r|r)/z3
        #
        # Entonces:
        # Tu = z4*u + z5*|u|u + z1*(u_dot_des - v*r)
        # Tr = z8*r + z9*|r|r + z3*r_dot_des
        # ------------------------------------------------------------
        Tu = (
            PARAMS["z4"] * u
            + PARAMS["z5"] * abs(u) * u
            + PARAMS["z1"] * (u_dot_des - v * r)
        )

        Tr = (
            PARAMS["z8"] * r
            + PARAMS["z9"] * abs(r) * r
            + PARAMS["z3"] * r_dot_des
        )

        tau_required = np.array([Tu, 0.0, Tr])
        tau_delivered, thruster_forces = allocate_thrusters(Tu, Tr)

        return tau_required, tau_delivered, thruster_forces

    def controller(t, state):
        return compute_tau(t, state)[1]

    def required_tau(t, state):
        return compute_tau(t, state)[0]

    def delivered_tau(t, state):
        return compute_tau(t, state)[1]

    def thruster_forces(t, state):
        return compute_tau(t, state)[2]

    controller.output = compute_tau
    controller.required_tau = required_tau
    controller.delivered_tau = delivered_tau
    controller.thruster_forces = thruster_forces
    controller.lookahead_L = lookahead_L

    return controller
