import numpy as np
from scipy.integrate import solve_ivp


# ============================================================
# Parametros identificados del IAcquaBot
# Tabla 4.21
# ============================================================

PARAMS = {
    "z1": 124.5923,   # m - X_udot
    "z2": 120.0275,   # m - Y_vdot
    "z3": 141.5164,   # Iz - N_rdot

    "z4": 25.9335,    # Xu
    "z5": 35.9833,    # Xuu
    "z6": 37.0247,    # Yv
    "z7": 25.7968,    # Yvv
    "z8": 200.2944,   # Nr
    "z9": 18.5319,    # Nrr
}


# ============================================================
# Limites fisicos de los propulsores
# ============================================================

THRUSTER_FWD_MAX = 36.38   # N
THRUSTER_REV_MAX = 28.44   # N
N_THRUSTERS = 4
THRUSTER_DISTANCE = 0.670  # m
THRUSTER_ARM = THRUSTER_DISTANCE / 2.0
PWM_MIN_US = 1100.0
PWM_NEUTRAL_US = 1500.0
PWM_MAX_US = 1900.0
PWM_HALF_RANGE_US = 400.0

TU_MAX = N_THRUSTERS * THRUSTER_FWD_MAX
TU_MIN = -N_THRUSTERS * THRUSTER_REV_MAX

TR_MAX = THRUSTER_ARM * (
    2.0 * THRUSTER_FWD_MAX + 2.0 * THRUSTER_REV_MAX
)
TR_MIN = -TR_MAX

TU_LIMITS = (TU_MIN, TU_MAX)
TR_LIMITS = (TR_MIN, TR_MAX)


def thrust_to_normalized_cmd(force):
    """
    Convierte el empuje de un propulsor [N] a un comando normalizado.

    La normalizacion usa limites diferentes para avance y reversa.
    """
    force = np.asarray(force, dtype=float)
    force = np.clip(force, -THRUSTER_REV_MAX, THRUSTER_FWD_MAX)

    cmd = np.where(
        force >= 0.0,
        force / THRUSTER_FWD_MAX,
        force / THRUSTER_REV_MAX,
    )

    return np.clip(cmd, -1.0, 1.0)


def normalized_cmd_to_pwm_us(cmd):
    """Convierte un comando normalizado a ancho de pulso PWM [us]."""
    cmd = np.asarray(cmd, dtype=float)
    cmd = np.clip(cmd, -1.0, 1.0)

    return PWM_NEUTRAL_US + PWM_HALF_RANGE_US * cmd


def thrust_to_pwm_us(force):
    """Convierte el empuje de un propulsor [N] a ancho de pulso PWM [us]."""
    return normalized_cmd_to_pwm_us(thrust_to_normalized_cmd(force))


def thruster_forces_to_pwm_us(forces):
    """Convierte fuerzas [F1, F2, F3, F4] [N] a PWM por propulsor [us]."""
    forces = np.asarray(forces, dtype=float)
    return thrust_to_pwm_us(forces)


def allocate_thrusters(Tu_cmd, Tr_cmd):
    """
    Asignacion simple para 4 propulsores diferenciales.

    Propulsores:
    F1, F2: lado izquierdo
    F3, F4: lado derecho

    Devuelve:
    tau_real = [Tu_real, 0, Tr_real]
    forces   = [F1, F2, F3, F4]
    """
    b = THRUSTER_ARM

    F_left_total = 0.5 * Tu_cmd - 0.5 * Tr_cmd / b
    F_right_total = 0.5 * Tu_cmd + 0.5 * Tr_cmd / b

    forces_cmd = np.array([
        F_left_total / 2.0,
        F_left_total / 2.0,
        F_right_total / 2.0,
        F_right_total / 2.0,
    ])

    forces_real = np.clip(
        forces_cmd,
        -THRUSTER_REV_MAX,
        THRUSTER_FWD_MAX,
    )

    F_left_real = forces_real[0] + forces_real[1]
    F_right_real = forces_real[2] + forces_real[3]

    Tu_real = F_left_real + F_right_real
    Tr_real = b * (F_right_real - F_left_real)
    tau_real = np.array([Tu_real, 0.0, Tr_real])

    return tau_real, forces_real


# ============================================================
# Cinematica
# eta = [x, y, psi]
# nu  = [u, v, r]
# ============================================================

def j_matrix(psi):
    """
    Matriz cinematica que transforma velocidades del marco cuerpo
    al marco inercial.

    eta_dot = J(psi) @ nu
    """
    c = np.cos(psi)
    s = np.sin(psi)

    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def kinematics(eta, nu):
    """
    eta = [x, y, psi]
    nu  = [u, v, r]
    """
    return j_matrix(eta[2]) @ nu


# ============================================================
# Dinamica identificada
#
# z1*(u_dot - v*r) + z4*u + z5*|u|u = Tu
# z2*(v_dot + u*r) + z6*v + z7*|v|v = Tv
# z3*r_dot         + z8*r + z9*|r|r = Tr
# ============================================================

def dynamics_body(nu, tau, params=PARAMS):
    """
    Calcula nu_dot = [u_dot, v_dot, r_dot].

    nu  = [u, v, r]
    tau = [Tu, Tv, Tr]
    """
    u, v, r = nu
    Tu, Tv, Tr = tau

    z1 = params["z1"]
    z2 = params["z2"]
    z3 = params["z3"]
    z4 = params["z4"]
    z5 = params["z5"]
    z6 = params["z6"]
    z7 = params["z7"]
    z8 = params["z8"]
    z9 = params["z9"]

    u_dot = (Tu - z4 * u - z5 * abs(u) * u) / z1 + v * r
    v_dot = (Tv - z6 * v - z7 * abs(v) * v) / z2 - u * r
    r_dot = (Tr - z8 * r - z9 * abs(r) * r) / z3

    return np.array([u_dot, v_dot, r_dot])


# ============================================================
# Modelo completo en espacio de estados
# state = [x, y, psi, u, v, r]
# ============================================================

def vessel_rhs(
    t,
    state,
    tau_fun,
    disturbance_fun=None,
    params=PARAMS,
    reference_governor_fun=None,
):
    """
    Ecuacion diferencial completa.

    tau_fun(t, state) debe devolver [Tu, Tv, Tr].
    disturbance_fun(t, state) debe devolver una perturbacion aditiva
    [dTu, dTv, dTr]. Si es None, se usa perturbacion nula.
    """
    vessel_state = state[0:6]
    eta = vessel_state[0:3]
    nu = vessel_state[3:6]

    gamma = None
    gamma_dot = None
    state_for_control = state
    if state.size >= 7:
        if reference_governor_fun is None:
            governor_output = {
                "gamma_target": 1.0,
                "tau_gamma": 1.0,
            }
        else:
            governor_output = reference_governor_fun(t, state)
            if not isinstance(governor_output, dict):
                governor_output = {
                    "gamma_target": float(governor_output),
                    "tau_gamma": 1.0,
                }

        gamma_target = float(governor_output["gamma_target"])
        tau_gamma = max(float(governor_output.get("tau_gamma", 1.0)), 1e-6)
        gamma = state[7] if state.size >= 8 else gamma_target
        gamma_dot = (gamma_target - gamma) / tau_gamma
        state_for_control = np.concatenate([state, [gamma]])

    tau = np.asarray(tau_fun(t, state_for_control), dtype=float)

    if disturbance_fun is not None:
        tau += np.asarray(disturbance_fun(t, state), dtype=float)

    eta_dot = kinematics(eta, nu)
    nu_dot = dynamics_body(nu, tau, params)

    state_dot = np.concatenate([eta_dot, nu_dot])
    if state.size >= 7:
        state_dot = np.concatenate([state_dot, [gamma]])
    if state.size >= 8:
        state_dot = np.concatenate([state_dot, [gamma_dot]])

    return state_dot


def simulate(
    state0,
    tau_fun,
    t_final,
    dt=0.02,
    disturbance_fun=None,
    params=PARAMS,
    reference_governor_fun=None,
    rtol=1e-8,
    atol=1e-10,
):
    """
    Simula la postura y velocidades del bote.

    Devuelve:
    t     : vector de tiempo
    state : matriz con columnas [x, y, psi, u, v, r] y opcionalmente sigma
    """
    t_eval = np.linspace(0.0, t_final, int(t_final / dt) + 1)

    sol = solve_ivp(
        fun=lambda t, x: vessel_rhs(
            t,
            x,
            tau_fun,
            disturbance_fun,
            params,
            reference_governor_fun,
        ),
        t_span=(0.0, t_final),
        y0=state0,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.t, sol.y.T
