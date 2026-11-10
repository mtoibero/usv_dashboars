import numpy as np


def no_disturbance(t, state):
    """Perturbacion nula."""
    return np.zeros(3)


def combined_disturbance(*disturbance_funs):
    """Combina varias perturbaciones aditivas d(t) = [dTu, dTv, dTr]."""
    def disturbance_fun(t, state):
        total = np.zeros(3)

        for single_disturbance_fun in disturbance_funs:
            total += np.asarray(single_disturbance_fun(t, state), dtype=float)

        return total

    return disturbance_fun


def constant_disturbance(dTu=0.0, dTv=0.0, dTr=0.0):
    """
    Devuelve una perturbacion constante en fuerzas/torque.

    dTu: perturbacion longitudinal
    dTv: perturbacion lateral
    dTr: perturbacion de guinada
    """
    disturbance = np.array([dTu, dTv, dTr], dtype=float)

    def disturbance_fun(t, state):
        return disturbance

    return disturbance_fun


def sinusoidal_disturbance(
    amplitude=(0.0, 0.0, 0.0),
    frequency=(0.1, 0.1, 0.1),
    phase=(0.0, 0.0, 0.0),
    bias=(0.0, 0.0, 0.0),
):
    """
    Devuelve una perturbacion senoidal aditiva.

    amplitude, frequency, phase y bias son vectores para [Tu, Tv, Tr].
    frequency esta en Hz.
    """
    amplitude = np.asarray(amplitude, dtype=float)
    frequency = np.asarray(frequency, dtype=float)
    phase = np.asarray(phase, dtype=float)
    bias = np.asarray(bias, dtype=float)

    def disturbance_fun(t, state):
        return bias + amplitude * np.sin(2.0 * np.pi * frequency * t + phase)

    return disturbance_fun


def pure_sinusoidal_disturbance(t, state):
    """
    Perturbacion senoidal pura sin bias.

    Todas las componentes oscilan alrededor de cero.
    """
    dTu = 3.0 * np.sin(2.0 * np.pi * 0.035 * t)
    dTv = 4.0 * np.sin(2.0 * np.pi * 0.050 * t + 0.8)
    dTr = 2.0 * np.sin(2.0 * np.pi * 0.025 * t + 1.4)

    return np.array([dTu, dTv, dTr])


def directional_force_disturbance(magnitude=5.0, direction_deg=45.0):
    """
    Perturbacion como fuerza fija en el marco inercial.

    La fuerza se aplica con modulo `magnitude` y direccion `direction_deg`.
    Luego se proyecta al marco cuerpo para devolver [dTu, dTv, dTr].
    """
    direction_rad = np.deg2rad(direction_deg)
    force_inertial = np.array([
        magnitude * np.cos(direction_rad),
        magnitude * np.sin(direction_rad),
    ])

    def disturbance_fun(t, state):
        psi = state[2]
        c = np.cos(psi)
        s = np.sin(psi)

        dTu = c * force_inertial[0] + s * force_inertial[1]
        dTv = -s * force_inertial[0] + c * force_inertial[1]
        dTr = 0.0

        return np.array([dTu, dTv, dTr])

    return disturbance_fun


def wind_like_disturbance(t, state):
    """
    Perturbacion simple tipo viento/corriente.

    Combina un empuje lateral medio con variaciones lentas en avance y guinada.
    """
    dTu = 2.0 * np.sin(2.0 * np.pi * 0.035 * t)
    dTv = 3.0 + 4.0 * np.sin(2.0 * np.pi * 0.050 * t + 0.8)
    dTr = 2.5 * np.sin(2.0 * np.pi * 0.025 * t + 1.4)

    return np.array([dTu, dTv, dTr])
