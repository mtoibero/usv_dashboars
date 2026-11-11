"""Adapter between a ``SimulationConfig`` and the existing simulator.

No dynamics or control law is implemented here.  The engine

1. builds the existing objects: ``controllers.smc_figure_eight`` (surge and
   yaw SMC, thruster allocation) and ``disturbances.sinusoidal_disturbance``
   / ``disturbances.no_disturbance``;
2. integrates the model of ``usv_model`` in one of two ways:

   * ``continuous``: ``usv_model.simulate`` exactly as ``run_simulation.main``
     (the controller is evaluated continuously by the adaptive integrator);
   * ``digital``: the controller is sampled every ``Ts = 1/f_s`` and its
     delivered forces are held constant (zero-order hold) while
     ``usv_model.vessel_rhs`` is integrated over each sample interval with the
     same integrator and tolerances as ``usv_model.simulate`` (RK45,
     rtol 1e-8, atol 1e-10).  This is how the controller runs on the vessel,
     and it is what makes chattering visible for thin boundary layers;

3. reads the internal variables of the control law with
   ``controller.signals(t, state)`` (sliding variables, commands...) and
   assembles a ``pandas.DataFrame`` with the phase-plane coordinates.

Phase-plane coordinates (errors are *actual minus desired*, so that the
sliding variables of ``smc_figure_eight`` are exactly ``s = e_dot + k e``)::

    n      = v_cmd / |v_cmd| = [cos psi_cmd, sin psi_cmd]   commanded course
    e_u    = n . (p_s - p_s,ref)          along-course lookahead error  [m]
    e_u'   = u - n . p_s,ref_dot          along-course speed error     [m/s]
    s_u    = e_u' + k_pos e_u  (= u - u_cmd)
    e_psi  = psi - psi_cmd  (wrapped)     heading error               [rad]
    e_psi' = r - psi_cmd_dot              yaw-rate error            [rad/s]
    s_r    = e_psi' + k_psi e_psi  (= r - r_cmd)

``e_dot`` uses the controller's own feed-forward terms (psi_cmd_dot,
p_s,ref_dot), so the identities above are exact.
"""

from __future__ import annotations

import datetime as _dt
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from dashboard02_smc import PROJECT_ROOT  # noqa: F401
from dashboard02_smc import config as cfg
from dashboard02_smc import metrics as met

import controllers
import disturbances
import usv_model

# Same tolerances as usv_model.simulate.
RTOL = 1e-8
ATOL = 1e-10


class SimulationError(RuntimeError):
    """The simulator could not produce a valid result."""


class SimulationTimeoutError(SimulationError):
    """The run exceeded the allowed wall-clock time."""


# ---------------------------------------------------------------------------
# Builders (config -> existing simulator objects).
# ---------------------------------------------------------------------------
def build_controller(config):
    """``controllers.smc_figure_eight`` with the gains of ``config``."""
    return controllers.smc_figure_eight(**config.smc_kwargs())


def build_disturbance(dist):
    """Existing disturbance function of ``disturbances.py`` for ``dist``.

    Same mapping as the first dashboard: the defaults reproduce the modes of
    ``run_simulation.DISTURBANCE_OPTIONS``.
    """
    kind = dist.kind
    if kind == "none":
        return disturbances.no_disturbance
    if kind == "constant":
        d_u, d_v, d_r = dist.constant
        return disturbances.constant_disturbance(dTu=d_u, dTv=d_v, dTr=d_r)
    if kind == "sinusoidal":
        return disturbances.sinusoidal_disturbance(
            amplitude=dist.sin_amplitude, frequency=dist.sin_frequency_hz,
            phase=dist.sin_phase_rad, bias=dist.sin_bias)
    if kind == "directional":
        return disturbances.directional_force_disturbance(
            magnitude=dist.dir_magnitude_N, direction_deg=dist.dir_direction_deg)
    if kind == "wind":
        return disturbances.wind_like_disturbance
    if kind == "combined":
        # Same component order as run_simulation.DISTURBANCE_OPTIONS["combined"].
        components = []
        if dist.include_wind:
            components.append(disturbances.wind_like_disturbance)
        components.append(disturbances.directional_force_disturbance(
            magnitude=dist.dir_magnitude_N, direction_deg=dist.dir_direction_deg))
        components.append(disturbances.sinusoidal_disturbance(
            amplitude=dist.sin_amplitude, frequency=dist.sin_frequency_hz,
            phase=dist.sin_phase_rad))
        return disturbances.combined_disturbance(*components)
    raise ValueError(f"Unknown disturbance model '{kind}'")


class _Deadline:
    def __init__(self, max_wall_time_s):
        self.limit = max_wall_time_s
        self.deadline = (None if max_wall_time_s is None
                         else time.perf_counter() + float(max_wall_time_s))

    def check(self, t):
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise SimulationTimeoutError(
                f"Simulation aborted after {self.limit:g} s of wall-clock time "
                f"(reached t = {t:.2f} s)."
            )


# ---------------------------------------------------------------------------
# Integration.
# ---------------------------------------------------------------------------
def simulate_continuous(controller, disturbance_fun, state0, t_final, dt,
                        deadline):
    """``usv_model.simulate`` (as ``run_simulation.main``), with a deadline."""

    def tau_fun(t, state):
        deadline.check(t)
        return controller(t, state)

    t, state = usv_model.simulate(
        state0=state0, tau_fun=tau_fun, t_final=t_final, dt=dt,
        disturbance_fun=disturbance_fun,
    )
    signals = [controller.signals(ti, xi) for ti, xi in zip(t, state)]
    return t, state, signals


def simulate_digital(controller, disturbance_fun, state0, t_final, Ts,
                     deadline):
    """Sampled controller with zero-order hold.

    At ``t_k = k Ts`` the controller is evaluated on the sampled state; its
    delivered generalized forces are applied unchanged on ``[t_k, t_k+1)``
    while ``usv_model.vessel_rhs`` (kinematics, identified dynamics and the
    continuous-time disturbance) is integrated.  The returned samples are the
    states at ``t_k`` and the controller variables computed there, i.e.
    exactly what the digital controller saw and commanded.
    """
    n = int(np.floor(t_final / Ts + 1e-9))
    t = np.arange(n + 1, dtype=float) * Ts
    states = np.empty((n + 1, 6))
    signals = []
    x = np.asarray(state0, dtype=float)
    for k in range(n + 1):
        deadline.check(t[k])
        states[k] = x
        sig = controller.signals(t[k], x)
        signals.append(sig)
        if k == n:
            break
        tau_k = np.asarray(sig["tau_delivered"], dtype=float)

        # vessel_rhs adds the disturbance to the returned array in place,
        # so a fresh copy of the held value is returned at every call.
        def held_tau(_t, _state, tau_k=tau_k):
            return tau_k.copy()

        def rhs(ti, xi, held_tau=held_tau):
            return usv_model.vessel_rhs(ti, xi, held_tau, disturbance_fun)

        sol = solve_ivp(rhs, (t[k], t[k + 1]), x, rtol=RTOL, atol=ATOL,
                        first_step=t[k + 1] - t[k])
        if not sol.success:
            raise SimulationError(f"Integration failed at t = {t[k]:.3f} s: "
                                  f"{sol.message}")
        x = sol.y[:, -1]
    return t, states, signals


# ---------------------------------------------------------------------------
# Result container.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SimulationResult:
    """Outputs of one run, permanently tied to the config that produced them."""

    config: cfg.SimulationConfig
    data: pd.DataFrame
    metrics: dict
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: str = field(
        default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds")
    )
    solver_time_s: float = float("nan")
    total_time_s: float = float("nan")
    warnings: tuple = ()

    def to_csv_bytes(self):
        return self.data.to_csv(index=False).encode("utf-8")


def _stack(signals, name):
    return np.array([s[name] for s in signals], dtype=float)


def build_time_series(config, t, state, signals, disturbance_fun):
    """All time series of a run (one row per output / controller sample)."""
    ref = config.reference
    x, y, psi, u, v, r = state[:, :6].T
    x_ref, y_ref, *_ = controllers.figure_eight_reference(
        t, ref.amplitude_m, ref.period_s
    )
    p_s = _stack(signals, "p_s")
    p_s_ref = _stack(signals, "p_s_ref")
    p_s_dot_ref = _stack(signals, "p_s_dot_ref")
    v_cmd = _stack(signals, "v_cmd")
    psi_cmd = _stack(signals, "psi_cmd")
    tau_req = _stack(signals, "tau_required")
    tau_del = _stack(signals, "tau_delivered")
    forces = _stack(signals, "thruster_forces")
    s_u = _stack(signals, "s_u")
    s_r = _stack(signals, "s_r")

    # Commanded course direction n = v_cmd/|v_cmd| and its normal.
    n = v_cmd / np.maximum(np.linalg.norm(v_cmd, axis=1), 1e-12)[:, None]
    n_perp = np.column_stack([-n[:, 1], n[:, 0]])
    delta_p = p_s - p_s_ref

    disturbance = np.array(
        [disturbance_fun(ti, xi) for ti, xi in zip(t, state)], dtype=float
    ).reshape(len(t), 3)

    surge, yaw = config.surge, config.yaw
    df = pd.DataFrame({
        "t": t, "x": x, "y": y, "psi": psi, "u": u, "v": v, "r": r,
        "x_ref": x_ref, "y_ref": y_ref,
        "x_s": p_s[:, 0], "y_s": p_s[:, 1],
        "x_s_ref": p_s_ref[:, 0], "y_s_ref": p_s_ref[:, 1],
        # Position errors.
        "e_p": np.linalg.norm(delta_p, axis=1),
        "e_c": np.hypot(x_ref - x, y_ref - y),
        "e_cross": np.sum(n_perp * delta_p, axis=1),
        # Surge SMC.
        "u_cmd": _stack(signals, "speed_cmd"),
        "u_cmd_dot": _stack(signals, "speed_dot_cmd"),
        "e_u": np.sum(n * delta_p, axis=1),
        "e_u_dot": u - np.sum(n * p_s_dot_ref, axis=1),
        "s_u": s_u,
        "sat_u": np.clip(s_u / surge.phi_u, -1.0, 1.0),
        "u_dot_des": _stack(signals, "u_dot_des"),
        # Yaw SMC (e_psi of the controller is psi_cmd - psi).
        "psi_cmd": psi_cmd,
        "psi_cmd_dot": _stack(signals, "psi_dot_cmd"),
        "r_cmd": _stack(signals, "r_cmd"),
        "e_psi": -_stack(signals, "e_psi"),
        "e_psi_dot": r - _stack(signals, "psi_dot_cmd"),
        "s_r": s_r,
        "sat_r": np.clip(s_r / yaw.phi_r, -1.0, 1.0),
        "r_dot_des": _stack(signals, "r_dot_des"),
        # Control inputs.
        "Tu_cmd": tau_req[:, 0], "Tr_cmd": tau_req[:, 2],
        "Tu": tau_del[:, 0], "Tr": tau_del[:, 2],
        "F1": forces[:, 0], "F2": forces[:, 1],
        "F3": forces[:, 2], "F4": forces[:, 3],
        # Disturbances.
        "dTu": disturbance[:, 0], "dTv": disturbance[:, 1],
        "dTr": disturbance[:, 2],
    })
    return df


def run_from_config(config, *, max_wall_time_s=None):
    """Validate ``config``, run the simulator and return a ``SimulationResult``.

    Raises ``config.ConfigValidationError`` for invalid configurations and
    ``SimulationError`` if the integration fails, times out or diverges.
    """
    if not isinstance(config, cfg.SimulationConfig):
        raise TypeError("run_from_config expects a SimulationConfig")
    report = cfg.validate_config(config)
    if not report.ok:
        raise cfg.ConfigValidationError(report.errors)

    start = time.perf_counter()
    deadline = _Deadline(max_wall_time_s)
    controller = build_controller(config)
    disturbance_fun = build_disturbance(config.disturbance)
    sim = config.simulation
    state0 = config.initial_state()

    try:
        if sim.mode == cfg.DIGITAL:
            t, state, signals = simulate_digital(
                controller, disturbance_fun, state0, config.t_final_s,
                sim.sample_period_s, deadline,
            )
        else:
            t, state, signals = simulate_continuous(
                controller, disturbance_fun, state0, config.t_final_s,
                sim.output_dt_s, deadline,
            )
    except SimulationError:
        raise
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        raise SimulationError(f"Integration failed: {exc}") from exc
    solver_time = time.perf_counter() - start

    if not np.all(np.isfinite(state)):
        raise SimulationError("The simulated state became non-finite (diverged).")
    if len(t) < 2:
        raise SimulationError("The simulation produced fewer than two samples.")

    data = build_time_series(config, t, state, signals, disturbance_fun)
    return SimulationResult(
        config=config,
        data=data,
        metrics=met.compute_metrics(data, config),
        solver_time_s=solver_time,
        total_time_s=time.perf_counter() - start,
        warnings=tuple(report.warnings),
    )
