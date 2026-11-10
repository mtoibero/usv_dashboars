"""Adapter between a ``SimulationConfig`` and the existing simulator.

This module contains no dynamics or control law.  It only

1. builds the existing controller / disturbance objects from a config
   (``controllers.smc_figure_eight``, ``controllers.feedback_figure_eight``,
   ``disturbances.*``),
2. calls ``usv_model.simulate``, exactly as ``run_simulation.main`` does,
3. post-processes with the existing helpers
   (``run_simulation.controller_output_history``,
   ``run_simulation.reference_history``,
   ``usv_model.thruster_forces_to_pwm_us``) into a ``pandas.DataFrame`` whose
   first columns match the CSV written by ``run_simulation.save_csv``.

It has no Streamlit dependency and can be used from scripts::

    from dashboard.config import SimulationConfig
    from dashboard.engine import run_from_config
    result = run_from_config(SimulationConfig.from_json(text))
"""

from __future__ import annotations

import datetime as _dt
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dashboard import PROJECT_ROOT  # noqa: F401
from dashboard import config as cfg
from dashboard import metrics as met

import controllers
import disturbances
import run_simulation
import usv_model


class SimulationError(RuntimeError):
    """The simulator could not produce a valid result."""


class SimulationTimeoutError(SimulationError):
    """The run exceeded the allowed wall-clock time."""


# Column names in the order used by run_simulation.save_csv (no CBF/governor).
CLI_CSV_COLUMNS = (
    ["t", "x", "y", "psi", "u", "v", "r"]
    + ["x_ref", "y_ref", "psi_ref", "u_ref", "r_ref",
       "x_s", "y_s", "x_s_dot", "y_s_dot",
       "x_s_ref", "y_s_ref", "x_s_dot_ref", "y_s_dot_ref"]
    + ["tau_u_cmd", "tau_r_cmd", "tau_u_real", "tau_r_real",
       "F1", "F2", "F3", "F4", "PWM1", "PWM2", "PWM3", "PWM4"]
)


# ---------------------------------------------------------------------------
# Builders (config -> existing simulator objects).
# ---------------------------------------------------------------------------
def build_controller(config):
    """Instantiate the existing controller described by ``config``."""
    ref, ctrl = config.reference, config.controller
    if isinstance(ctrl, cfg.SMCConfig):
        return controllers.smc_figure_eight(
            k_pos=ctrl.k_pos,
            k_psi=ctrl.k_psi,
            lambda_u=ctrl.lambda_u,
            lambda_r=ctrl.lambda_r,
            k_s_u=ctrl.k_s_u,
            k_s_r=ctrl.k_s_r,
            phi_u=ctrl.phi_u,
            phi_r=ctrl.phi_r,
            lookahead_L=ctrl.lookahead_L,
            amplitude=ref.amplitude_m,
            period=ref.period_s,
        )
    if isinstance(ctrl, cfg.FeedbackConfig):
        return controllers.feedback_figure_eight(
            k_pos=ctrl.k_pos,
            k_u=ctrl.k_u,
            k_psi=ctrl.k_psi,
            k_r=ctrl.k_r,
            amplitude=ref.amplitude_m,
            period=ref.period_s,
        )
    raise TypeError(f"Unsupported controller config {type(ctrl).__name__}")


def build_disturbance(dist):
    """Instantiate the existing disturbance function described by ``dist``."""
    if isinstance(dist, cfg.NoDisturbance):
        return disturbances.no_disturbance
    if isinstance(dist, cfg.ConstantDisturbance):
        return disturbances.constant_disturbance(
            dTu=dist.dTu, dTv=dist.dTv, dTr=dist.dTr
        )
    if isinstance(dist, cfg.SinusoidalDisturbance):
        return disturbances.sinusoidal_disturbance(
            amplitude=dist.amplitude,
            frequency=dist.frequency_hz,
            phase=dist.phase_rad,
            bias=dist.bias,
        )
    if isinstance(dist, cfg.DirectionalDisturbance):
        return disturbances.directional_force_disturbance(
            magnitude=dist.magnitude_N, direction_deg=dist.direction_deg
        )
    if isinstance(dist, cfg.WindLikeDisturbance):
        return disturbances.wind_like_disturbance
    if isinstance(dist, cfg.CombinedDisturbance):
        # Same component order as run_simulation.DISTURBANCE_OPTIONS["combined"].
        components = []
        if dist.include_wind:
            components.append(disturbances.wind_like_disturbance)
        components.append(disturbances.directional_force_disturbance(
            magnitude=dist.directional_magnitude_N,
            direction_deg=dist.directional_direction_deg,
        ))
        components.append(disturbances.sinusoidal_disturbance(
            amplitude=dist.sin_amplitude,
            frequency=dist.sin_frequency_hz,
            phase=dist.sin_phase_rad,
        ))
        return disturbances.combined_disturbance(*components)
    raise TypeError(f"Unsupported disturbance config {type(dist).__name__}")


def _with_deadline(tau_fun, max_wall_time_s):
    """Wrap ``tau_fun`` so that integration aborts after a wall-clock limit.

    The returned values are those of ``tau_fun``; numerics are unchanged.
    """
    if max_wall_time_s is None:
        return tau_fun
    deadline = time.perf_counter() + float(max_wall_time_s)

    def guarded(t, state):
        if time.perf_counter() > deadline:
            raise SimulationTimeoutError(
                f"Simulation aborted after {max_wall_time_s:g} s of wall-clock "
                f"time (reached t = {t:.2f} s). Thin boundary layers or very "
                "high gains make the adaptive integrator take tiny steps."
            )
        return tau_fun(t, state)

    return guarded


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

    @property
    def t(self):
        return self.data["t"].to_numpy()

    def to_csv_bytes(self):
        """CSV: CLI columns (same names/order as save_csv) + extra columns."""
        return self.data.to_csv(index=False).encode("utf-8")


def build_time_series(config, t, state, controller, disturbance_fun):
    """Assemble all time series of a run with the existing helper functions."""
    ref = config.reference
    lookahead_L = config.evaluation_lookahead_L

    tau_required, tau_delivered, thruster_forces = (
        run_simulation.controller_output_history(controller, t, state)
    )
    references = run_simulation.reference_history(
        t, state, lookahead_L, None,
        amplitude=ref.amplitude_m, period=ref.period_s,
    )
    pwm_us = usv_model.thruster_forces_to_pwm_us(thruster_forces)
    disturbance = np.array(
        [disturbance_fun(ti, state_i) for ti, state_i in zip(t, state)],
        dtype=float,
    ).reshape(len(t), 3)

    columns = np.column_stack([
        t, state[:, :6], *references,
        tau_required[:, 0], tau_required[:, 2],
        tau_delivered[:, 0], tau_delivered[:, 2],
        thruster_forces, pwm_us,
    ])
    df = pd.DataFrame(columns, columns=CLI_CSV_COLUMNS)
    df["dTu"], df["dTv"], df["dTr"] = disturbance.T
    return met.add_error_columns(df)


def run_from_config(config, *, max_wall_time_s=None):
    """Validate ``config``, run the simulator and return a ``SimulationResult``.

    Raises ``config.ConfigValidationError`` for invalid configurations and
    ``SimulationError`` if integration fails, times out or diverges.
    """
    if not isinstance(config, cfg.SimulationConfig):
        raise TypeError("run_from_config expects a SimulationConfig")
    report = cfg.validate_config(config)
    if not report.ok:
        raise cfg.ConfigValidationError(report.errors)

    start = time.perf_counter()
    controller = build_controller(config)
    disturbance_fun = build_disturbance(config.disturbance)
    sim = config.simulation

    try:
        t, state = usv_model.simulate(
            state0=sim.initial_state(),
            tau_fun=_with_deadline(controller, max_wall_time_s),
            t_final=sim.t_final_s,
            dt=sim.dt_s,
            disturbance_fun=disturbance_fun,
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

    data = build_time_series(config, t, state, controller, disturbance_fun)
    return SimulationResult(
        config=config,
        data=data,
        metrics=met.compute_metrics(data),
        solver_time_s=solver_time,
        total_time_s=time.perf_counter() - start,
        warnings=tuple(report.warnings),
    )
