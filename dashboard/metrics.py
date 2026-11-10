"""Tracking-error columns and performance metrics (no Streamlit imports).

Error definitions follow the existing simulator:

* Lookahead-point position error ``e_s = ||p_s,ref - p_s||`` is the
  quantity regulated by the SMC and the one plotted by
  ``plot_results.plot_tracking_error``; it is computed with
  ``analysis_utils.position_error``.
* Hull-centre error ``e_c = ||p_ref - p||`` is controller-agnostic and is
  the fairest metric when comparing controllers with different lookahead.
* Thruster saturation uses ``analysis_utils.saturation_duty_cycle``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard import PROJECT_ROOT  # noqa: F401

from analysis_utils import (
    is_saturated,
    position_error,
    saturation_duty_cycle,
    wrap_angle,
)
from usv_model import THRUSTER_FWD_MAX, THRUSTER_REV_MAX

STEADY_STATE_FRACTION = 0.25   # final fraction of the run used as "steady state"
SATURATION_TOL_N = 1e-6        # |F - limit| below this counts as saturated
DEFAULT_TOLERANCE_M = 0.5      # default pass/fail tolerance shown in the UI
SATURATION_WARNING_PCT = 5.0


def add_error_columns(df):
    """Return a copy of ``df`` with tracking-error columns appended."""
    df = df.copy()
    df["e_s_x"] = df["x_s_ref"] - df["x_s"]
    df["e_s_y"] = df["y_s_ref"] - df["y_s"]
    df["e_s"] = position_error(df)
    df["e_s_dot"] = np.hypot(
        df["x_s_dot_ref"] - df["x_s_dot"], df["y_s_dot_ref"] - df["y_s_dot"]
    )
    df["e_c_x"] = df["x_ref"] - df["x"]
    df["e_c_y"] = df["y_ref"] - df["y"]
    df["e_c"] = np.hypot(df["e_c_x"], df["e_c_y"])
    df["e_psi"] = wrap_angle(df["psi_ref"].to_numpy() - df["psi"].to_numpy())
    df["e_u"] = df["u_ref"] - df["u"]
    return df


def _rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def _total_variation_rate(values, duration):
    values = np.asarray(values, dtype=float)
    if values.size < 2 or duration <= 0.0:
        return float("nan")
    return float(np.sum(np.abs(np.diff(values))) / duration)


def steady_state_mask(t, fraction=STEADY_STATE_FRACTION):
    t = np.asarray(t, dtype=float)
    return t >= t[-1] - fraction * (t[-1] - t[0])


def compute_metrics(df, steady_state_fraction=STEADY_STATE_FRACTION):
    """Scalar performance metrics of one run (``df`` needs error columns)."""
    t = df["t"].to_numpy()
    duration = float(t[-1] - t[0])
    ss = steady_state_mask(t, steady_state_fraction)
    e_s, e_c = df["e_s"].to_numpy(), df["e_c"].to_numpy()
    e_psi_deg = np.degrees(df["e_psi"].to_numpy())

    duty = saturation_duty_cycle(
        df, THRUSTER_FWD_MAX, -THRUSTER_REV_MAX, SATURATION_TOL_N
    )
    forces = df[["F1", "F2", "F3", "F4"]].to_numpy()
    any_saturated = is_saturated(
        forces, THRUSTER_FWD_MAX, -THRUSTER_REV_MAX, SATURATION_TOL_N
    ).any(axis=1)

    metrics = {
        "lookahead_err_max_m": float(np.max(e_s)),
        "lookahead_err_final_m": float(e_s[-1]),
        "lookahead_err_rms_m": _rms(e_s),
        "lookahead_err_ss_max_m": float(np.max(e_s[ss])),
        "lookahead_err_ss_rms_m": _rms(e_s[ss]),
        "center_err_max_m": float(np.max(e_c)),
        "center_err_final_m": float(e_c[-1]),
        "center_err_rms_m": _rms(e_c),
        "center_err_ss_max_m": float(np.max(e_c[ss])),
        "heading_err_max_deg": float(np.max(np.abs(e_psi_deg))),
        "heading_err_rms_deg": _rms(e_psi_deg),
        "surge_speed_err_rms_mps": _rms(df["e_u"].to_numpy()),
        "tau_u_rms_N": _rms(df["tau_u_real"].to_numpy()),
        "tau_r_rms_Nm": _rms(df["tau_r_real"].to_numpy()),
        "tau_u_variation_Nps": _total_variation_rate(
            df["tau_u_real"].to_numpy(), duration
        ),
        "tau_r_variation_Nmps": _total_variation_rate(
            df["tau_r_real"].to_numpy(), duration
        ),
        "saturation_any_pct": 100.0 * float(np.mean(any_saturated)),
    }
    for thruster, fraction in duty.items():
        metrics[f"saturation_{thruster}_pct"] = 100.0 * fraction
    metrics["t_final_s"] = float(t[-1])
    metrics["samples"] = int(len(t))
    return metrics


# (key, label, unit, description) - order used by the UI tables.
METRIC_INFO = [
    ("lookahead_err_max_m", "Max lookahead-point error", "m",
     "maximum over the whole run (includes the transient)"),
    ("lookahead_err_final_m", "Final lookahead-point error", "m",
     "value at the final time"),
    ("lookahead_err_rms_m", "RMS lookahead-point error", "m",
     "root mean square over the whole run"),
    ("lookahead_err_ss_max_m", "Steady-state max lookahead error", "m",
     f"max over the final {STEADY_STATE_FRACTION:.0%} of the run"),
    ("lookahead_err_ss_rms_m", "Steady-state RMS lookahead error", "m",
     f"RMS over the final {STEADY_STATE_FRACTION:.0%} of the run"),
    ("center_err_max_m", "Max hull-centre error", "m",
     "maximum over the whole run (vessel centre vs reference point)"),
    ("center_err_final_m", "Final hull-centre error", "m",
     "value at the final time"),
    ("center_err_rms_m", "RMS hull-centre error", "m",
     "root mean square over the whole run"),
    ("center_err_ss_max_m", "Steady-state max hull-centre error", "m",
     f"max over the final {STEADY_STATE_FRACTION:.0%} of the run"),
    ("heading_err_max_deg", "Max |heading error|", "deg",
     "max |ψ_ref − ψ| (wrapped), ψ_ref = reference path tangent"),
    ("heading_err_rms_deg", "RMS heading error", "deg", ""),
    ("surge_speed_err_rms_mps", "RMS surge-speed error", "m/s",
     "RMS of (‖ṗ_ref‖ − u)"),
    ("tau_u_rms_N", "RMS delivered surge force", "N", "control effort"),
    ("tau_r_rms_Nm", "RMS delivered yaw moment", "N m", "control effort"),
    ("tau_u_variation_Nps", "Surge-force variation rate", "N/s",
     "Σ|ΔTᵤ| / Tₛᵢₘ at the output rate (chattering indicator)"),
    ("tau_r_variation_Nmps", "Yaw-moment variation rate", "N m/s",
     "Σ|ΔTᵣ| / Tₛᵢₘ at the output rate (chattering indicator)"),
    ("saturation_any_pct", "Time with any thruster saturated", "%",
     "share of samples with at least one thruster at its limit"),
    ("saturation_F1_pct", "Saturation duty F1 (left)", "%", ""),
    ("saturation_F2_pct", "Saturation duty F2 (left)", "%", ""),
    ("saturation_F3_pct", "Saturation duty F3 (right)", "%", ""),
    ("saturation_F4_pct", "Saturation duty F4 (right)", "%", ""),
    ("t_final_s", "Simulated time", "s", ""),
    ("samples", "Output samples", "-", ""),
]


def metrics_table(metrics):
    """DataFrame (Metric, Value, Unit, Definition) for display."""
    rows = [
        {"Metric": label, "Value": metrics.get(key, np.nan), "Unit": unit,
         "Definition": description}
        for key, label, unit, description in METRIC_INFO
    ]
    return pd.DataFrame(rows)


# Subset shown next to the XY plot when the previous run is overlaid.
KEY_METRICS = (
    "lookahead_err_max_m",
    "lookahead_err_final_m",
    "lookahead_err_ss_rms_m",
    "center_err_max_m",
    "center_err_rms_m",
    "heading_err_rms_deg",
    "tau_r_rms_Nm",
    "saturation_any_pct",
)


def comparison_table(metrics_a, metrics_b, keys=None):
    """Side-by-side metrics with absolute and relative differences.

    ``keys`` optionally restricts (and orders) the metrics shown.
    """
    info = {key: (key, label, unit, d) for key, label, unit, d in METRIC_INFO}
    selected = METRIC_INFO if keys is None else [info[key] for key in keys]
    rows = []
    for key, label, unit, _ in selected:
        a, b = metrics_a.get(key, np.nan), metrics_b.get(key, np.nan)
        delta = b - a
        relative = 100.0 * delta / abs(a) if a not in (0, 0.0) else np.nan
        rows.append({"Metric": label, "Unit": unit, "Run A": a, "Run B": b,
                     "B - A": delta, "Change %": relative})
    return pd.DataFrame(rows)


def evaluate_indicators(metrics, tolerance_m=DEFAULT_TOLERANCE_M):
    """Success/failure indicators for a completed run.

    Returns a list of dicts with ``indicator``, ``passed`` (True/False) and
    ``detail``.  A run that fails to integrate never reaches this point; the
    dashboard reports it as a failed run instead.
    """
    final = metrics["lookahead_err_final_m"]
    ss_max = metrics["lookahead_err_ss_max_m"]
    saturation = metrics["saturation_any_pct"]
    return [
        {"indicator": "Simulation completed (solver success)", "passed": True,
         "detail": f"{metrics['samples']} samples up to "
                   f"{metrics['t_final_s']:g} s"},
        {"indicator": f"Final lookahead error ≤ {tolerance_m:g} m",
         "passed": final <= tolerance_m, "detail": f"{final:.3f} m"},
        {"indicator": f"Steady-state max lookahead error ≤ {tolerance_m:g} m",
         "passed": ss_max <= tolerance_m, "detail": f"{ss_max:.3f} m "
         f"(final {STEADY_STATE_FRACTION:.0%} of the run)"},
        {"indicator": f"Thruster saturation < {SATURATION_WARNING_PCT:g}% of time",
         "passed": saturation < SATURATION_WARNING_PCT,
         "detail": f"{saturation:.1f}% of samples"},
    ]
