"""Plotly figures of the SMC dashboard (no Streamlit imports).

Every figure overlays two runs on the *same axes*:

* the current run as a thin line in a bright colour, on top, and
* a comparison run (by default the previous one) as a wide pale-gray line
  underneath, so it remains visible where both runs coincide.

Legend entries carry the parameters of each run.  Plotly inside Streamlit
cannot render LaTeX, so labels use Unicode/HTML (λ<sub>u</sub>, φ<sub>r</sub>);
the equations themselves are rendered with KaTeX by the app.

``t_window`` (display only) restricts the plotted time range.  Signals longer
than ``MAX_BUCKETS`` samples are reduced with a min/max envelope (two points
per bucket) that keeps chattering visible instead of aliasing it.  Indices
always use the full-resolution data.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard02_smc import PROJECT_ROOT  # noqa: F401
from dashboard02_smc import config as cfg
from dashboard02_smc import metrics as met

import controllers
from usv_model import THRUSTER_FWD_MAX, THRUSTER_REV_MAX, TR_LIMITS, TU_LIMITS

CURRENT_COLOR = "#0050ff"                  # bright: current run
CURRENT_FILL = "rgba(0, 80, 255, 0.13)"
COMMANDED_COLOR = "#00b8ff"
PREVIOUS_COLOR = "#bdbdbd"                 # pale: comparison run
PREVIOUS_EDGE = "#8f8f8f"
PREVIOUS_FILL = "rgba(140, 140, 140, 0.18)"
REF_COLOR = "#222222"
LIMIT_COLOR = "#d62728"
CURRENT_WIDTH = 1.8
PREVIOUS_WIDTH = 5.0
MAX_BUCKETS = 2500                         # -> at most ~5000 points per trace
LEGEND_ROW_PX = 78                         # height of one legend row (4 lines)


# ---------------------------------------------------------------------------
# Labels.
# ---------------------------------------------------------------------------
def _v(value):
    return cfg.format_value(value)


def disturbance_html(dist):
    """Short description of the disturbance model for legends."""
    kind = dist.kind
    if kind == "none":
        return "no disturbance"
    if kind == "constant":
        d_u, d_v, d_r = (_v(c) for c in dist.constant)
        return f"δT = [{d_u} N, {d_v} N, {d_r} N m]"
    if kind == "sinusoidal":
        amps = ", ".join(_v(a) for a in dist.sin_amplitude)
        freqs = ", ".join(_v(f) for f in dist.sin_frequency_hz)
        text = f"sinusoidal A = ({amps}), f = ({freqs}) Hz"
        if any(dist.sin_bias):
            text += f", b = ({', '.join(_v(b) for b in dist.sin_bias)})"
        return text
    if kind == "directional":
        return (f"directional F = {_v(dist.dir_magnitude_N)} N, "
                f"β = {_v(dist.dir_direction_deg)}°")
    if kind == "wind":
        return "wind-like (fixed profile)"
    amps = ", ".join(_v(a) for a in dist.sin_amplitude)
    wind = "wind + " if dist.include_wind else ""
    return (f"combined: {wind}F = {_v(dist.dir_magnitude_N)} N at "
            f"{_v(dist.dir_direction_deg)}° + sin A = ({amps})")


def matched_bound_html(dist, axis):
    bound = dist.bounds()[0 if axis == "u" else 2]
    unit = "N m" if axis == "r" else "N"
    return f"max|δT<sub>{axis}</sub>| = {_v(bound)} {unit}"


def surge_params_html(config):
    s = config.surge
    return (f"λ<sub>u</sub>={_v(s.lambda_u)}, k<sub>s,u</sub>={_v(s.k_s_u)}, "
            f"φ<sub>u</sub>={_v(s.phi_u)}, k<sub>pos</sub>={_v(s.k_pos)}")


def yaw_params_html(config):
    y = config.yaw
    return (f"λ<sub>r</sub>={_v(y.lambda_r)}, k<sub>s,r</sub>={_v(y.k_s_r)}, "
            f"φ<sub>r</sub>={_v(y.phi_r)}, k<sub>ψ</sub>={_v(y.k_psi)}")


def context_html(config):
    ref, sim = config.reference, config.simulation
    impl = (f"digital f<sub>s</sub>={_v(sim.sample_rate_hz)} Hz"
            if sim.mode == cfg.DIGITAL else "continuous")
    text = (f"f={_v(ref.frequency_hz)} Hz, v<sub>max</sub>="
            f"{_v(ref.peak_speed_mps)} m/s, A={_v(ref.amplitude_m)} m; {impl}")
    return text + "; " + disturbance_html(config.disturbance)


def run_title(record, role):
    """'Current · Run #3' (+ user label)."""
    label = f" “{record.label}”" if record.label else ""
    return f"<b>{role} · Run #{record.number}</b>{label}"


def run_label(record, role, channel="all"):
    """Legend text with the parameters of ``record`` relevant to a figure."""
    config = record.config
    title = run_title(record, role)
    if channel == "surge":
        return (f"{title}<br>{surge_params_html(config)}<br>"
                f"{matched_bound_html(config.disturbance, 'u')}")
    if channel == "yaw":
        return (f"{title}<br>{yaw_params_html(config)}<br>"
                f"{matched_bound_html(config.disturbance, 'r')}")
    return (f"{title}<br>Surge: {surge_params_html(config)}<br>"
            f"Yaw: {yaw_params_html(config)}<br>{context_html(config)}")


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def envelope(t, y, max_buckets=MAX_BUCKETS):
    """Min/max decimation: keeps the envelope of fast oscillations."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size
    if n <= 2 * max_buckets:
        return t, y
    size = int(math.ceil(n / max_buckets))
    usable = (n // size) * size
    yb = y[:usable].reshape(-1, size)
    base = np.arange(0, usable, size)
    idx = np.concatenate([base + np.argmin(yb, axis=1),
                          base + np.argmax(yb, axis=1),
                          np.arange(usable, n), [0, n - 1]])
    idx = np.unique(idx)
    return t[idx], y[idx]


def stride(values, max_points=2 * MAX_BUCKETS):
    values = np.asarray(values)
    step = max(1, int(math.ceil(len(values) / max_points)))
    out = values[::step]
    if step > 1 and (len(values) - 1) % step:
        out = np.append(out, values[-1:], axis=0)
    return out


def window(data, t_window):
    """Rows of ``data`` with t inside ``t_window`` (all rows if None/empty)."""
    if t_window is None:
        return data
    t0, t1 = t_window
    mask = (data["t"] >= t0) & (data["t"] <= t1)
    return data[mask] if mask.sum() >= 2 else data


def _runs(current, previous):
    """(record, role, colour, width) tuples, comparison run first."""
    runs = []
    if previous is not None:
        runs.append((previous, "Previous", PREVIOUS_COLOR, PREVIOUS_WIDTH))
    runs.append((current, "Current", CURRENT_COLOR, CURRENT_WIDTH))
    return runs


def _series(fig, data, name, legend_name, color, width, row, col,
            legendgroup, showlegend, dash=None):
    t, y = envelope(data["t"].to_numpy(), data[name].to_numpy())
    fig.add_trace(go.Scatter(
        x=t, y=y, mode="lines", name=legend_name, legendgroup=legendgroup,
        showlegend=showlegend, line=dict(color=color, width=width, dash=dash)),
        row=row, col=col)


def _band(fig, t0, t1, phi, fill, row, col, name=None):
    """Shaded boundary layer |s| <= phi over [t0, t1]."""
    fig.add_trace(go.Scatter(
        x=[t0, t1, t1, t0, t0], y=[-phi, -phi, phi, phi, -phi], mode="lines",
        fill="toself", fillcolor=fill, line=dict(width=0), hoverinfo="skip",
        name=name or "", showlegend=name is not None), row=row, col=col)


def _layout(fig, plot_height, legend_rows=1):
    """Legend at the top of the figure, above every subplot title."""
    top = 40 + LEGEND_ROW_PX * legend_rows
    fig.update_layout(
        height=plot_height + top,
        margin=dict(l=70, r=20, t=top, b=50),
        legend=dict(orientation="h", yref="container", y=1.0,
                    yanchor="top", x=0.0, font=dict(size=11)),
        hovermode="x unified",
    )
    return fig


def reference_path(reference, samples=800):
    t = np.linspace(0.0, reference.period_s, samples)
    x, y, *_ = controllers.figure_eight_reference(
        t, reference.amplitude_m, reference.period_s
    )
    return x, y


# ---------------------------------------------------------------------------
# 1) XY trajectory.
# ---------------------------------------------------------------------------
DISTURBANCE_COLOR = "rgba(214, 96, 0, 0.55)"     # subtle orange
DISTURBANCE_TEXT = "#8a5a2b"
N_FORCE_ARROWS = 24


def _inertial_force(data):
    """Disturbance force on the hull in the inertial frame (Fx, Fy) [N]."""
    psi = data["psi"].to_numpy()
    d_u, d_v = data["dTu"].to_numpy(), data["dTv"].to_numpy()
    return (np.cos(psi) * d_u - np.sin(psi) * d_v,
            np.sin(psi) * d_u + np.cos(psi) * d_v)


def _yaw_arrow(value):
    # psi grows counter-clockwise in the XY plot (y up): dTr > 0 turns left.
    return "↺" if value > 0 else "↻"


def disturbance_note(dist, data):
    """Two short lines describing the disturbance of the current run."""
    b_u, b_v, b_r = dist.bounds()
    line1 = f"Disturbance of the current run: {disturbance_html(dist)}"
    if dist.kind == "constant":
        line1 += " (body frame: rotates with the vessel)"
    parts = [f"max|δT<sub>u</sub>| = {_v(b_u)} N",
             f"max|δT<sub>v</sub>| = {_v(b_v)} N"]
    d_r = data["dTr"].to_numpy()
    if np.any(d_r != 0.0):
        if np.ptp(d_r) == 0.0:
            parts.append(f"δT<sub>r</sub> = {_v(d_r[0])} N m "
                         f"{_yaw_arrow(d_r[0])}")
        else:
            parts.append(f"δT<sub>r</sub> ∈ [{_v(d_r.min())}, "
                         f"{_v(d_r.max())}] N m (↺ > 0)")
    return line1 + "<br>" + " · ".join(parts)


def _force_arrows(fig, data, length):
    """Faint arrows of the disturbance force along the path of one run."""
    fx, fy = _inertial_force(data)
    magnitude = np.hypot(fx, fy)
    peak = float(np.max(magnitude)) if magnitude.size else 0.0
    if peak <= 1e-9:
        return None
    idx = np.unique(np.linspace(0, len(data) - 1, N_FORCE_ARROWS).astype(int))
    x, y = data["x"].to_numpy()[idx], data["y"].to_numpy()[idx]
    ux, uy = fx[idx] * length / peak, fy[idx] * length / peak
    xs, ys, sizes = [], [], []
    for xi, yi, dx, dy in zip(x, y, ux, uy):
        if dx == 0.0 and dy == 0.0:
            continue
        xs += [xi, xi + dx, None]
        ys += [yi, yi + dy, None]
        sizes += [0, 9, 0]
    spacing = float(data["t"].iloc[idx[1]] - data["t"].iloc[idx[0]]) \
        if len(idx) > 1 else 0.0
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers", hoverinfo="skip",
        name=(f"Disturbance force on the hull (every {spacing:.3g} s; "
              f"longest = {_v(peak)} N)"),
        line=dict(color=DISTURBANCE_COLOR, width=1.3),
        marker=dict(symbol="arrow", angleref="previous", size=sizes,
                    color=DISTURBANCE_COLOR)))
    return peak


def xy_figure(current, previous=None, t_window=None):
    """XY plot: reference, comparison run (gray), current run (bright).

    * Time window: the part of the current run inside ``t_window`` is drawn
      normally, the rest of the run faintly, with the window limits marked.
    * Disturbance (current run, subtle): a fixed-direction force (directional
      model or component) as one arrow with its true orientation in the band
      above the path; time-varying or body-fixed forces as faint arrows along
      the path (inertial frame); a short note with the model and its bounds.
    """
    fig = go.Figure()
    full = current.result.data
    data = window(full, t_window)
    windowed = len(data) < len(full)
    cur_ref = current.config.reference
    dist = current.config.disturbance

    if previous is not None and previous.config.reference != cur_ref:
        x, y = reference_path(previous.config.reference)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name="Reference of the comparison run",
            line=dict(color=PREVIOUS_EDGE, dash="dot", width=1.5)))
    x_ref, y_ref = reference_path(cur_ref)
    fig.add_trace(go.Scatter(
        x=x_ref, y=y_ref, mode="lines",
        name="Reference p<sub>ref</sub> (figure eight)",
        line=dict(color=REF_COLOR, dash="dash", width=1.3)))
    if windowed:
        xy = stride(full[["x", "y"]].to_numpy())
        fig.add_trace(go.Scatter(
            x=xy[:, 0], y=xy[:, 1], mode="lines", hoverinfo="skip",
            name="Current run outside the time window",
            line=dict(color=CURRENT_COLOR, width=1.0), opacity=0.22))
    for record, role, color, width in _runs(current, previous):
        run_data = window(record.result.data, t_window)
        xy = stride(run_data[["x", "y", "t"]].to_numpy())
        fig.add_trace(go.Scatter(
            x=xy[:, 0], y=xy[:, 1], mode="lines", customdata=xy[:, 2],
            hovertemplate="t = %{customdata:.1f} s<extra></extra>",
            name=run_label(record, role), legendgroup=role,
            line=dict(color=color, width=width + 0.6)))
        marker_color = color if role == "Current" else PREVIOUS_EDGE
        for point, symbol, size in ((0, "circle", 10), (-1, "x", 11)):
            fig.add_trace(go.Scatter(
                x=[xy[point, 0]], y=[xy[point, 1]], mode="markers",
                hovertemplate=f"{role}: t = {xy[point, 2]:.1f} s<extra></extra>",
                name=f"{role} start/end", legendgroup=role, showlegend=False,
                marker=dict(color=marker_color, size=size, symbol=symbol)))
    if windowed:
        for point, anchor in ((0, "left"), (-1, "right")):
            row = data.iloc[point]
            fig.add_annotation(
                x=row["x"], y=row["y"], text=f"t = {row['t']:.1f} s",
                showarrow=False, xanchor=anchor, yanchor="bottom", xshift=6
                if anchor == "left" else -6, yshift=4,
                font=dict(size=11, color=CURRENT_COLOR))

    # Axis ranges: the paths plus a free band above them for the
    # disturbance information (so it never covers the trajectory).
    xs = np.concatenate([x_ref, full["x"].to_numpy()]
                        + ([previous.result.data["x"].to_numpy()]
                           if previous is not None else []))
    ys = np.concatenate([y_ref, full["y"].to_numpy()]
                        + ([previous.result.data["y"].to_numpy()]
                           if previous is not None else []))
    x_lo, x_hi = float(xs.min()), float(xs.max())
    y_lo, y_hi = float(ys.min()), float(ys.max())
    span = max(x_hi - x_lo, y_hi - y_lo, 1.0)
    band = 0.22 * (y_hi - y_lo) if dist.active else 0.0
    pad = 0.05 * span

    if dist.active:
        has_fixed = dist.kind in ("directional", "combined") and \
            dist.dir_magnitude_N > 0.0
        if dist.kind != "directional":
            _force_arrows(fig, data, 0.08 * span)
        if has_fixed:
            beta = math.radians(dist.dir_direction_deg)
            length = 0.12 * span
            x0 = x_lo + 0.10 * span
            y0 = y_hi + pad + 0.5 * band
            fig.add_annotation(
                x=x0 + 0.5 * length * math.cos(beta),
                y=y0 + 0.5 * length * math.sin(beta),
                ax=x0 - 0.5 * length * math.cos(beta),
                ay=y0 - 0.5 * length * math.sin(beta),
                xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                arrowhead=2, arrowsize=1.3, arrowwidth=2.2,
                arrowcolor=DISTURBANCE_COLOR, text="")
            # Short label next to the arrow; the note names the model.
            fig.add_annotation(
                x=x0 + 0.6 * length + 0.02 * span, y=y0, xref="x", yref="y",
                text=(f"F = {_v(dist.dir_magnitude_N)} N, "
                      f"β = {_v(dist.dir_direction_deg)}°"),
                showarrow=False, xanchor="left", align="left",
                font=dict(size=11, color=DISTURBANCE_TEXT))
        fig.add_annotation(
            x=x_hi, y=y_hi + pad + band, xref="x", yref="y",
            text=disturbance_note(dist, data), showarrow=False,
            xanchor="right", yanchor="top", align="right",
            font=dict(size=10.5, color=DISTURBANCE_TEXT),
            bgcolor="rgba(255,255,255,0.75)")

    fig.update_xaxes(title_text="x [m]", range=[x_lo - pad, x_hi + pad],
                     constrain="domain")
    fig.update_yaxes(title_text="y [m]", range=[y_lo - pad, y_hi + pad + band],
                     scaleanchor="x", scaleratio=1)
    _layout(fig, 620 if dist.active else 560, legend_rows=1)
    fig.update_layout(hovermode="closest")
    return fig


# ---------------------------------------------------------------------------
# 2) Controller errors.
# ---------------------------------------------------------------------------
def errors_figure(current, previous=None, t_window=None):
    fig = make_subplots(
        rows=4, cols=2, shared_xaxes=True, vertical_spacing=0.07,
        horizontal_spacing=0.09,
        specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}], [{}, {}]],
        subplot_titles=(
            "Lookahead-point position error ‖e<sub>p</sub>‖ = "
            "‖p<sub>s</sub> − p<sub>s,ref</sub>‖",
            "Surge SMC · along-course error e<sub>u</sub>",
            "Yaw SMC · heading error e<sub>ψ</sub> = ψ − ψ<sub>cmd</sub>",
            "Surge SMC · speed error ė<sub>u</sub>",
            "Yaw SMC · yaw-rate error ė<sub>ψ</sub> = r − ψ̇<sub>cmd</sub>",
            "Surge sliding variable s<sub>u</sub> (shaded: |s<sub>u</sub>| ≤ "
            "φ<sub>u</sub>)",
            "Yaw sliding variable s<sub>r</sub> (shaded: |s<sub>r</sub>| ≤ "
            "φ<sub>r</sub>)",
        ),
    )
    for record, role, color, width in _runs(current, previous):
        data = window(record.result.data, t_window)
        label = run_label(record, role)
        fill = CURRENT_FILL if role == "Current" else PREVIOUS_FILL
        t0, t1 = data["t"].iloc[0], data["t"].iloc[-1]
        _band(fig, t0, t1, record.config.surge.phi_u, fill, 4, 1)
        _band(fig, t0, t1, record.config.yaw.phi_r, fill, 4, 2)
        for i, (name, row, col) in enumerate((
            ("e_p", 1, 1), ("e_u", 2, 1), ("e_psi", 2, 2),
            ("e_u_dot", 3, 1), ("e_psi_dot", 3, 2), ("s_u", 4, 1),
            ("s_r", 4, 2),
        )):
            _series(fig, data, name, label, color, width, row, col, role,
                    showlegend=i == 0)
    for row, col, unit in ((1, 1, "[m]"), (2, 1, "[m]"), (2, 2, "[rad]"),
                           (3, 1, "[m/s]"), (3, 2, "[rad/s]"), (4, 1, "[m/s]"),
                           (4, 2, "[rad/s]")):
        fig.update_yaxes(title_text=unit, row=row, col=col)
    fig.update_xaxes(title_text="t [s]", row=4, col=1)
    fig.update_xaxes(title_text="t [s]", row=4, col=2)
    return _layout(fig, 980, legend_rows=1)


# ---------------------------------------------------------------------------
# 2b) Vessel states.
# ---------------------------------------------------------------------------
def _break_wraps(values, threshold=180.0):
    """NaN where a wrapped angle jumps, to avoid vertical lines."""
    values = np.asarray(values, dtype=float).copy()
    values[1:][np.abs(np.diff(values)) > threshold] = np.nan
    return values


def vessel_states_figure(current, previous=None, t_window=None):
    """u, v, r, psi of both runs; dotted: commands of the current run."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("Surge velocity u (dotted: u<sub>cmd</sub>)",
                        "Sway velocity v",
                        "Yaw rate r (dotted: r<sub>cmd</sub>)",
                        "Heading ψ, wrapped to ±180° (dotted: ψ<sub>cmd</sub>)"),
    )
    for record, role, color, width in _runs(current, previous):
        data = window(record.result.data, t_window)
        label = run_label(record, role)
        for i, (name, row) in enumerate((("u", 1), ("v", 2), ("r", 3))):
            _series(fig, data, name, label, color, width, row, 1, role,
                    showlegend=i == 0)
        psi = np.degrees(np.angle(np.exp(1j * data["psi"].to_numpy())))
        fig.add_trace(go.Scatter(
            x=data["t"], y=_break_wraps(psi), mode="lines", name=label,
            legendgroup=role, showlegend=False,
            line=dict(color=color, width=width)), row=4, col=1)
        if role == "Current":
            for i, (name, row) in enumerate((("u_cmd", 1), ("r_cmd", 3))):
                _series(fig, data, name, "Current: commands u<sub>cmd</sub>, "
                        "r<sub>cmd</sub>, ψ<sub>cmd</sub>", COMMANDED_COLOR,
                        1.2, row, 1, "cmd", showlegend=i == 0, dash="dot")
            psi_cmd = np.degrees(np.angle(np.exp(1j * data["psi_cmd"]
                                                 .to_numpy())))
            fig.add_trace(go.Scatter(
                x=data["t"], y=_break_wraps(psi_cmd), mode="lines",
                legendgroup="cmd", showlegend=False, name="ψ<sub>cmd</sub>",
                line=dict(color=COMMANDED_COLOR, width=1.2, dash="dot")),
                row=4, col=1)
    for row, unit in ((1, "[m/s]"), (2, "[m/s]"), (3, "[rad/s]"), (4, "[°]")):
        fig.update_yaxes(title_text=unit, row=row, col=1)
    fig.update_yaxes(range=[-185, 185], tickvals=[-180, -90, 0, 90, 180],
                     row=4, col=1)
    fig.update_xaxes(title_text="t [s]", row=4, col=1)
    return _layout(fig, 900, legend_rows=2)


# ---------------------------------------------------------------------------
# 3) Control inputs.
# ---------------------------------------------------------------------------
def _limits_range(limits, pad=0.12):
    lo, hi = limits
    span = hi - lo
    return [lo - pad * span, hi + pad * span]


def controls_figure(current, previous=None, show_commanded=True,
                    t_window=None):
    """Delivered T_u, T_r and the switching functions.

    The force axes are fixed to the thruster limits (plus a margin): the
    delivered forces can never leave them, and every run gets the same scale.
    """
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.12,
        horizontal_spacing=0.09, row_heights=[0.62, 0.38],
        subplot_titles=(
            "Surge force T<sub>u</sub> (red: limits with saturated thrusters)",
            "Yaw moment T<sub>r</sub> (red: limits with saturated thrusters)",
            "Switching function sat(s<sub>u</sub>/φ<sub>u</sub>)",
            "Switching function sat(s<sub>r</sub>/φ<sub>r</sub>)",
        ),
    )
    for record, role, color, width in _runs(current, previous):
        data = window(record.result.data, t_window)
        label = run_label(record, role)
        if role == "Current" and show_commanded:
            for i, (name, col) in enumerate((("Tu_cmd", 1), ("Tr_cmd", 2))):
                _series(fig, data, name,
                        "Current: commanded before thruster saturation "
                        "(clipped by the axes)",
                        COMMANDED_COLOR, 1.0, 1, col, "cmd",
                        showlegend=i == 0, dash="dot")
        for i, (name, row, col) in enumerate((
            ("Tu", 1, 1), ("Tr", 1, 2), ("sat_u", 2, 1), ("sat_r", 2, 2),
        )):
            _series(fig, data, name, label, color, width, row, col, role,
                    showlegend=i == 0)
    for col, limits in ((1, TU_LIMITS), (2, TR_LIMITS)):
        for value in limits:
            fig.add_hline(y=value, line=dict(color=LIMIT_COLOR, dash="dot",
                                             width=1), row=1, col=col)
        fig.update_yaxes(range=_limits_range(limits), row=1, col=col)
    fig.update_yaxes(title_text="[N]", row=1, col=1)
    fig.update_yaxes(title_text="[N m]", row=1, col=2)
    fig.update_yaxes(title_text="[-]", range=[-1.15, 1.15], row=2, col=1)
    fig.update_yaxes(title_text="[-]", range=[-1.15, 1.15], row=2, col=2)
    fig.update_xaxes(title_text="t [s]", row=2, col=1)
    fig.update_xaxes(title_text="t [s]", row=2, col=2)
    return _layout(fig, 700, legend_rows=1 if not show_commanded else 2)


def thrusters_figure(current, t_window=None):
    """Thruster forces of the current run (F1 = F2 left, F3 = F4 right)."""
    fig = go.Figure()
    data = window(current.result.data, t_window)
    for name, label, color in (
        ("F1", "Left thrusters F<sub>1</sub> = F<sub>2</sub>", CURRENT_COLOR),
        ("F3", "Right thrusters F<sub>3</sub> = F<sub>4</sub>", "#ff7f0e"),
    ):
        t, y = envelope(data["t"], data[name])
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=label,
                                 line=dict(color=color, width=1.5)))
    for value in (-THRUSTER_REV_MAX, THRUSTER_FWD_MAX):
        fig.add_hline(y=value, line=dict(color=LIMIT_COLOR, dash="dot", width=1))
    fig.update_xaxes(title_text="t [s]")
    fig.update_yaxes(title_text="[N]",
                     range=_limits_range((-THRUSTER_REV_MAX, THRUSTER_FWD_MAX)))
    fig.update_layout(height=320, margin=dict(l=70, r=20, t=40, b=50),
                      legend=dict(orientation="h", y=1.12, x=0),
                      hovermode="x unified")
    return fig


# ---------------------------------------------------------------------------
# 4) Phase plane with the sliding surface and the boundary layer.
# ---------------------------------------------------------------------------
PHASE = {
    "surge": dict(e="e_u", e_dot="e_u_dot", s="s_u",
                  k=lambda c: c.surge.k_pos, phi=lambda c: c.surge.phi_u,
                  x_title="e<sub>u</sub> [m]", y_title="ė<sub>u</sub> [m/s]",
                  surface="s<sub>u</sub> = ė<sub>u</sub> + k<sub>pos</sub> "
                          "e<sub>u</sub>",
                  k_html="k<sub>pos</sub>", phi_html="φ<sub>u</sub>"),
    "yaw": dict(e="e_psi", e_dot="e_psi_dot", s="s_r",
                k=lambda c: c.yaw.k_psi, phi=lambda c: c.yaw.phi_r,
                x_title="e<sub>ψ</sub> [rad]", y_title="ė<sub>ψ</sub> [rad/s]",
                surface="s<sub>r</sub> = ė<sub>ψ</sub> + k<sub>ψ</sub> "
                        "e<sub>ψ</sub>",
                k_html="k<sub>ψ</sub>", phi_html="φ<sub>r</sub>"),
}


def _padded(lo, hi, minimum_half_width, pad=0.08):
    if not (math.isfinite(lo) and math.isfinite(hi)):
        lo, hi = -1.0, 1.0
    center, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    half = max(half * (1.0 + pad), minimum_half_width)
    return center - half, center + half


def phase_figure(current, previous=None, channel="surge", t_window=None):
    """ė vs e with the line s = 0 and the shaded band |s| <= φ."""
    spec = PHASE[channel]
    runs = _runs(current, previous)
    windows = {id(r): window(r.result.data, t_window) for r, *_ in runs}

    # Axis ranges from the data of both runs; the band around the origin is
    # always kept visible.
    xs = np.concatenate([windows[id(r)][spec["e"]].to_numpy() for r, *_ in runs])
    ys = np.concatenate([windows[id(r)][spec["e_dot"]].to_numpy()
                         for r, *_ in runs])
    phi_max = max(spec["phi"](r.config) for r, *_ in runs)
    k_max = max(spec["k"](r.config) for r, *_ in runs)
    x_lo, x_hi = _padded(min(xs.min(), 0.0), max(xs.max(), 0.0),
                         0.6 * phi_max / max(k_max, 1e-9))
    y_lo, y_hi = _padded(min(ys.min(), -phi_max), max(ys.max(), phi_max),
                         1.2 * phi_max)

    fig = go.Figure()
    for record, role, _color, _width in runs:
        k, phi = spec["k"](record.config), spec["phi"](record.config)
        current = role == "Current"
        xb = np.array([x_lo, x_hi])
        fig.add_trace(go.Scatter(
            x=[xb[0], xb[1], xb[1], xb[0], xb[0]],
            y=[-k * xb[0] - phi, -k * xb[1] - phi, -k * xb[1] + phi,
               -k * xb[0] + phi, -k * xb[0] - phi],
            mode="lines", fill="toself",
            fillcolor=CURRENT_FILL if current else PREVIOUS_FILL,
            line=dict(width=0.8 if not current else 0,
                      color=PREVIOUS_EDGE, dash="dot"),
            hoverinfo="skip", legendgroup=f"{role}-surface",
            name=(f"{role}: boundary layer |s| ≤ {spec['phi_html']} = "
                  f"{_v(phi)}")))
        fig.add_trace(go.Scatter(
            x=xb, y=-k * xb, mode="lines", hoverinfo="skip",
            legendgroup=f"{role}-surface",
            name=f"{role}: surface s = 0, slope −{spec['k_html']} = −{_v(k)}",
            line=dict(color=CURRENT_COLOR if current else PREVIOUS_EDGE,
                      dash="dash", width=1.3)))
    for record, role, color, width in runs:
        data = windows[id(record)]
        pts = stride(data[[spec["e"], spec["e_dot"], "t"]].to_numpy())
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="lines",
            name=run_label(record, role, channel), legendgroup=role,
            customdata=pts[:, 2],
            hovertemplate="t = %{customdata:.2f} s<br>e = %{x:.4g}<br>"
                          "ė = %{y:.4g}<extra></extra>",
            line=dict(color=color, width=width if role == "Current" else 4.0)))
        marker_color = color if role == "Current" else PREVIOUS_EDGE
        fig.add_trace(go.Scatter(
            x=pts[:1, 0], y=pts[:1, 1], mode="markers", legendgroup=role,
            name=f"start (t = {pts[0, 2]:.3g} s)", showlegend=False,
            hovertemplate=f"{role} start<extra></extra>",
            marker=dict(color=marker_color, size=10, symbol="circle",
                        line=dict(color="white", width=1))))
        fig.add_trace(go.Scatter(
            x=pts[-1:, 0], y=pts[-1:, 1], mode="markers", legendgroup=role,
            name="end", showlegend=False,
            hovertemplate=f"{role} end<extra></extra>",
            marker=dict(color=marker_color, size=11, symbol="x")))
    fig.update_xaxes(title_text=spec["x_title"], range=[x_lo, x_hi],
                     zeroline=True, zerolinecolor="#999")
    fig.update_yaxes(title_text=spec["y_title"], range=[y_lo, y_hi],
                     zeroline=True, zerolinecolor="#999")
    title = "Surge SMC" if channel == "surge" else "Yaw SMC"
    fig.update_layout(
        title=dict(text=f"{title}: {spec['surface']}  (● start, ✕ end)",
                   x=0, font=dict(size=15)),
        height=760, hovermode="closest",
        margin=dict(l=70, r=20, t=50, b=60),
        legend=dict(orientation="h", yref="container", y=0.0,
                    yanchor="bottom", x=0, font=dict(size=11)),
    )
    # Room for the legend below the axes.
    fig.update_yaxes(domain=[0.0, 1.0])
    fig.update_layout(margin=dict(b=230))
    return fig


# ---------------------------------------------------------------------------
# 5) Disturbances.
# ---------------------------------------------------------------------------
def disturbance_figure(current, previous=None, t_window=None):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=("Surge disturbance δT<sub>u</sub> (matched, surge SMC)",
                        "Sway disturbance δT<sub>v</sub> (not actuated)",
                        "Yaw disturbance δT<sub>r</sub> (matched, yaw SMC)"),
    )
    for record, role, color, width in _runs(current, previous):
        data = window(record.result.data, t_window)
        label = run_label(record, role)
        for i, name in enumerate(("dTu", "dTv", "dTr")):
            _series(fig, data, name, label, color, width, i + 1, 1, role,
                    showlegend=i == 0)
    for row, unit in ((1, "[N]"), (2, "[N]"), (3, "[N m]")):
        fig.update_yaxes(title_text=unit, row=row, col=1)
    fig.update_xaxes(title_text="t [s]", row=3, col=1)
    return _layout(fig, 640, legend_rows=1)


# ---------------------------------------------------------------------------
# 6) Performance indices.
# ---------------------------------------------------------------------------
def indices_figure(current, previous=None, keys=met.KEY_INDICES, cols=4):
    """Small multiples: one bar per run for each index (own y scale)."""
    rows = int(math.ceil(len(keys) / cols))
    titles = []
    for key in keys:
        _group, _sym, label, unit, better, _ = met.INDEX_INFO[key]
        arrow = {"lower": " ↓", "higher": " ↑"}.get(better, "")
        titles.append(f"{label} [{unit}]{arrow}")
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        vertical_spacing=0.45 / rows, horizontal_spacing=0.07)
    for i, key in enumerate(keys):
        row, col = divmod(i, cols)
        names, values, colors, texts, edges = [], [], [], [], []
        for record, role, _color, _w in _runs(current, previous):
            value = record.result.metrics.get(key, float("nan"))
            names.append(f"{role} #{record.number}")
            finite = math.isfinite(value)
            values.append(value if finite else 0.0)
            texts.append(_v(value) if finite else "never")
            colors.append(CURRENT_COLOR if role == "Current" else PREVIOUS_COLOR)
            edges.append(CURRENT_COLOR if role == "Current" else PREVIOUS_EDGE)
        fig.add_trace(go.Bar(
            x=names, y=values, marker_color=colors,
            marker_line=dict(color=edges, width=1), text=texts,
            textposition="outside", cliponaxis=False, showlegend=False,
            hovertemplate="%{x}: %{text}<extra></extra>"),
            row=row + 1, col=col + 1)
        top = max(abs(v) for v in values)
        fig.update_yaxes(range=[0, top * 1.3 if top > 0 else 1.0],
                         row=row + 1, col=col + 1)
    fig.update_annotations(font_size=11)
    fig.update_layout(height=230 * rows, margin=dict(l=40, r=20, t=40, b=30),
                      bargap=0.35)
    return fig
