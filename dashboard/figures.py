"""Plotly figures for dashboard results (no Streamlit imports).

Figures only read ``SimulationResult.data``; nothing is recomputed here
except the closed reference path drawn in the XY plot (one period of
``controllers.figure_eight_reference``).  Long signals are decimated for
display only (metrics always use the full-resolution data).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard import PROJECT_ROOT  # noqa: F401

import controllers
from usv_model import THRUSTER_FWD_MAX, THRUSTER_REV_MAX, TR_LIMITS, TU_LIMITS

MAX_PLOT_POINTS = 4000
RUN_COLORS = ("#1f77b4", "#d62728")   # run A / run B in comparisons
REF_COLOR = "#444444"
PREVIOUS_COLOR = "#c8c8c8"            # previous run overlaid in the XY plot
THRUSTER_COLORS = ("#1f77b4", "#17becf", "#d62728", "#ff7f0e")


def _stride(n, max_points=MAX_PLOT_POINTS):
    return max(1, int(np.ceil(n / max_points)))


def _col(result, name):
    data = result.data
    step = _stride(len(data))
    values = data[name].to_numpy()[::step]
    if step > 1 and (len(data) - 1) % step:
        values = np.append(values, data[name].to_numpy()[-1])
    return values


def _break_wraps(angle_deg, threshold=180.0):
    """Insert NaN where a wrapped angle jumps, to avoid vertical lines."""
    angle_deg = np.asarray(angle_deg, dtype=float).copy()
    jumps = np.abs(np.diff(angle_deg)) > threshold
    angle_deg[1:][jumps] = np.nan
    return angle_deg


def _layout(fig, height, legend="right"):
    """Common layout; multi-panel figures keep the legend on the right so
    that it never covers the subplot titles."""
    if legend == "top":
        legend_style = dict(orientation="h", yanchor="bottom", y=1.02, x=0)
        margin = dict(l=60, r=20, t=40, b=40)
    else:
        legend_style = dict(orientation="v", yanchor="top", y=1.0, x=1.01,
                            xanchor="left")
        margin = dict(l=60, r=10, t=40, b=40)
    fig.update_layout(height=height, margin=margin, legend=legend_style,
                      hovermode="x unified")
    return fig


def reference_path(reference, samples=800):
    t = np.linspace(0.0, reference.period_s, samples)
    x, y, *_ = controllers.figure_eight_reference(
        t, reference.amplitude_m, reference.period_s
    )
    return x, y


# ---------------------------------------------------------------------------
# Single-run figures.
# ---------------------------------------------------------------------------
def xy_trajectory(result, previous=None):
    """XY plot of ``result``; optionally overlays the ``previous`` run.

    ``previous`` is a run record (``.result``, ``.title``) drawn underneath in
    light gray, in the same axes, so the effect of a parameter change can be
    seen directly. Drawing order: previous run, reference, current run.
    """
    config = result.config
    x, y = _col(result, "x"), _col(result, "y")
    fig = go.Figure()
    if previous is not None:
        fig.add_trace(go.Scatter(
            x=_col(previous.result, "x"), y=_col(previous.result, "y"),
            mode="lines", name=f"Previous: {previous.title}",
            line=dict(color=PREVIOUS_COLOR, width=3)))
        previous_reference = previous.result.config.reference
        if previous_reference != config.reference:
            x_prev_path, y_prev_path = reference_path(previous_reference)
            fig.add_trace(go.Scatter(
                x=x_prev_path, y=y_prev_path, mode="lines",
                name="Previous reference path",
                line=dict(color=PREVIOUS_COLOR, dash="dot", width=1.5)))
    x_path, y_path = reference_path(config.reference)
    fig.add_trace(go.Scatter(x=x_path, y=y_path, mode="lines",
                             name="Reference path",
                             line=dict(color=REF_COLOR, dash="dash", width=1.5)))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="USV hull centre <b>p</b>",
                             line=dict(color=RUN_COLORS[0], width=2)))
    fig.add_trace(go.Scatter(x=_col(result, "x_s"), y=_col(result, "y_s"),
                             mode="lines", name="Lookahead point <b>p</b><sub>s</sub>",
                             line=dict(color="#2ca02c", width=1),
                             visible="legendonly"))
    fig.add_trace(go.Scatter(x=_col(result, "x_s_ref"), y=_col(result, "y_s_ref"),
                             mode="lines", name="Lookahead reference <b>p</b><sub>s,ref</sub>",
                             line=dict(color="#9467bd", width=1, dash="dot"),
                             visible="legendonly"))
    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers", name="Start",
                             marker=dict(symbol="circle", size=10,
                                         color="#2ca02c")))
    fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers", name="End",
                             marker=dict(symbol="x", size=11, color="#d62728")))
    fig.update_xaxes(title_text="x [m]")
    fig.update_yaxes(title_text="y [m]", scaleanchor="x", scaleratio=1)
    _layout(fig, 560, legend="top")
    fig.update_layout(hovermode="closest")
    return fig


def tracking_errors(result):
    t = _col(result, "t")
    L = result.config.evaluation_lookahead_L
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=(
            f"Lookahead-point position error e<sub>s</sub> (L = {L:g} m)",
            "Hull-centre position error e<sub>c</sub>",
            "Lookahead-point velocity error e<sub>v</sub>",
            "Heading error e<sub>ψ</sub> = ψ<sub>ref</sub> − ψ (path tangent)",
        ),
    )
    for name, label, dash in (("e_s", "e<sub>s</sub>", None),
                              ("e_s_x", "e<sub>s,x</sub>", "dot"),
                              ("e_s_y", "e<sub>s,y</sub>", "dot")):
        fig.add_trace(go.Scatter(x=t, y=_col(result, name), name=label,
                                 line=dict(dash=dash)), row=1, col=1)
    for name, label, dash in (("e_c", "e<sub>c</sub>", None),
                              ("e_c_x", "e<sub>c,x</sub>", "dot"),
                              ("e_c_y", "e<sub>c,y</sub>", "dot")):
        fig.add_trace(go.Scatter(x=t, y=_col(result, name), name=label,
                                 line=dict(dash=dash)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=_col(result, "e_s_dot"), name="e<sub>v</sub>"),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.degrees(_col(result, "e_psi")),
                             name="e<sub>ψ</sub>"), row=4, col=1)
    fig.update_yaxes(title_text="[m]", row=1, col=1)
    fig.update_yaxes(title_text="[m]", row=2, col=1)
    fig.update_yaxes(title_text="[m/s]", row=3, col=1)
    fig.update_yaxes(title_text="[°]", row=4, col=1)
    fig.update_xaxes(title_text="t [s]", row=4, col=1)
    return _layout(fig, 900)


def vessel_states(result):
    t = _col(result, "t")
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("Surge velocity u", "Sway velocity v", "Yaw rate r",
                        "Heading ψ (wrapped to ±180°)"),
    )
    fig.add_trace(go.Scatter(x=t, y=_col(result, "u"), name="u"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=_col(result, "u_ref"), name="‖ṗ<sub>ref</sub>‖ (reference speed)",
                             line=dict(color=REF_COLOR, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=_col(result, "v"), name="v"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.degrees(_col(result, "r")), name="r"),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.degrees(_col(result, "r_ref")),
                             name="r<sub>ref</sub>",
                             line=dict(color=REF_COLOR, dash="dash")), row=3, col=1)
    psi = np.degrees(np.angle(np.exp(1j * _col(result, "psi"))))
    psi_ref = np.degrees(_col(result, "psi_ref"))
    fig.add_trace(go.Scatter(x=t, y=_break_wraps(psi), name="ψ"), row=4, col=1)
    fig.add_trace(go.Scatter(x=t, y=_break_wraps(psi_ref), name="ψ<sub>ref</sub> (path tangent)",
                             line=dict(color=REF_COLOR, dash="dash")), row=4, col=1)
    for row, unit in enumerate(("[m/s]", "[m/s]", "[°/s]", "[°]"), start=1):
        fig.update_yaxes(title_text=unit, row=row, col=1)
    fig.update_yaxes(range=[-185, 185], tickvals=[-180, -90, 0, 90, 180],
                     row=4, col=1)
    fig.update_xaxes(title_text="t [s]", row=4, col=1)
    return _layout(fig, 900)


def _limit_lines(fig, limits, row, label):
    for i, value in enumerate(limits):
        annotation = (
            dict(annotation_text=label, annotation_position="top left")
            if i == 1 else {}
        )
        fig.add_hline(y=value, line=dict(color="black", dash="dot", width=1),
                      row=row, col=1, **annotation)


def control_signals(result):
    t = _col(result, "t")
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("Surge force T<sub>u</sub>: commanded vs delivered",
                        "Yaw moment T<sub>r</sub>: commanded vs delivered",
                        "Thruster forces F<sub>1</sub>…F<sub>4</sub> (delivered)",
                        "Thruster PWM commands"),
    )
    fig.add_trace(go.Scatter(x=t, y=_col(result, "tau_u_cmd"), name="T<sub>u</sub> commanded",
                             line=dict(color="#ff7f0e")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=_col(result, "tau_u_real"), name="T<sub>u</sub> delivered",
                             line=dict(color="#1f77b4")), row=1, col=1)
    _limit_lines(fig, TU_LIMITS, 1, "limits")
    fig.add_trace(go.Scatter(x=t, y=_col(result, "tau_r_cmd"), name="T<sub>r</sub> commanded",
                             line=dict(color="#ff7f0e", dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=_col(result, "tau_r_real"), name="T<sub>r</sub> delivered",
                             line=dict(color="#1f77b4", dash="dot")), row=2, col=1)
    _limit_lines(fig, TR_LIMITS, 2, "limits")
    for i, color in enumerate(THRUSTER_COLORS, start=1):
        side = "left" if i <= 2 else "right"
        fig.add_trace(go.Scatter(x=t, y=_col(result, f"F{i}"), name=f"F<sub>{i}</sub> {side}",
                                 line=dict(color=color)), row=3, col=1)
        fig.add_trace(go.Scatter(x=t, y=_col(result, f"PWM{i}"), name=f"PWM<sub>{i}</sub>",
                                 line=dict(color=color), showlegend=False),
                      row=4, col=1)
    _limit_lines(fig, (-THRUSTER_REV_MAX, THRUSTER_FWD_MAX), 3, "limits")
    fig.update_yaxes(title_text="[N]", row=1, col=1)
    fig.update_yaxes(title_text="[N m]", row=2, col=1)
    fig.update_yaxes(title_text="[N]", row=3, col=1)
    fig.update_yaxes(title_text="[µs]", row=4, col=1)
    fig.update_xaxes(title_text="t [s]", row=4, col=1)
    return _layout(fig, 950)


def disturbance_signals(result):
    t = _col(result, "t")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Disturbance forces δT<sub>u</sub>, "
                                        "δT<sub>v</sub> (body frame)",
                                        "Disturbance yaw moment δT<sub>r</sub>"))
    fig.add_trace(go.Scatter(x=t, y=_col(result, "dTu"), name="δT<sub>u</sub> (surge)"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=_col(result, "dTv"), name="δT<sub>v</sub> (sway)"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=_col(result, "dTr"), name="δT<sub>r</sub> (yaw)"),
                  row=2, col=1)
    fig.update_yaxes(title_text="[N]", row=1, col=1)
    fig.update_yaxes(title_text="[N m]", row=2, col=1)
    fig.update_xaxes(title_text="t [s]", row=2, col=1)
    return _layout(fig, 500)


# ---------------------------------------------------------------------------
# Two-run comparison figures.
# ---------------------------------------------------------------------------
def compare_xy(record_a, record_b):
    fig = go.Figure()
    ref_a, ref_b = record_a.config.reference, record_b.config.reference
    references = [(ref_a, "Reference path")]
    if ref_b != ref_a:
        references = [(ref_a, "Reference A"), (ref_b, "Reference B")]
    for i, (reference, name) in enumerate(references):
        x_path, y_path = reference_path(reference)
        fig.add_trace(go.Scatter(
            x=x_path, y=y_path, mode="lines", name=name,
            line=dict(color=REF_COLOR if i == 0 else "#999999", dash="dash",
                      width=1.5)))
    for record, color, tag in ((record_a, RUN_COLORS[0], "A"),
                               (record_b, RUN_COLORS[1], "B")):
        fig.add_trace(go.Scatter(
            x=_col(record.result, "x"), y=_col(record.result, "y"), mode="lines",
            name=f"{tag}: {record.title}", line=dict(color=color, width=2)))
    fig.update_xaxes(title_text="x [m]")
    fig.update_yaxes(title_text="y [m]", scaleanchor="x", scaleratio=1)
    _layout(fig, 560, legend="top")
    fig.update_layout(hovermode="closest")
    return fig


def compare_time_series(record_a, record_b):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("Lookahead-point position error e<sub>s</sub>",
                        "Hull-centre position error e<sub>c</sub>",
                        "Delivered surge force T<sub>u</sub>",
                        "Delivered yaw moment T<sub>r</sub>"),
    )
    for record, color, tag in ((record_a, RUN_COLORS[0], "A"),
                               (record_b, RUN_COLORS[1], "B")):
        t = _col(record.result, "t")
        for row, name in enumerate(("e_s", "e_c", "tau_u_real", "tau_r_real"),
                                   start=1):
            fig.add_trace(go.Scatter(
                x=t, y=_col(record.result, name), name=f"{tag}: {record.title}",
                line=dict(color=color), legendgroup=tag, showlegend=row == 1),
                row=row, col=1)
    for row, unit in enumerate(("[m]", "[m]", "[N]", "[N m]"), start=1):
        fig.update_yaxes(title_text=unit, row=row, col=1)
    fig.update_xaxes(title_text="t [s]", row=4, col=1)
    return _layout(fig, 900)
