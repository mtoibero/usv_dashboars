"""Streamlit dashboard: sliding mode control of the IAcquaBot USV.

Launch from the project root with either of::

    streamlit run dashboard02_smc/app.py
    python -m dashboard02_smc

This file only contains the user interface.  Configuration, simulation,
indices, figures and LaTeX texts live in GUI-independent modules.

All session-state keys start with ``smc2_`` so that this page can later be
aggregated with other dashboards in one multipage app without collisions.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import math  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard02_smc import config as cfg  # noqa: E402
from dashboard02_smc import engine, figures  # noqa: E402
from dashboard02_smc import latex as tex  # noqa: E402
from dashboard02_smc import metrics as met  # noqa: E402
from dashboard02_smc import tuning  # noqa: E402
from dashboard02_smc.session import RunHistory, staleness  # noqa: E402

from usv_model import (  # noqa: E402
    PARAMS, THRUSTER_ARM, THRUSTER_FWD_MAX, THRUSTER_REV_MAX,
)

PREFIX = "smc2_"
MAX_WALL_TIME_S = 600.0
MAX_CACHED_FIGURES = 40
SIDEBAR_WIDTH_PX = 430


def K(name):
    """Session-state key of this dashboard."""
    return PREFIX + name


st.set_page_config(
    page_title="USV sliding mode control lab",
    page_icon=":material/sailing:",
    layout="wide",
    initial_sidebar_state=SIDEBAR_WIDTH_PX,
)


# ---------------------------------------------------------------------------
# Widget specifications (slider range, step, number format, hard minimum).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Spec:
    lo: float
    hi: float
    step: float
    fmt: str = "%g"
    minimum: float | None = None   # number-box lower bound


SPECS = {
    "ref_amplitude_m": Spec(2.0, 40.0, 0.5, "%.2f", 0.0),
    "ref_frequency_hz": Spec(0.002, 0.03, 0.0001, "%.5f", 0.0),
    "ref_peak_speed_mps": Spec(0.1, 2.0, 0.01, "%.3f", 0.0),
    "ref_laps": Spec(0.25, 6.0, 0.25, "%.2f", 0.0),
    "sim_t_final_s": Spec(10.0, 1200.0, 10.0, "%.1f", 0.0),
    "sim_sample_rate_hz": Spec(5.0, 200.0, 1.0, "%.0f", 0.0),
    "sim_output_dt_s": Spec(0.005, 0.1, 0.005, "%.3f", 0.0),
    "sim_lookahead_L_m": Spec(0.0, 3.0, 0.05, "%.2f", 0.0),
    "init_dx_m": Spec(-10.0, 10.0, 0.5, "%.1f"),
    "init_dy_m": Spec(-10.0, 10.0, 0.5, "%.1f"),
    "init_dpsi_deg": Spec(-180.0, 180.0, 5.0, "%.0f"),
    "init_du_mps": Spec(-1.0, 1.0, 0.05, "%.2f"),
    "surge_lambda_u": Spec(0.0, 10.0, 0.05, "%.2f", 0.0),
    "surge_k_s_u": Spec(0.0, 3.0, 0.01, "%.3f", 0.0),
    "surge_phi_u": Spec(0.001, 0.5, 0.001, "%.3f", 0.0),
    "surge_k_pos": Spec(0.0, 2.0, 0.01, "%.2f", 0.0),
    "yaw_lambda_r": Spec(0.0, 10.0, 0.05, "%.2f", 0.0),
    "yaw_k_s_r": Spec(0.0, 5.0, 0.01, "%.3f", 0.0),
    "yaw_phi_r": Spec(0.001, 0.5, 0.001, "%.3f", 0.0),
    "yaw_k_psi": Spec(0.0, 10.0, 0.05, "%.2f", 0.0),
    "dist_const_dTu": Spec(-50.0, 50.0, 1.0, "%.1f"),
    "dist_const_dTv": Spec(-50.0, 50.0, 1.0, "%.1f"),
    "dist_const_dTr": Spec(-40.0, 40.0, 1.0, "%.1f"),
    "dist_dir_magnitude_N": Spec(0.0, 60.0, 1.0, "%.1f", 0.0),
    "dist_dir_direction_deg": Spec(-180.0, 180.0, 5.0, "%.0f"),
    "dist_comb_dir_magnitude_N": Spec(0.0, 60.0, 1.0, "%.1f", 0.0),
    "dist_comb_dir_direction_deg": Spec(-180.0, 180.0, 5.0, "%.0f"),
}
for _axis in cfg.AXES:
    for _prefix in ("dist_sin", "dist_comb"):
        SPECS[f"{_prefix}_amp_{_axis}"] = Spec(0.0, 40.0, 0.5, "%.2f")
        SPECS[f"{_prefix}_freq_{_axis}"] = Spec(0.0, 0.5, 0.005, "%.3f", 0.0)
        SPECS[f"{_prefix}_phase_{_axis}"] = Spec(-3.15, 3.15, 0.1, "%.2f")
    SPECS[f"dist_sin_bias_{_axis}"] = Spec(-40.0, 40.0, 0.5, "%.2f")

REF_MODES = {
    "freq_speed": r"Frequency $f$ and speed $v_{max}$",
    "amp_freq": r"Amplitude $A$ and frequency $f$",
    "amp_speed": r"Amplitude $A$ and speed $v_{max}$",
}
DURATION_MODES = {
    "laps": r"Number of laps $\times$ period ($N_{laps}\,T$)",
    "manual": "Custom duration",
}
INITIAL_MODES = {
    "reference": "On the reference at $t = 0$ (simulator default)",
    "offset": "Offset from the reference",
}
SIM_MODE_LABELS = {
    cfg.DIGITAL: "Digital controller (sampled, zero-order hold)",
    cfg.CONTINUOUS: "Continuous-time controller (usv_model.simulate)",
}
AXIS_TITLES = (("u", r"Surge $\delta T_u$", "N"), ("v", r"Sway $\delta T_v$", "N"),
               ("r", r"Yaw $\delta T_r$", "N m"))


# ---------------------------------------------------------------------------
# Session state.
# ---------------------------------------------------------------------------
def init_session_state():
    ss = st.session_state
    for name, value in cfg.default_form().items():
        key = K(name)
        if key not in ss:
            ss[key] = value
        else:
            # Re-assigning keeps the values of widgets that are hidden in
            # this script run (Streamlit drops state of unrendered widgets).
            ss[key] = ss[key]
    for name, value in (("use_sliders", False), ("run_label", ""),
                        ("preset", "defaults")):
        if K(name) not in ss:
            ss[K(name)] = value
        else:
            ss[K(name)] = ss[K(name)]
    ss.setdefault(K("history"), RunHistory())
    ss.setdefault(K("pending_run"), None)
    ss.setdefault(K("last_failure"), None)


def current_form():
    return {name: st.session_state[K(name)] for name in cfg.FORM_KEYS}


def apply_form(values):
    for name, value in values.items():
        if name in cfg.FORM_KEYS:
            st.session_state[K(name)] = value


# ---------------------------------------------------------------------------
# Callbacks (run before the script re-executes).
# ---------------------------------------------------------------------------
def on_run_clicked():
    st.session_state[K("pending_run")] = {
        "label": st.session_state[K("run_label")]}
    st.session_state[K("run_label")] = ""


def on_reset_defaults():
    apply_form(cfg.preset_form("defaults", base=current_form()))


def on_apply_preset():
    apply_form(cfg.preset_form(st.session_state[K("preset")],
                               base=current_form()))
    sync_reference_fields()


def on_use_suggested_gain(channel, value):
    """Set k_s of one SMC to the suggested value (never runs a simulation)."""
    name = "surge_k_s_u" if channel == "surge" else "yaw_k_s_r"
    st.session_state[K(name)] = float(value)


def on_load_run(number):
    record = st.session_state[K("history")].get(number)
    apply_form(cfg.config_to_form(record.config, base=current_form()))


def sync_reference_fields():
    """Keep amplitude, frequency and peak speed mutually consistent.

    The two inputs of the selected mode define the figure eight; the third
    field is updated so that switching the mode keeps the same trajectory.
    """
    amplitude, frequency, speed = cfg.resolve_reference(current_form())
    if all(math.isfinite(v) and v > 0.0 for v in (amplitude, frequency, speed)):
        ss = st.session_state
        ss[K("ref_amplitude_m")] = amplitude
        ss[K("ref_frequency_hz")] = frequency
        ss[K("ref_peak_speed_mps")] = speed


def on_duration_mode_changed():
    """Keep the same duration when switching between laps and custom time."""
    ss = st.session_state
    frequency = cfg.resolve_reference(current_form())[1]
    if not (math.isfinite(frequency) and frequency > 0.0):
        return
    if ss[K("sim_duration_mode")] == "manual" and ss[K("ref_laps")] > 0.0:
        ss[K("sim_t_final_s")] = ss[K("ref_laps")] / frequency
    elif ss[K("sim_duration_mode")] == "laps" and ss[K("sim_t_final_s")] > 0.0:
        ss[K("ref_laps")] = ss[K("sim_t_final_s")] * frequency


# ---------------------------------------------------------------------------
# Sidebar widgets.
# ---------------------------------------------------------------------------
def param(label, name, help=None, on_change=None):
    """Slider or number box (user preference) bound to ``K(name)``."""
    spec = SPECS[name]
    key = K(name)
    if st.session_state[K("use_sliders")]:
        value = float(st.session_state[key])
        lo = min(spec.lo, value) if math.isfinite(value) else spec.lo
        hi = max(spec.hi, value) if math.isfinite(value) else spec.hi
        return st.slider(label, min_value=float(lo), max_value=float(hi),
                         step=spec.step, format=spec.fmt, key=key, help=help,
                         on_change=on_change)
    return st.number_input(label, step=spec.step, format=spec.fmt,
                           min_value=spec.minimum, key=key, help=help,
                           on_change=on_change)


def pair(left, right):
    """Two parameters side by side: (label, name, help) for surge | yaw."""
    c1, c2 = st.columns(2)
    with c1:
        param(*left)
    with c2:
        param(*right)


def equations(*lines):
    st.markdown(tex.aligned(*lines))


# -- 1 · Simulation settings ---------------------------------------------------
def sidebar_simulation(check):
    ss = st.session_state
    st.radio(r"Total simulation time $T_{sim}$", options=list(DURATION_MODES),
             format_func=DURATION_MODES.get, key=K("sim_duration_mode"),
             on_change=on_duration_mode_changed)
    if ss[K("sim_duration_mode")] == "manual":
        param(r"Total simulation time $T_{sim}$ [s]", "sim_t_final_s")
    elif check.config is not None:
        ref = check.config.reference
        st.caption(rf"$T_{{sim}} = N_{{laps}}\,T$ = {ref.laps:g} $\times$ "
                   f"{ref.period_s:.4g} s = **{ref.t_final_s:.4g} s** "
                   r"(set $N_{laps}$ in *Reference trajectory*)")
    st.radio("Controller implementation", options=list(SIM_MODE_LABELS),
             format_func=SIM_MODE_LABELS.get, key=K("sim_mode"))
    if ss[K("sim_mode")] == cfg.DIGITAL:
        param(r"Controller sample rate $f_s$ [Hz]", "sim_sample_rate_hz",
              help=r"The control law runs every $T_s = 1/f_s$ and its forces "
                   "are held constant in between (zero-order hold), as on "
                   "the vessel. The vessel model is still integrated with the "
                   "adaptive RK45 solver of `usv_model.simulate`.")
        st.caption(r"Chattering appears when $\Gamma_i = T_s(\lambda_i + "
                   r"k_{s,i}/\phi_i) > 1$ (see the controller hints).")
    else:
        param(r"Output sampling interval $\Delta t$ [s]", "sim_output_dt_s",
              help="Spacing of the stored samples. The adaptive integrator "
                   "evaluates the controller continuously (as in "
                   "`run_simulation.py`).")
        st.caption("An ideal continuous controller does not chatter, and thin "
                   "boundary layers make the integration slow.")
    st.radio("Initial state", options=list(INITIAL_MODES),
             format_func=INITIAL_MODES.get, key=K("sim_initial_mode"))
    if ss[K("sim_initial_mode")] == "offset":
        pair((r"$\Delta x_0$ [m]", "init_dx_m"), (r"$\Delta y_0$ [m]", "init_dy_m"))
        pair((r"$\Delta\psi_0$ [deg]", "init_dpsi_deg"),
             (r"$\Delta u_0$ [m/s]", "init_du_mps"))
        st.caption("Offsets from the reference state at $t = 0$; they make the "
                   "reaching phase visible in the phase plots.")
    elif check.config is not None:
        x0, y0, psi0, u0, _v0, _r0 = check.config.initial_state()
        st.caption(f"$x_0$ = {x0:.3g} m, $y_0$ = {y0:.3g} m, "
                   rf"$\psi_0$ = {math.degrees(psi0):.4g}°, "
                   f"$u_0$ = {u0:.4g} m/s, $v_0$ = $r_0$ = 0")


# -- 2 · Reference trajectory ------------------------------------------------
def sidebar_reference(check):
    ss = st.session_state
    equations(
        r"x_{ref}(t) &= A\sin(\omega t)",
        r"y_{ref}(t) &= A\sin(\omega t)\cos(\omega t)",
        r"\omega &= 2\pi f,\qquad v_{max} = \sqrt{2}\,A\,\omega",
    )
    st.radio("Define the figure eight by", options=list(REF_MODES),
             format_func=REF_MODES.get, key=K("ref_mode"))
    mode = ss[K("ref_mode")]
    if mode in ("amp_freq", "amp_speed"):
        param(r"Amplitude $A$ [m]", "ref_amplitude_m",
              help=r"Half-width of the figure eight along $x$ (along $y$ it "
                   r"is $A/2$).", on_change=sync_reference_fields)
    if mode in ("freq_speed", "amp_freq"):
        param(r"Frequency $f$ [Hz]", "ref_frequency_hz",
              help=r"Laps per second of the figure eight; period $T = 1/f$.",
              on_change=sync_reference_fields)
    if mode in ("freq_speed", "amp_speed"):
        param(r"Peak speed $v_{max}$ [m/s]", "ref_peak_speed_mps",
              help=r"Reference speed at the crossing point, "
                   r"$v_{max} = \sqrt{2}A\omega$ (the highest speed of the "
                   "lap).", on_change=sync_reference_fields)
    if ss[K("sim_duration_mode")] == "laps":
        param(r"Number of laps $N_{laps}$", "ref_laps",
              help=r"Simulated time $T_{sim} = N_{laps}\,T$.")
    if check.config is not None:
        ref = check.config.reference
        if ref.amplitude_m > 0 and ref.period_s > 0:
            v_mean = cfg.mean_speed(ref.amplitude_m, ref.period_s)
            st.caption(
                f"$A$ = **{ref.amplitude_m:.3g} m**, $f$ = "
                f"**{ref.frequency_hz:.4g} Hz** ($T$ = {ref.period_s:.4g} s), "
                f"$v_{{max}}$ = **{ref.peak_speed_mps:.3g} m/s**, "
                rf"$\bar v$ = {v_mean:.3g} m/s, laps simulated "
                f"{ref.laps:.3g}"
            )


# -- 3 · Controller ----------------------------------------------------------
def gain_hint(check, channel, where):
    """Switching-gain estimate for one SMC (see tuning.py), with a button
    that copies the suggested k_s into the controller settings."""
    if check.config is None:
        return
    hint = tuning.gain_hints(check.config)[channel]
    suggested = tuning.suggested_value(hint.k_suggested)
    block, _blank, *lines = tex.gain_hint_lines(hint, suggested)
    st.markdown(block)
    st.markdown("  \n".join(lines))
    suggest_button(hint, channel, where)


def suggest_button(hint, channel, where):
    suggested = tuning.suggested_value(hint.k_suggested)
    status, _icon = tex.gain_status(hint)
    wants_less = (not hint.chattering_ok and hint.window_exists
                  and suggested < hint.k_s)
    if (status != "robust" or wants_less) and suggested > 0.0:
        symbol = "k_{s,u}" if channel == "surge" else "k_{s,r}"
        st.button(f"Use suggested ${symbol}$ = {suggested:g}",
                  key=K(f"use_ks_{channel}_{where}"), icon=":material/tune:",
                  on_click=on_use_suggested_gain, args=(channel, suggested),
                  help="Copies the value into the controller settings. It "
                       "does not run a simulation.")


def switching_gain_summary(check):
    """One status line per SMC (details under *4 · Disturbance*)."""
    if check.config is None:
        return
    hints = tuning.gain_hints(check.config)
    for channel, name in (("surge", "Surge"), ("yaw", "Yaw")):
        hint = hints[channel]
        _status, icon = tex.gain_status(hint)
        i = hint.axis
        suggested = tuning.suggested_value(hint.k_suggested)
        text = (rf"{icon} {name}: $k_{{s,{i}}} = {tex.num(hint.k_s, 3)}$, "
                rf"min. ${tex.num(hint.k_min, 3)}$, suggested "
                rf"${tex.num(suggested, 3)}$")
        if not hint.chattering_ok:
            text += " · ⚠️ chatters"
        st.markdown(text)
        suggest_button(hint, channel, "controller")


def design_hints(check):
    if check.config is None:
        return
    lines = {"surge": [], "yaw": []}
    for item in cfg.design_checks(check.config):
        if item["kind"] == "sliding":
            continue
        icon = {True: "✅", False: "⚠️", None: "ℹ️"}[item["ok"]]
        lines[item["channel"]].append(f"{icon} ${item['latex']}$")
    # Static label: Streamlit identifies expanders by their label.
    with st.expander("Design hints (approximate theory)"):
        for channel, name in (("surge", "Surge SMC"), ("yaw", "Yaw SMC")):
            st.markdown(f"**{name}**")
            st.markdown("  \n".join(lines[channel]))
        st.caption(
            r"Steady $s$ inside the layer for a constant disturbance; discrete "
            r"gain $\Gamma$ of the sampled boundary layer; time constant of "
            "the error on the surface. They ignore thruster saturation and the "
            "remaining model approximations."
        )


def sidebar_controller(check):
    st.caption(r"Two sliding mode controllers of `smc_figure_eight`: surge "
               r"speed $u$ (force $T_u$) and yaw rate $r$ (moment $T_r$).")
    st.markdown("**Guidance (outer loop)**")
    st.caption(r"Lookahead point $\boldsymbol{p}_s = \boldsymbol{p} + "
               r"L\,[\cos\psi,\ \sin\psi]^\top$")
    equations(
        r"\boldsymbol{v}_{cmd} &= \dot{\boldsymbol{p}}_{s,ref} + k_{pos}"
        r"(\boldsymbol{p}_{s,ref} - \boldsymbol{p}_s)",
        r"u_{cmd} &= \lVert \boldsymbol{v}_{cmd} \rVert,\quad \psi_{cmd} = "
        r"\operatorname{atan2}(v_{cmd,y}, v_{cmd,x})",
        r"r_{cmd} &= \dot{\psi}_{cmd} + k_{\psi}\,(\psi_{cmd} - \psi)",
    )
    param(r"Lookahead distance $L$ [m]", "sim_lookahead_L_m",
          help="Distance of the controlled point ahead of the hull centre.")
    pair((r"$k_{pos}$ [1/s]", "surge_k_pos",
          r"Position-error gain: slope of the surface $s_u = \dot e_u + "
          r"k_{pos} e_u$."),
         (r"$k_{\psi}$ [1/s]", "yaw_k_psi",
          r"Heading-error gain: slope of the surface $s_r = \dot e_\psi + "
          r"k_\psi e_\psi$."))

    st.markdown("**Sliding surfaces and reaching law**")
    equations(
        r"s_u &= u - u_{cmd} = \dot e_u + k_{pos} e_u",
        r"s_r &= r - r_{cmd} = \dot e_\psi + k_\psi e_\psi",
        r"\dot{u}_{des} &= \dot{u}_{cmd} - \lambda_u s_u - k_{s,u}\,"
        r"\mathrm{sat}(s_u/\phi_u)",
        r"\dot{r}_{des} &= -\lambda_r s_r - k_{s,r}\,\mathrm{sat}(s_r/\phi_r)",
    )
    st.caption(r"with $\mathrm{sat}(x) = \min(1, \max(-1, x))$. Left column: "
               "surge, right column: yaw.")
    pair((r"$\lambda_u$ [1/s]", "surge_lambda_u",
          r"Proportional gain of the surge reaching law."),
         (r"$\lambda_r$ [1/s]", "yaw_lambda_r",
          r"Proportional gain of the yaw reaching law."))
    pair((r"$k_{s,u}$ [m/s$^2$]", "surge_k_s_u",
          r"Saturation (switching) gain of the surge SMC."),
         (r"$k_{s,r}$ [rad/s$^2$]", "yaw_k_s_r",
          r"Saturation (switching) gain of the yaw SMC."))
    pair((r"$\phi_u$ [m/s]", "surge_phi_u",
          r"Boundary-layer half-width of the surge SMC (> 0)."),
         (r"$\phi_r$ [rad/s]", "yaw_phi_r",
          r"Boundary-layer half-width of the yaw SMC (> 0)."))

    st.markdown("**Inversion of the identified dynamics**")
    equations(
        r"T_u &= z_4 u + z_5 |u| u + z_1 (\dot{u}_{des} - v r)",
        r"T_r &= z_8 r + z_9 |r| r + z_3\, \dot{r}_{des}",
    )
    st.caption(r"$T_v = 0$ (no lateral actuation). $T_u$, $T_r$ are then "
               "allocated to the four thrusters and saturated.")

    st.markdown("**Switching gains vs the disturbance**")
    st.caption(r"Sliding condition $k_{s,i} + \lambda_i\phi_i \ge \bar d_i$; "
               "details and formulas under *4 · Disturbance*.")
    switching_gain_summary(check)
    design_hints(check)


# -- 4 · Disturbance -----------------------------------------------------------
def _axis_inputs(prefix, fields):
    """Inputs per axis; fields = [(suffix, symbol, unit or None)]."""
    for axis, title, unit in AXIS_TITLES:
        st.markdown(f"*{title}*")
        for start in range(0, len(fields), 2):
            columns = st.columns(2)
            for column, (suffix, symbol, field_unit) in zip(
                    columns, fields[start:start + 2]):
                with column:
                    param(f"${symbol}_{axis}$ [{field_unit or unit}]",
                          f"{prefix}_{suffix}_{axis}")


def sidebar_disturbance(check):
    ss = st.session_state
    st.selectbox("Disturbance model", options=list(cfg.DIST_KINDS),
                 format_func=cfg.DIST_LABELS.get, key=K("dist_kind"),
                 help="Models implemented in `disturbances.py`.")
    st.caption(
        r"Additive generalized forces $\boldsymbol{\delta} = [\delta T_u,\ "
        r"\delta T_v,\ \delta T_r]^\top$ in the body frame: "
        r"$\boldsymbol{\tau} = \boldsymbol{\tau}_{delivered} + "
        r"\boldsymbol{\delta}$ (added after thruster saturation)."
    )
    kind = ss[K("dist_kind")]
    if kind == "constant":
        param(r"Surge $\delta T_u$ [N]", "dist_const_dTu")
        param(r"Sway $\delta T_v$ [N]", "dist_const_dTv")
        param(r"Yaw $\delta T_r$ [N m]", "dist_const_dTr")
    elif kind == "sinusoidal":
        equations(r"\delta T_i(t) &= b_i + A_i \sin(2\pi f_i t + \varphi_i)",
                  r"i &\in \{u,\ v,\ r\}")
        _axis_inputs("dist_sin", [("amp", "A", None), ("freq", "f", "Hz"),
                                  ("phase", r"\varphi", "rad"),
                                  ("bias", "b", None)])
    elif kind == "directional":
        st.caption(r"Fixed force $F$ in the inertial frame at angle $\beta$ "
                   r"from $+x$, projected onto the body axes:")
        equations(r"\delta T_u &= F\cos(\beta - \psi)",
                  r"\delta T_v &= F\sin(\beta - \psi), \qquad \delta T_r = 0")
        param(r"Magnitude $F$ [N]", "dist_dir_magnitude_N")
        param(r"Direction $\beta$ [deg from $+x$]", "dist_dir_direction_deg")
    elif kind == "wind":
        st.caption("Fixed profile of `wind_like_disturbance` (no parameters):")
        equations(
            r"\delta T_u &= 2\sin(2\pi\,0.035\,t)",
            r"\delta T_v &= 3 + 4\sin(2\pi\,0.05\,t + 0.8)",
            r"\delta T_r &= 2.5\sin(2\pi\,0.025\,t + 1.4)",
        )
    elif kind == "combined":
        equations(r"\boldsymbol{\delta} &= \boldsymbol{\delta}_{wind} + "
                  r"\boldsymbol{\delta}_{dir} + \boldsymbol{\delta}_{sin}")
        st.caption("Defaults = `combined` mode of `run_simulation.py`.")
        st.checkbox(r"Include wind-like component $\boldsymbol{\delta}_{wind}$",
                    key=K("dist_comb_wind"))
        st.markdown(r"**Directional component** $\boldsymbol{\delta}_{dir}$")
        pair((r"$F$ [N]", "dist_comb_dir_magnitude_N"),
             (r"$\beta$ [deg]", "dist_comb_dir_direction_deg"))
        st.markdown(r"**Sinusoidal component** $\boldsymbol{\delta}_{sin}$")
        _axis_inputs("dist_comb", [("amp", "A", None), ("freq", "f", "Hz"),
                                   ("phase", r"\varphi", "rad")])
    if check.config is None:
        return
    dist = check.config.disturbance
    if dist.active:
        b_u, b_v, b_r = (tex.num(b, 3) for b in dist.bounds())
        st.markdown("Worst case over time and heading:")
        equations(rf"\max|\delta T_u| &= {b_u}\ \mathrm{{N}}",
                  rf"\max|\delta T_v| &= {b_v}\ \mathrm{{N}}"
                  r"\quad\text{(not matched)}",
                  rf"\max|\delta T_r| &= {b_r}\ \mathrm{{N\,m}}")

    st.markdown("**Switching-gain hints for this disturbance**")
    with st.expander("How the gain hints are computed"):
        st.markdown(
            r"With the model inversion, each sliding variable obeys "
            r"$\dot s_i = -\lambda_i s_i - k_{s,i}\,\mathrm{sat}(s_i/\phi_i)"
            r" + d_i(t)$. The matched uncertainty $d_i$ is the disturbance "
            r"divided by the inertia ($z_1$ for surge, $z_3$ for yaw) plus "
            r"$\bar d_{traj,i}$, what the controller itself neglects along the "
            r"trajectory (approximate $\dot u_{cmd}$, missing $\dot r_{cmd}$; "
            r"it grows with the square of the reference frequency). Outside the "
            r"layer $s_i\dot s_i \le -\eta|s_i|$ holds when"
        )
        equations(r"k_{s,i} + \lambda_i\phi_i &\ge \bar d_i + \eta_i,"
                  r"\qquad \bar d_i = \frac{\max|\delta T_i|}{z} + "
                  r"\bar d_{traj,i}")
        st.markdown(
            r"The suggested value $k_{s,i} = 1.25\,\bar d_i$ adds a 25 % "
            r"margin and does not rely on $\lambda_i\phi_i$. A digital "
            r"controller does not chatter while $k_{s,i} < \phi_i(f_s - "
            r"\lambda_i)$, so a robust, chattering-free $k_{s,i}$ exists only if"
            r" $\phi_i > 1.25\,\bar d_i/(f_s - \lambda_i)$. Checked against "
            "the simulator: with the suggested gain the sliding variable stayed "
            "inside the layer in every tested case without thruster "
            "saturation. When the thrusters saturate (check below), the "
            "commanded force is not delivered and sliding is lost whatever "
            "the gains. The sway component is not matched: no SMC gain "
            "rejects it directly."
        )
    hints = tuning.gain_hints(check.config)
    for channel, name in (("surge", "surge"), ("yaw", "yaw")):
        _status, icon = tex.gain_status(hints[channel])
        with st.container(border=True):
            st.markdown(f"{icon} **Gain hint for the {name} SMC**")
            gain_hint(check, channel, "disturbance")
    st.markdown("  \n".join(tex.actuator_lines(
        tuning.actuator_check(check.config))))


def render_sidebar(check):
    ss = st.session_state
    with st.sidebar:
        st.header("SMC lab · IAcquaBot USV")
        st.text_input("Run label (optional)", key=K("run_label"),
                      placeholder=r"e.g. thin boundary layer")
        st.button("Run simulation", type="primary", width="stretch",
                  icon=":material/play_arrow:", disabled=not check.report.ok,
                  on_click=on_run_clicked,
                  help="Simulates a snapshot of the current settings. "
                       "Changing a parameter never starts a simulation.")
        status_box = st.container()
        st.button("Reset all settings to defaults", width="stretch",
                  on_click=on_reset_defaults)
        st.toggle("Sliders (off: number boxes)", key=K("use_sliders"))
        with st.expander("Scenario presets"):
            st.selectbox("Preset", options=list(cfg.PRESETS),
                         format_func=lambda name: cfg.PRESETS[name][0],
                         key=K("preset"), label_visibility="collapsed")
            st.caption(cfg.PRESETS[ss[K("preset")]][1])
            st.button("Apply preset", on_click=on_apply_preset,
                      width="stretch")
        with st.expander("1 - Simulation settings", expanded=True):
            sidebar_simulation(check)
        with st.expander("2 - Reference trajectory", expanded=True):
            sidebar_reference(check)
        with st.expander("3 - Controller", expanded=True):
            sidebar_controller(check)
        with st.expander("4 - Disturbance", expanded=False):
            sidebar_disturbance(check)
        st.caption("Obstacles, CBF filtering and the reference governor are "
                   "not used in this dashboard.")
    return status_box


def render_status(status_box, check, history):
    with status_box:
        for message in check.report.errors:
            st.error(message, icon=":material/error:")
        for message in check.report.warnings:
            st.warning(message, icon=":material/warning:")
        latest = history.latest
        if latest is not None and check.config is not None:
            stale, differences = staleness(latest, check.config)
            if stale:
                st.info(f"{len(differences)} setting(s) changed since "
                        f"{latest.title}. Press **Run simulation** to "
                        "update the results.", icon=":material/pending:")


# ---------------------------------------------------------------------------
# Running.
# ---------------------------------------------------------------------------
def execute_pending_run(check, history):
    ss = st.session_state
    request = ss[K("pending_run")]
    ss[K("pending_run")] = None
    if request is None:
        return
    # Immutable snapshot of the widget values of this script run.
    snapshot = check.config
    if snapshot is None or not check.report.ok:
        ss[K("last_failure")] = ("Configuration is invalid: "
                                 + "; ".join(check.report.errors))
        return
    sim = snapshot.simulation
    if sim.mode == cfg.DIGITAL:
        # About 0.5-1 ms of computation per controller sample.
        samples = snapshot.t_final_s * sim.sample_rate_hz
        mode = (f"digital controller at {sim.sample_rate_hz:g} Hz, usually "
                f"{max(1, round(samples / 2000))}-{max(2, round(samples / 1000))}"
                " s")
    else:
        mode = "continuous controller; thin boundary layers take longer"
    # st.status instead of st.spinner: in Streamlit 1.66 the spinner's timer
    # thread competes for the GIL and doubles the simulation time.
    placeholder = st.empty()
    with placeholder.container():
        status = st.status(f"Simulating {snapshot.t_final_s:.4g} s ({mode})...",
                           state="running")
    try:
        result = engine.run_from_config(snapshot,
                                        max_wall_time_s=MAX_WALL_TIME_S)
    except (cfg.ConfigValidationError, engine.SimulationError) as exc:
        ss[K("last_failure")] = str(exc)
        return
    except Exception as exc:  # noqa: BLE001 - report any simulator error
        ss[K("last_failure")] = (f"Unexpected error "
                                 f"({type(exc).__name__}): {exc}")
        return
    finally:
        status.update(state="complete")
        placeholder.empty()
    history.add(result, request["label"])
    ss[K("last_failure")] = None


# ---------------------------------------------------------------------------
# Results.
# ---------------------------------------------------------------------------
def cached_figure(builder, current, comparison, *args):
    """Build a figure once per (runs, display options).

    Any sidebar change reruns the script; without this cache every rerun
    would rebuild all figures (about 1.5 s for two long runs).  Results are
    immutable and identified by their run id, so the cache can never show
    outdated data.
    """
    cache = st.session_state.setdefault(K("fig_cache"), {})
    runs = (current.result.run_id,
            comparison.result.run_id if comparison is not None else None)
    if cache.get("runs") != runs or len(cache) > MAX_CACHED_FIGURES:
        cache.clear()
        cache["runs"] = runs
    key = (builder.__name__,) + tuple(
        tuple(a) if isinstance(a, (list, tuple)) else a for a in args)
    if key not in cache:
        cache[key] = builder(current, comparison, *args)
    return cache[key]


def comparison_record(history):
    """The run drawn in gray (default: the previous run), or None."""
    ss = st.session_state
    latest = history.latest
    others = [r.number for r in history.records if r.number != latest.number]
    if not others:
        return None
    options = list(reversed(others)) + [0]
    if ss.get(K("cmp_seen_latest")) != latest.number or \
            ss.get(K("cmp_number")) not in options:
        ss[K("cmp_number")] = history.previous.number
        ss[K("cmp_seen_latest")] = latest.number
    titles = {r.number: r.title for r in history.records}
    titles[0] = "None (current run only)"
    number = st.selectbox(
        "Run drawn in gray for comparison", options=options,
        format_func=lambda n: titles[n] + (" (previous run)"
                                           if n == history.previous.number
                                           else ""),
        key=K("cmp_number"),
        help="Display only: changing it never runs a simulation.")
    return None if number == 0 else history.get(number)


def time_window(current, comparison):
    """Display-only time window shared by all time and phase plots."""
    t_end = float(current.result.data["t"].iloc[-1])
    if comparison is not None:
        t_end = max(t_end, float(comparison.result.data["t"].iloc[-1]))
    key = K("window")
    # Reset to the full run whenever the plotted runs change length.
    if st.session_state.get(K("window_span")) != t_end:
        st.session_state[K("window_span")] = t_end
        st.session_state[key] = (0.0, t_end)
    return st.slider(
        "Time window [s] (XY trajectory, phase planes and time plots)",
        min_value=0.0, max_value=t_end,
        step=max(round(t_end / 560, 2), 0.01), key=key,
        help="Display only (no new simulation): exclude the initial "
             "transient to zoom on the sliding phase. The performance "
             "indices always use the whole run.")


def render_run_cards(current, comparison):
    changed = [key for key, _o, _n in cfg.config_diff(
        comparison.config, current.config)] if comparison else []
    columns = st.columns(2 if comparison else 1)
    with columns[0].container(border=True):
        st.markdown(
            rf"$\textcolor{{{tex.CURRENT_TEX_COLOR}}}{{\blacksquare}}$ "
            f"**Current: {current.title}** · {current.result.created_at}"
        )
        st.markdown(tex.run_card(current.config, changed))
        if comparison is not None:
            st.caption("Parameters in orange differ from the comparison run.")
    if comparison is not None:
        with columns[1].container(border=True):
            st.markdown(
                rf"$\textcolor{{{tex.PREVIOUS_TEX_COLOR}}}{{\blacksquare}}$ "
                f"**Comparison: {comparison.title}** · "
                f"{comparison.result.created_at}"
            )
            st.markdown(tex.run_card(comparison.config))


CARD_INDICES = (
    ("e_p_rms", r"RMS $\lVert e_p \rVert$ [m]"),
    ("e_psi_rms", r"RMS $e_\psi$ [rad]"),
    ("in_layer_u_pct", r"$\eta_{BL,u}$ [%]"),
    ("in_layer_r_pct", r"$\eta_{BL,r}$ [%]"),
    ("Tu_hf_rms", r"$\mathrm{HF}(T_u)$ [N]"),
    ("Tr_hf_rms", r"$\mathrm{HF}(T_r)$ [N m]"),
)


def render_metric_cards(current, comparison):
    cols = st.columns(len(CARD_INDICES))
    m = current.result.metrics
    pm = comparison.result.metrics if comparison is not None else None
    for col, (key, label) in zip(cols, CARD_INDICES):
        better = met.INDEX_INFO[key][4]
        delta, color = None, "inverse" if better == "lower" else "normal"
        arrow = "auto"
        if pm is not None and math.isfinite(m[key]) and math.isfinite(pm[key]):
            change = m[key] - pm[key]
            if abs(change) <= 1e-9 * max(abs(m[key]), abs(pm[key]), 1e-12):
                delta, color = f"no change vs #{comparison.number}", "off"
                arrow = "off"
            else:
                delta = f"{cfg.format_value(change)} vs #{comparison.number}"
        col.metric(label, _fmt(m[key]), delta=delta, delta_color=color,
                   delta_arrow=arrow,
                   help=met.INDEX_INFO[key][2] + ". Change vs the comparison "
                        "run below the value (green = better).")


def render_header(record, check):
    result = record.result
    st.subheader(record.title)
    sim = record.config.simulation
    impl = (f"digital controller, {sim.sample_rate_hz:g} Hz"
            if sim.mode == cfg.DIGITAL else "continuous controller")
    st.caption(f"Computed {result.created_at} in {result.total_time_s:.1f} s "
               f"({impl}, {result.metrics['samples']} samples, "
               f"$T_{{sim}}$ = {result.metrics['t_final_s']:.4g} s) · run id "
               f"{result.run_id}")
    stale, differences = staleness(record, check.config)
    if stale:
        st.warning(
            f"**The sidebar settings differ from {record.title}.** The results "
            "below were computed with that run's own parameters. Press **Run "
            "simulation** to simulate the current settings.",
            icon=":material/history:")
        if differences:
            with st.expander(f"Show the {len(differences)} changed "
                             "setting(s)"):
                st.markdown(tex.change_list(differences))
    else:
        st.success("The results correspond to the current sidebar settings.",
                   icon=":material/check_circle:")


def render_main_tab(current, comparison):
    """XY trajectory with the time window and the disturbance, followed by
    the phase planes with the sliding surfaces.  Returns the time window,
    which the other tabs use too."""
    window = time_window(current, comparison)
    st.plotly_chart(cached_figure(figures.xy_figure, current, comparison,
                                  window), key=K("fig_xy"))
    st.caption(
        "Bright: current run. Gray: comparison run. Dashed: reference. Both "
        "runs share the same axes and scale (1 m = 1 m). With a reduced time "
        "window the rest of the current run is drawn faintly and the window "
        "limits are labelled. Orange: disturbance of the current run (a "
        "fixed-direction force as one arrow above the path; time-varying or "
        "body-fixed forces as faint arrows along the path, which can be hidden "
        r"from the legend). The lookahead point $\boldsymbol p_s$ can be shown "
        "from the legend.")
    st.markdown("#### Sliding surfaces in the phase plane")
    render_phase_plots(current, comparison, window)
    return window


def render_errors_tab(current, comparison, window):
    st.plotly_chart(cached_figure(figures.errors_figure, current, comparison,
                                  window), key=K("fig_errors"))
    st.markdown(
        "Errors are *actual minus desired* (Slotine's convention), so the "
        "sliding variables of `smc_figure_eight` are exactly "
        r"$s_u = \dot e_u + k_{pos} e_u$ and $s_r = \dot e_\psi + k_\psi "
        r"e_\psi$. $e_u$ is the lookahead-point error projected on the "
        r"commanded course $\boldsymbol n$; the shaded bands are the boundary "
        r"layers $|s_i| \le \phi_i$."
    )


def render_controls_tab(current, comparison, window):
    show_cmd = st.checkbox("Show the commanded forces before thruster "
                           "saturation (current run)", value=True,
                           key=K("show_cmd"))
    st.plotly_chart(cached_figure(figures.controls_figure, current,
                                  comparison, show_cmd, window),
                    key=K("fig_controls"))
    st.markdown(
        r"The switching function $\mathrm{sat}(s_i/\phi_i)$ shows the "
        r"activity of the switching term $k_{s,i}\,\mathrm{sat}(s_i/\phi_i)$: "
        r"it stays inside $(-1, 1)$ while the state is in the boundary layer "
        r"and jumps between $\pm 1$ when the controller chatters. The force "
        r"axes are fixed to the limits of $T_u$, $T_r$ with all thrusters "
        "saturated (red dotted), so every run is drawn at the same scale."
    )
    with st.expander("Thruster forces of the current run"):
        st.plotly_chart(cached_figure(_thrusters, current, None, window),
                        key=K("fig_thrusters"))


def _thrusters(current, _comparison, window):
    return figures.thrusters_figure(current, window)


def render_phase_plots(current, comparison, window):
    st.markdown(
        r"Sliding surface $s_i = \dot e_i + k_i e_i = 0$ (dashed line of slope "
        r"$-k_i$) and boundary layer $|s_i| \le \phi_i$, i.e. "
        r"$-k_i e_i - \phi_i \le \dot e_i \le -k_i e_i + \phi_i$ (shaded). "
        r"Both runs use the same axes; the comparison run's surface and band "
        "are drawn in gray."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(cached_figure(figures.phase_figure, current,
                                      comparison, "surge", window),
                        key=K("fig_phase_u"))
    with c2:
        st.plotly_chart(cached_figure(figures.phase_figure, current,
                                      comparison, "yaw", window),
                        key=K("fig_phase_r"))
    st.caption(r"$\dot e_i$ uses the controller's own feed-forward terms, so "
               r"the relation with $s_i$ is exact. Because that feed-forward "
               r"is approximate, a state resting on $s_i = 0$ can still show "
               r"a small steady error (see the *Equations* tab).")


def _style_verdict(value):
    return {"better": "color: #2b8a3e; font-weight: 600",
            "worse": "color: #c92a2a; font-weight: 600"}.get(value, "")


def render_indices_tab(current, comparison):
    cm = current.result.metrics
    pm = comparison.result.metrics if comparison is not None else None
    st.plotly_chart(cached_figure(figures.indices_figure, current,
                                  comparison), key=K("fig_indices"))
    st.caption("↓ lower is better, ↑ higher is better. \"never\" = the "
               "boundary layer was never reached.")
    table = met.index_table(cm, pm)
    reaching = table["key"].str.startswith("t_reach")
    for column in ["Current"] + (["Previous"] if pm else []):
        table[column] = [_fmt(v, never=r) for v, r in zip(table[column],
                                                         reaching)]
    for column in (["Change", "Change %"] if pm else []):
        table[column] = table[column].map(_fmt)
    table = table.drop(columns="key")
    if pm is not None:
        table = table.rename(columns={"Previous": f"Comparison #"
                                                  f"{comparison.number}",
                                      "Current": f"Current #{current.number}"})
        styled = table.style.map(_style_verdict, subset=["Verdict"])
    else:
        table = table.rename(columns={"Current": f"Current #{current.number}"})
        styled = table
    st.dataframe(styled, hide_index=True, height=35 * (len(table) + 1) + 3)
    with st.expander("Definitions of the indices", icon=":material/functions:",
                     expanded=False):
        render_index_definitions()


def _fmt(value, never=False):
    """Compact number; NaN is "never" for reaching times, "–" otherwise."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "never" if never and math.isnan(value) else "–"
    return cfg.format_value(value)


def render_index_definitions():
    st.markdown(
        r"$N$ samples $t_k$ (every controller sample in the digital "
        r"implementation), integrals by the trapezoidal rule, "
        rf"$\Omega_{{ss}}$ = final {met.STEADY_STATE_FRACTION:.0%} of the run "
        r"(steady state), $|\Omega_{ss}|$ its duration. The chattering "
        r"indices use only $\Omega_{ss}$, so the reaching transient is not "
        r"counted as chattering. $\langle T \rangle_{1\,\mathrm{s}}$ is a "
        rf"centred moving average over {met.HF_WINDOW_S:g} s, so "
        r"$T - \langle T \rangle_{1\,\mathrm{s}}$ keeps the high-frequency "
        "(chattering) part of the control."
    )
    for group in met.GROUPS:
        lines = [
            rf"{sym} &= {defn} && [{tex.unit(unit)}]"
            for key, (g, sym, _label, unit, _b, defn) in met.INDEX_INFO.items()
            if g == group
        ]
        st.markdown(f"**{group}**")
        st.markdown(tex.aligned(*lines))


def render_disturbance_tab(current, comparison, window):
    st.plotly_chart(cached_figure(figures.disturbance_figure, current,
                                  comparison, window), key=K("fig_dist"))


def render_equations_tab():
    st.markdown("**1. Identified vessel model** (`usv_model.py`), "
                r"$\boldsymbol\eta = [x, y, \psi]^\top$, "
                r"$\boldsymbol\nu = [u, v, r]^\top$")
    equations(
        r"\dot{\boldsymbol\eta} &= \boldsymbol J(\psi)\,\boldsymbol\nu",
        r"z_1(\dot u - v r) + z_4 u + z_5|u|u &= T_u + \delta T_u",
        r"z_2(\dot v + u r) + z_6 v + z_7|v|v &= \delta T_v",
        r"z_3\dot r + z_8 r + z_9|r|r &= T_r + \delta T_r",
    )
    st.markdown(", ".join(f"$z_{i}$ = {PARAMS[f'z{i}']:g}"
                          for i in range(1, 10)))
    st.markdown("**2. Thruster allocation and saturation** "
                r"(4 thrusters, arm $b$ = " f"{THRUSTER_ARM:g} m)")
    equations(
        r"F_{1,2} &= \tfrac{1}{4}\left(T_u - T_r/b\right),\quad "
        r"F_{3,4} = \tfrac{1}{4}\left(T_u + T_r/b\right)",
        rf"F_i &\leftarrow \min\!\left({THRUSTER_FWD_MAX:g},\ \max(-"
        rf"{THRUSTER_REV_MAX:g},\ F_i)\right)\ \mathrm{{N}}",
    )
    st.markdown("**3. Reference and lookahead point**")
    equations(
        r"\boldsymbol p_{ref}(t) &= [A\sin\omega t,\ A\sin\omega t\cos\omega t]"
        r"^\top,\quad \omega = 2\pi f",
        r"\boldsymbol p_s &= \boldsymbol p + L[\cos\psi,\ \sin\psi]^\top,"
        r"\quad \boldsymbol p_{s,ref} = \boldsymbol p_{ref} + L[\cos\psi_{ref},"
        r"\ \sin\psi_{ref}]^\top",
    )
    st.markdown("**4. Outer loop (commands)**")
    equations(
        r"\boldsymbol v_{cmd} &= \dot{\boldsymbol p}_{s,ref} + k_{pos}"
        r"(\boldsymbol p_{s,ref} - \boldsymbol p_s)",
        r"u_{cmd} &= \lVert \boldsymbol v_{cmd} \rVert,\quad \psi_{cmd} = "
        r"\operatorname{atan2}(v_{cmd,y}, v_{cmd,x}),\quad r_{cmd} = "
        r"\dot\psi_{cmd} + k_\psi\operatorname{wrap}(\psi_{cmd} - \psi)",
    )
    st.markdown("**5. Sliding mode controllers** (`smc_figure_eight`)")
    equations(
        r"s_u &= u - u_{cmd}, & s_r &= r - r_{cmd}",
        r"\dot u_{des} &= \dot u_{cmd} - \lambda_u s_u - k_{s,u}\,"
        r"\mathrm{sat}(s_u/\phi_u), & \dot r_{des} &= -\lambda_r s_r - "
        r"k_{s,r}\,\mathrm{sat}(s_r/\phi_r)",
        r"T_u &= z_4 u + z_5|u|u + z_1(\dot u_{des} - v r), & T_r &= z_8 r + "
        r"z_9|r|r + z_3\,\dot r_{des}",
        r"\mathrm{sat}(x) &= \min(1, \max(-1, x)) & &",
    )
    st.markdown("**6. Phase-plane coordinates** (errors = actual − desired)")
    equations(
        r"\boldsymbol n &= \boldsymbol v_{cmd}/\lVert\boldsymbol v_{cmd}\rVert"
        r" = [\cos\psi_{cmd},\ \sin\psi_{cmd}]^\top",
        r"e_u &= \boldsymbol n^\top(\boldsymbol p_s - \boldsymbol p_{s,ref}),"
        r"\quad \dot e_u = u - \boldsymbol n^\top\dot{\boldsymbol p}_{s,ref}"
        r"\ \Rightarrow\ s_u = \dot e_u + k_{pos} e_u",
        r"e_\psi &= \operatorname{wrap}(\psi - \psi_{cmd}),\quad \dot e_\psi = "
        r"r - \dot\psi_{cmd}\ \Rightarrow\ s_r = \dot e_\psi + k_\psi e_\psi",
        r"|s_i| \le \phi_i\ &\Longleftrightarrow\ -k_i e_i - \phi_i \le "
        r"\dot e_i \le -k_i e_i + \phi_i",
    )
    st.caption(
        r"These identities are exact (the tests check them to $10^{-12}$). "
        r"$\dot e_u$ equals $de_u/dt$ only when $\psi = \psi_{cmd}$ and "
        r"$\boldsymbol n$ does not rotate; $\dot e_\psi$ uses the controller's "
        r"$\dot\psi_{cmd}$ (next section)."
    )
    st.markdown("**7. Sliding dynamics, disturbances and design hints**")
    equations(
        r"\dot s_i &= -\lambda_i s_i - k_{s,i}\,\mathrm{sat}(s_i/\phi_i) + d_i,"
        r"\qquad d_u = \frac{\delta T_u}{z_1} + \Delta_u,\quad d_r = "
        r"\frac{\delta T_r}{z_3} - \dot r_{cmd}",
        r"s_i\dot s_i &\le -\eta_i|s_i| \text{ for } |s_i| \ge \phi_i "
        r"\ \Longleftarrow\ k_{s,i} + \lambda_i\phi_i \ge \bar d_i + \eta_i,"
        r"\qquad \bar d_i = \frac{\max|\delta T_i|}{z} + \bar d_{traj,i}",
        r"s_{i,ss} &= \frac{d_i}{\lambda_i + k_{s,i}/\phi_i}\ \ (\text{constant }"
        r"d_i,\ \text{inside the layer}),\qquad e_{i,ss} \approx "
        r"\frac{s_{i,ss}}{k_i}",
    )
    st.caption(
        r"$\Delta_u$: error of the feed-forward $\dot u_{cmd}$, which (like "
        r"$\dot\psi_{cmd}$) is computed with the hull-centre reference "
        r"acceleration $\ddot{\boldsymbol p}_{ref}$ instead of "
        r"$\ddot{\boldsymbol p}_{s,ref}$. The yaw law has no $\dot r_{cmd}$ "
        r"feed-forward, so $\dot r_{cmd}$ acts as a matched disturbance. The "
        r"$\dot\psi_{cmd}$ error is what leaves a small heading error on the "
        r"surface: $e_\psi \approx \Delta_\psi / k_\psi$. These neglected "
        r"terms form $\bar d_{traj,i}$ in the gain hints (under *Disturbances*)."
    )
    st.markdown("**8. Digital implementation** (sample period "
                r"$T_s = 1/f_s$, zero-order hold)")
    equations(
        r"\boldsymbol\tau(t) &= \boldsymbol\tau(t_k),\quad t \in [t_k, t_k + "
        r"T_s)",
        r"s_{k+1} &\approx (1 - \Gamma_i)\, s_k \ \text{inside the layer},\qquad"
        r" \Gamma_i = T_s\left(\lambda_i + \frac{k_{s,i}}{\phi_i}\right)",
        r"\Gamma_i &< 1:\ \text{monotone};\quad 1 < \Gamma_i < 2:\ "
        r"\text{sign alternation (chattering)};\quad \Gamma_i > 2:\ "
        r"\text{limit cycle}",
    )
    st.markdown("**9. Disturbances**")
    equations(r"\delta T_i(t) &= b_i + a_i\sin(2\pi f_i t + \theta_i),\quad "
              r"i \in \{u, v, r\}")


def render_runs_tab(current, history):
    result = current.result
    st.markdown(f"**Configuration of {current.title}** (run id "
                f"{result.run_id})")
    flat = result.config.flat()
    st.dataframe(pd.DataFrame({"Parameter": list(flat),
                               "Value": [str(v) for v in flat.values()]}),
                 hide_index=True)
    stem = f"usv_smc_run{current.number}_{result.run_id}"
    c1, c2 = st.columns(2)
    c1.download_button("Download configuration (JSON)",
                       data=result.config.to_json(), file_name=f"{stem}.json",
                       mime="application/json", on_click="ignore",
                       icon=":material/download:")
    c2.download_button("Download time series (CSV)", data=result.to_csv_bytes,
                       file_name=f"{stem}.csv", mime="text/csv",
                       on_click="ignore", icon=":material/download:")
    with st.expander("Reproduce this run from Python"):
        st.code(
            "from dashboard02_smc.config import SimulationConfig\n"
            "from dashboard02_smc.engine import run_from_config\n\n"
            f"config = SimulationConfig.from_json(open('{stem}.json').read())\n"
            "result = run_from_config(config)\n"
            "print(result.metrics)\n"
            "result.data  # pandas DataFrame with all time series\n",
            language="python")

    st.divider()
    st.markdown("**Runs stored in this session**")
    rows = [{
        "Run": r.title,
        "λu": r.config.surge.lambda_u, "ks,u": r.config.surge.k_s_u,
        "φu": r.config.surge.phi_u, "kpos": r.config.surge.k_pos,
        "λr": r.config.yaw.lambda_r, "ks,r": r.config.yaw.k_s_r,
        "φr": r.config.yaw.phi_r, "kψ": r.config.yaw.k_psi,
        "RMS ‖ep‖ [m]": r.result.metrics["e_p_rms"],
        "HF(Tr) [N m]": r.result.metrics["Tr_hf_rms"],
        "Computed": r.result.created_at,
    } for r in reversed(history.records)]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    titles = {r.number: r.title for r in history.records}
    c1, c2, c3 = st.columns([2, 1, 1])
    number = c1.selectbox("Stored run", options=list(reversed(titles)),
                          format_func=titles.get, key=K("history_pick"),
                          label_visibility="collapsed")
    c2.button("Load its settings into the sidebar", on_click=on_load_run,
              args=(number,), icon=":material/upload:")
    c3.button("Clear stored runs", on_click=history.clear,
              icon=":material/delete:")


def render_results(history, check):
    failure = st.session_state[K("last_failure")]
    latest = history.latest
    if failure is not None:
        shown = (f" The results below still belong to {latest.title}."
                 if latest is not None else "")
        st.error(f"**The last run failed; no new results were produced.** "
                 f"{failure}{shown}", icon=":material/error:")
    if latest is None:
        st.info("Set up the reference, both sliding mode controllers and the "
                "disturbances in the sidebar, then press **Run simulation**. "
                "Changing a parameter never starts a simulation; each run "
                "keeps a frozen copy of its parameters. From the second run "
                "on, the previous run is drawn in gray under the current one.",
                icon=":material/info:")
        with st.expander("Model, control law and phase-plane equations",
                         expanded=True, icon=":material/functions:"):
            render_equations_tab()
        return

    render_header(latest, check)
    c1, _c2 = st.columns([2, 3])
    with c1:
        comparison = comparison_record(history)
    tabs = st.tabs(["Trajectory & sliding surfaces", "Controller errors",
                    "Control inputs", "Vessel states", "Disturbances",
                    "Performance indices", "Equations", "Runs & export"])
    with tabs[0]:
        window = render_main_tab(latest, comparison)
    t_end = float(latest.result.data["t"].iloc[-1])
    window_note = (None if window[0] <= 0.0 and window[1] >= t_end else
                   rf"Time window $t \in [{window[0]:.4g},\ {window[1]:.4g}]$ s "
                   "(set in the first tab).")
    for index, render in (
        (1, lambda: render_errors_tab(latest, comparison, window)),
        (2, lambda: render_controls_tab(latest, comparison, window)),
        (3, lambda: st.plotly_chart(cached_figure(
            figures.vessel_states_figure, latest, comparison, window),
            key=K("fig_states"))),
        (4, lambda: render_disturbance_tab(latest, comparison, window)),
    ):
        with tabs[index]:
            if window_note:
                st.caption(window_note)
            render()
    with tabs[5]:
        render_indices_tab(latest, comparison)
    with tabs[6]:
        render_equations_tab()
    with tabs[7]:
        render_runs_tab(latest, history)

    st.divider()
    render_comparison_section(latest, comparison)


def render_comparison_section(current, comparison):
    """Parameter cards and key indices, below the plots."""
    if comparison is not None:
        st.markdown(f"#### Changes: {current.title} vs {comparison.title}")
    else:
        st.markdown(f"#### Parameters and key indices of {current.title}")
    render_run_cards(current, comparison)
    render_metric_cards(current, comparison)


def main():
    init_session_state()
    check = cfg.check_form(current_form())
    history = st.session_state[K("history")]

    status_box = render_sidebar(check)
    st.title("Sliding mode control lab · IAcquaBot USV")
    st.caption("Surge and yaw sliding mode controllers of `controllers.py` "
               "tracking the figure-eight reference on the identified vessel "
               "model of `usv_model.py`.")
    progress_area = st.container()
    results_slot = st.empty()
    if st.session_state[K("pending_run")] is not None:
        # Old results are removed while the new run is computed, so they
        # cannot be mistaken for the new ones.
        results_slot.empty()
        with progress_area:
            execute_pending_run(check, history)
    render_status(status_box, check, history)
    with results_slot.container():
        render_results(history, check)


main()
