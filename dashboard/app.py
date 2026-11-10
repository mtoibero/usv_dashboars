"""Streamlit dashboard for the IAcquaBot USV simulator.

Launch from the project root with either of::

    streamlit run dashboard/app.py
    python -m dashboard

This file only contains the user interface.  Configuration, simulation,
metrics and figures live in GUI-independent modules of this package.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import math  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import config as cfg  # noqa: E402
from dashboard import engine, figures  # noqa: E402
from dashboard import metrics as met  # noqa: E402
from dashboard.session import RunHistory, staleness  # noqa: E402

import controllers  # noqa: E402  (simulator module, for the initial state only)
from usv_model import THRUSTER_FWD_MAX, THRUSTER_REV_MAX  # noqa: E402

MAX_WALL_TIME_S = 900.0   # abort runs that take longer than this (wall clock)
SIDEBAR_WIDTH_PX = 400

DURATION_MODES = {
    "laps": r"Number of laps $\times$ period ($N_{laps}\,T$)",
    "manual": "Custom duration",
}
INITIAL_MODES = {
    "reference": "On the reference at $t = 0$ (simulator default)",
    "custom": "Custom initial state",
}
TIMING_MODES = {
    "period": "Period $T$",
    "peak_speed": r"Peak speed $v_{peak}$",
}

st.set_page_config(
    page_title="IAcquaBot USV simulator",
    page_icon=":material/sailing:",
    layout="wide",
    # An integer sets the initial sidebar width in pixels (user-resizable);
    # wide enough for the LaTeX equations of the controller section.
    initial_sidebar_state=SIDEBAR_WIDTH_PX,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session_state():
    for key, value in cfg.default_form().items():
        if key not in st.session_state:
            st.session_state[key] = value
        else:
            # Re-assigning keeps values of widgets that are currently hidden
            # (Streamlit drops the state of widgets that are not rendered).
            st.session_state[key] = st.session_state[key]
    st.session_state.setdefault("history", RunHistory())
    st.session_state.setdefault("run_label", "")
    st.session_state.setdefault("pending_run", None)
    st.session_state.setdefault("last_failure", None)


def current_form():
    return {key: st.session_state[key] for key in cfg.FORM_KEYS}


def apply_form(values):
    for key, value in values.items():
        if key in cfg.FORM_KEYS:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Widget callbacks (run before the script re-executes)
# ---------------------------------------------------------------------------
def on_run_clicked():
    st.session_state.pending_run = {"label": st.session_state.run_label}
    st.session_state.run_label = ""


def on_reset_defaults():
    apply_form(cfg.default_form())


def on_load_run_config(number):
    record = st.session_state.history.get(number)
    apply_form(cfg.config_to_form(record.config, base=current_form()))


def on_timing_mode_changed():
    """Keep the same trajectory when switching between period/peak speed.

    Values are not rounded: switching a display mode must not change the
    experiment."""
    ss = st.session_state
    amplitude = ss.ref_amplitude_m
    if amplitude <= 0.0:
        return
    if ss.ref_timing_mode == "peak_speed" and ss.ref_period_s > 0.0:
        ss.ref_peak_speed_mps = cfg.figure_eight_peak_speed(
            amplitude, ss.ref_period_s
        )
    elif ss.ref_timing_mode == "period" and ss.ref_peak_speed_mps > 0.0:
        ss.ref_period_s = cfg.figure_eight_period_from_peak_speed(
            amplitude, ss.ref_peak_speed_mps
        )


def on_duration_mode_changed():
    """Keep the same duration when switching between laps/custom time."""
    ss = st.session_state
    period = cfg.form_period(current_form())
    if period is None:
        return
    if ss.sim_duration_mode == "manual" and ss.sim_laps > 0.0:
        ss.sim_t_final_s = ss.sim_laps * period
    elif ss.sim_duration_mode == "laps" and ss.sim_t_final_s > 0.0:
        ss.sim_laps = ss.sim_t_final_s / period


def on_initial_mode_changed():
    """Start custom initial states from the current reference state."""
    ss = st.session_state
    period = cfg.form_period(current_form())
    if ss.sim_initial_mode != "custom" or period is None:
        return
    if ss.ref_amplitude_m <= 0.0:
        return
    x0, y0, psi0, u0, v0, r0 = controllers.figure_eight_initial_state(
        ss.ref_amplitude_m, period
    )
    ss.sim_x0_m, ss.sim_y0_m = float(x0), float(y0)
    ss.sim_psi0_deg = math.degrees(psi0)
    ss.sim_u0_mps = float(u0)
    ss.sim_v0_mps, ss.sim_r0_degps = float(v0), math.degrees(r0)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def num(label, key, help=None, step=None, fmt="%g", min_value=None):
    return st.number_input(
        label, key=key, help=help, step=step, format=fmt, min_value=min_value
    )


def equations(*lines):
    """Render LaTeX lines as one aligned display block (KaTeX)."""
    body = r" \\ ".join(lines)
    st.markdown("$$\n\\small\\begin{aligned}" + body + "\\end{aligned}\n$$")


def sidebar_simulation(check):
    ss = st.session_state
    st.radio(
        r"Total simulation time $T_{sim}$", options=list(DURATION_MODES),
        format_func=DURATION_MODES.get, key="sim_duration_mode",
        on_change=on_duration_mode_changed,
    )
    if ss.sim_duration_mode == "manual":
        num(r"Total simulation time $T_{sim}$ [s]", "sim_t_final_s", step=10.0,
            min_value=0.0)
    elif check.config is not None:
        st.caption(
            rf"$T_{{sim}} = N_{{laps}}\,T$ = {ss.sim_laps:g} $\times$ "
            f"{check.config.reference.period_s:.4g} s = "
            f"**{check.config.simulation.t_final_s:.4g} s** "
            r"(set $N_{laps}$ in *Reference trajectory*)"
        )
    num(r"Output sampling interval $\Delta t$ [s]", "sim_dt_s", step=0.01,
        min_value=0.0,
        help=r"Spacing of the stored samples (`t_eval` of `solve_ivp`). The "
             r"integrator is adaptive (RK45, $\mathrm{rtol}=10^{-8}$) and the "
             r"controller is evaluated continuously, so $\Delta t$ does not "
             "change the closed-loop dynamics, only the output resolution.")

    st.radio(
        "Initial state", options=list(INITIAL_MODES),
        format_func=INITIAL_MODES.get, key="sim_initial_mode",
        on_change=on_initial_mode_changed,
    )
    if ss.sim_initial_mode == "custom":
        c1, c2 = st.columns(2)
        with c1:
            num(r"$x_0$ [m]", "sim_x0_m", step=0.5)
            num(r"Heading $\psi_0$ [deg]", "sim_psi0_deg", step=5.0)
            num(r"Sway $v_0$ [m/s]", "sim_v0_mps", step=0.05)
        with c2:
            num(r"$y_0$ [m]", "sim_y0_m", step=0.5)
            num(r"Surge $u_0$ [m/s]", "sim_u0_mps", step=0.05)
            num(r"Yaw rate $r_0$ [deg/s]", "sim_r0_degps", step=1.0)
    elif check.config is not None:
        s = check.config.simulation
        st.caption(
            f"$x_0$ = {s.x0_m:.3g} m, $y_0$ = {s.y0_m:.3g} m, "
            rf"$\psi_0$ = {math.degrees(s.psi0_rad):.4g}°, "
            f"$u_0$ = {s.u0_mps:.4g} m/s, $v_0$ = {s.v0_mps:g} m/s, "
            f"$r_0$ = {math.degrees(s.r0_radps):g}°/s"
        )


def sidebar_reference(check):
    ss = st.session_state
    st.selectbox(
        "Reference type", options=list(cfg.REFERENCE_TYPES),
        format_func=lambda kind: cfg.REFERENCE_TYPES[kind].label, key="ref_kind",
        help="Reference trajectories implemented in `controllers.py`.",
    )
    equations(
        r"x_{ref}(t) &= A\sin(\omega t)",
        r"y_{ref}(t) &= A\sin(\omega t)\cos(\omega t)",
        r"\omega &= 2\pi/T,\qquad v_{peak} = \sqrt{2}\,A\,\omega",
    )
    num(r"Amplitude $A$ [m]", "ref_amplitude_m", step=1.0, min_value=0.0,
        help=r"Half-width of the figure-eight along $x$ (its half-width "
             r"along $y$ is $A/2$).")
    st.radio("Define timing by", options=list(TIMING_MODES),
             format_func=TIMING_MODES.get, key="ref_timing_mode",
             horizontal=True, on_change=on_timing_mode_changed)
    if ss.ref_timing_mode == "period":
        num(r"Period $T$ [s]", "ref_period_s", step=10.0, min_value=0.0,
            help="Time for one full figure-eight. Shorter = faster reference.")
    else:
        num(r"Desired peak speed $v_{peak}$ [m/s]", "ref_peak_speed_mps",
            step=0.05, min_value=0.0,
            help=r"Peak reference speed, reached at the crossing point: "
                 r"$v_{peak} = \sqrt{2}\,A\,2\pi/T$. The period $T$ is "
                 "derived from it.")
    if ss.sim_duration_mode == "laps":
        num(r"Number of laps $N_{laps}$ [-]", "sim_laps", step=0.5,
            min_value=0.0, help=r"Total simulation time $T_{sim} = N_{laps}\,T$.")
    if check.config is not None:
        ref = check.config.reference
        if ref.amplitude_m > 0 and ref.period_s > 0:
            mean_speed = cfg.figure_eight_mean_speed(ref.amplitude_m, ref.period_s)
            st.caption(
                f"$T$ = **{ref.period_s:.4g} s**, "
                f"$v_{{peak}}$ = **{ref.peak_speed_mps:.3f} m/s**, "
                rf"$\bar v$ = {mean_speed:.3f} m/s, lap length "
                f"{mean_speed * ref.period_s:.1f} m, "
                f"laps simulated {check.config.laps:.3g}"
            )


def sidebar_controller():
    ss = st.session_state
    st.selectbox(
        "Controller", options=list(cfg.CONTROLLER_TYPES),
        format_func=lambda kind: cfg.CONTROLLER_TYPES[kind].label, key="ctrl_kind",
    )
    if ss.ctrl_kind == cfg.SMCConfig.kind:
        st.markdown("**Guidance (outer loop)**")
        st.caption(r"Lookahead point "
                   r"$\boldsymbol{p}_s = \boldsymbol{p} + L\,[\cos\psi,\ \sin\psi]^\top$")
        equations(
            r"\boldsymbol{e}_s &= \boldsymbol{p}_{s,ref} - \boldsymbol{p}_s",
            r"\boldsymbol{v}_{cmd} &= \dot{\boldsymbol{p}}_{s,ref} + k_{pos}\,\boldsymbol{e}_s",
            r"u_{cmd} &= \lVert \boldsymbol{v}_{cmd} \rVert",
            r"\psi_{cmd} &= \operatorname{atan2}(v_{cmd,y},\, v_{cmd,x})",
            r"r_{cmd} &= \dot{\psi}_{cmd} + k_{\psi}\, e_{\psi}",
        )
        st.caption(r"with $e_{\psi} = \operatorname{wrap}(\psi_{cmd} - \psi)$")
        num(r"Lookahead distance $L$ [m]", "smc_lookahead_L", step=0.1,
            min_value=0.0,
            help="Distance of the controlled point ahead of the hull centre.")
        c1, c2 = st.columns(2)
        with c1:
            num(r"$k_{pos}$ [1/s]", "smc_k_pos", step=0.05, min_value=0.0,
                help="Position-error feedback gain in the outer-loop surge "
                     "command.")
        with c2:
            num(r"$k_{\psi}$ [1/s]", "smc_k_psi", step=0.1, min_value=0.0,
                help="Heading-error feedback gain in the commanded yaw rate.")

        st.markdown("**Sliding surfaces and reaching law**")
        equations(
            r"s_u &= u - u_{cmd}, \qquad s_r = r - r_{cmd}",
            r"\dot{u}_{des} &= \dot{u}_{cmd} - \lambda_u s_u",
            r"&\quad - k_{s,u}\,\mathrm{sat}(s_u/\phi_u)",
            r"\dot{r}_{des} &= -\lambda_r s_r - k_{s,r}\,\mathrm{sat}(s_r/\phi_r)",
        )
        st.caption(r"with $\mathrm{sat}(x) = \min(1, \max(-1, x))$")
        c1, c2 = st.columns(2)
        with c1:
            num(r"$\lambda_u$ [1/s]", "smc_lambda_u", step=0.1, min_value=0.0,
                help="Linear reaching gain for the surge sliding error.")
        with c2:
            num(r"$\lambda_r$ [1/s]", "smc_lambda_r", step=0.1, min_value=0.0,
                help="Linear reaching gain for the yaw-rate sliding error.")
        c1, c2 = st.columns(2)
        with c1:
            num(r"$k_{s,u}$ [m/s$^2$]", "smc_k_s_u", step=0.05, min_value=0.0,
                help="Saturated switching gain for surge acceleration.")
        with c2:
            num(r"$k_{s,r}$ [rad/s$^2$]", "smc_k_s_r", step=0.05, min_value=0.0,
                help="Saturated switching gain for yaw acceleration.")
        c1, c2 = st.columns(2)
        with c1:
            num(r"$\phi_u$ [m/s]", "smc_phi_u", step=0.01, min_value=0.0,
                help="Surge boundary-layer half-width (m/s). Must be positive.")
        with c2:
            num(r"$\phi_r$ [rad/s]", "smc_phi_r", step=0.01, min_value=0.0,
                help="Yaw-rate boundary-layer half-width (rad/s). Must be "
                     "positive.")

        st.markdown("**Inversion of the identified dynamics**")
        equations(
            r"T_u &= z_4 u + z_5 |u| u + z_1 (\dot{u}_{des} - v r)",
            r"T_r &= z_8 r + z_9 |r| r + z_3\, \dot{r}_{des}",
        )
        st.caption(r"$T_v = 0$ (no lateral actuation). $T_u$, $T_r$ are then "
                   "allocated to the four thrusters and saturated.")
    else:
        st.caption("Existing `tau_figure_eight` law (tracks the hull centre):")
        equations(
            r"\boldsymbol{e} &= \boldsymbol{p}_{ref} - \boldsymbol{p}",
            r"\boldsymbol{v}_{cmd} &= \dot{\boldsymbol{p}}_{ref} + k_{pos}\,\boldsymbol{e}",
            r"u_{cmd} &= \lVert \boldsymbol{v}_{cmd} \rVert",
            r"\psi_{cmd} &= \operatorname{atan2}(v_{cmd,y},\, v_{cmd,x})",
            r"r_{cmd} &= \dot{\psi}_{ref} + k_{\psi}\, e_{\psi}",
            r"T_u &= z_4 u + z_5 |u| u + k_u (u_{cmd} - u)",
            r"T_r &= z_8 r + z_9 |r| r + k_r (r_{cmd} - r)",
        )
        st.caption(r"with $e_{\psi} = \operatorname{wrap}(\psi_{cmd} - \psi)$")
        c1, c2 = st.columns(2)
        with c1:
            num(r"$k_{pos}$ [1/s]", "fb_k_pos", step=0.05, min_value=0.0,
                help="Position-error feedback gain in the commanded velocity.")
            num(r"$k_{\psi}$ [1/s]", "fb_k_psi", step=0.1, min_value=0.0,
                help="Heading-error feedback gain in the commanded yaw rate.")
        with c2:
            num(r"$k_u$ [N s/m]", "fb_k_u", step=5.0, min_value=0.0,
                help="Proportional gain on the surge-speed error.")
            num(r"$k_r$ [N m s/rad]", "fb_k_r", step=5.0, min_value=0.0,
                help="Proportional gain on the yaw-rate error.")


AXIS_TITLES = (("u", r"Surge $\delta T_u$", "N"), ("v", r"Sway $\delta T_v$", "N"),
               ("r", r"Yaw $\delta T_r$", "N m"))


def _axis_inputs(prefix, fields):
    """Inputs per axis; fields = [(suffix, symbol, unit or None, step, min)]."""
    for axis, title, unit in AXIS_TITLES:
        st.markdown(f"*{title}*")
        # Two inputs per row: the sidebar is too narrow for more.
        for start in range(0, len(fields), 2):
            columns = st.columns(2)
            for column, (suffix, symbol, field_unit, step, minimum) in zip(
                columns, fields[start:start + 2]
            ):
                with column:
                    num(f"${symbol}_{axis}$ [{field_unit or unit}]",
                        f"{prefix}_{suffix}_{axis}", step=step,
                        min_value=minimum)


def sidebar_disturbance():
    ss = st.session_state
    st.selectbox(
        "Disturbance model", options=list(cfg.DISTURBANCE_TYPES),
        format_func=lambda kind: cfg.DISTURBANCE_TYPES[kind].label, key="dist_kind",
        help="Models implemented in `disturbances.py`.",
    )
    st.caption(
        r"Additive generalized forces "
        r"$\boldsymbol{\delta} = [\delta T_u,\ \delta T_v,\ \delta T_r]^\top$ "
        r"in the body frame: $\boldsymbol{\tau} = \boldsymbol{\tau}_{delivered} "
        r"+ \boldsymbol{\delta}$ (added after thruster saturation)."
    )
    kind = ss.dist_kind
    if kind == cfg.ConstantDisturbance.kind:
        num(r"Surge $\delta T_u$ [N]", "dist_const_dTu", step=1.0)
        num(r"Sway $\delta T_v$ [N]", "dist_const_dTv", step=1.0)
        num(r"Yaw $\delta T_r$ [N m]", "dist_const_dTr", step=1.0)
    elif kind == cfg.SinusoidalDisturbance.kind:
        equations(r"\delta T_i(t) &= b_i + A_i \sin(2\pi f_i t + \varphi_i)",
                  r"i &\in \{u,\ v,\ r\}")
        _axis_inputs("dist_sin", [
            ("amp", "A", None, 0.5, None),
            ("freq", "f", "Hz", 0.005, 0.0),
            ("phase", r"\varphi", "rad", 0.1, None),
            ("bias", "b", None, 0.5, None),
        ])
    elif kind == cfg.DirectionalDisturbance.kind:
        st.caption(r"Fixed force $F$ in the inertial frame at angle $\beta$ from "
                   r"$+x$, projected onto the body axes:")
        equations(r"\delta T_u &= F\cos(\beta - \psi)",
                  r"\delta T_v &= F\sin(\beta - \psi), \qquad \delta T_r = 0")
        num(r"Magnitude $F$ [N]", "dist_dir_magnitude_N", step=1.0,
            min_value=0.0)
        num(r"Direction $\beta$ [deg from $+x$]", "dist_dir_direction_deg",
            step=5.0)
    elif kind == cfg.WindLikeDisturbance.kind:
        st.caption("Fixed profile of `wind_like_disturbance` (no parameters):")
        equations(
            r"\delta T_u &= 2\sin(2\pi\,0.035\,t)",
            r"\delta T_v &= 3 + 4\sin(2\pi\,0.05\,t + 0.8)",
            r"\delta T_r &= 2.5\sin(2\pi\,0.025\,t + 1.4)",
        )
    elif kind == cfg.CombinedDisturbance.kind:
        equations(r"\boldsymbol{\delta} &= \boldsymbol{\delta}_{wind} + "
                  r"\boldsymbol{\delta}_{dir} + \boldsymbol{\delta}_{sin}")
        st.caption("Defaults = `combined` mode of `run_simulation.py`.")
        st.checkbox(r"Include wind-like component $\boldsymbol{\delta}_{wind}$",
                    key="dist_comb_wind")
        st.markdown(r"**Directional component** $\boldsymbol{\delta}_{dir}$")
        c1, c2 = st.columns(2)
        with c1:
            num(r"$F$ [N]", "dist_comb_dir_magnitude_N", step=1.0,
                min_value=0.0)
        with c2:
            num(r"$\beta$ [deg]", "dist_comb_dir_direction_deg", step=5.0)
        st.markdown(r"**Sinusoidal component** $\boldsymbol{\delta}_{sin}$")
        _axis_inputs("dist_comb", [
            ("amp", "A", None, 0.5, None),
            ("freq", "f", "Hz", 0.005, 0.0),
            ("phase", r"\varphi", "rad", 0.1, None),
        ])


def render_sidebar(check):
    with st.sidebar:
        st.header("IAcquaBot USV simulator")
        st.text_input("Run label (optional)", key="run_label",
                      placeholder="e.g. SMC k_s,r = 1.2, constant yaw dist.")
        st.button(
            "Run simulation", type="primary", icon=":material/play_arrow:",
            width="stretch", disabled=not check.report.ok,
            on_click=on_run_clicked,
            help="Runs the simulator with a snapshot of the current settings.",
        )
        status_box = st.container()
        st.button("Reset all settings to defaults", width="stretch",
                  on_click=on_reset_defaults)

        with st.expander("1 - Simulation settings", expanded=True):
            sidebar_simulation(check)
        with st.expander("2 - Reference trajectory", expanded=True):
            sidebar_reference(check)
        with st.expander("3 - Controller", expanded=False):
            sidebar_controller()
        with st.expander("4 - Disturbance", expanded=False):
            sidebar_disturbance()
        st.caption("Obstacles, CBF filtering and the reference governor are "
                   "disabled in the dashboard.")
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
                        f"{latest.title}. Press **Run simulation** to update.",
                        icon=":material/pending:")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def execute_pending_run(check, history):
    request = st.session_state.pending_run
    st.session_state.pending_run = None
    if request is None:
        return
    # Snapshot: the immutable config built from the widget values of this
    # script run. Nothing read later can modify it.
    snapshot = check.config
    if snapshot is None or not check.report.ok:
        st.session_state.last_failure = {
            "message": "Configuration is invalid: "
                       + "; ".join(check.report.errors),
            "config": snapshot,
        }
        return
    with st.spinner(
        f"Simulating {snapshot.simulation.t_final_s:g} s "
        f"({snapshot.controller.label}, {snapshot.disturbance.label})...",
        show_time=True,
    ):
        try:
            result = engine.run_from_config(
                snapshot, max_wall_time_s=MAX_WALL_TIME_S
            )
        except (cfg.ConfigValidationError, engine.SimulationError) as exc:
            st.session_state.last_failure = {"message": str(exc),
                                             "config": snapshot}
            return
        except Exception as exc:  # noqa: BLE001 - report any simulator error
            st.session_state.last_failure = {
                "message": f"Unexpected error ({type(exc).__name__}): {exc}",
                "config": snapshot,
            }
            return
        # Stored before leaving the spinner so that an interrupted rerun
        # cannot lose a finished result.
        history.add(result, request["label"])
        st.session_state.last_failure = None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def describe(config):
    ref, sim = config.reference, config.simulation
    return (
        f"{config.controller.label} | disturbance: {config.disturbance.label} | "
        f"figure-eight $A$ = {ref.amplitude_m:g} m, $T$ = {ref.period_s:.4g} s | "
        rf"$T_{{sim}}$ = {sim.t_final_s:.4g} s, $\Delta t$ = {sim.dt_s:g} s"
    )


def diff_frame(differences, old_title, new_title):
    return pd.DataFrame(
        [(key, str(old), str(new)) for key, old, new in differences],
        columns=["Parameter", old_title, new_title],
    )


def render_header(record, check, history):
    result = record.result
    st.subheader(record.title)
    st.caption(
        f"{describe(record.config)}  \nComputed {result.created_at} in "
        f"{result.total_time_s:.1f} s (integration {result.solver_time_s:.1f} s)"
        f" - run id {result.run_id}"
    )
    stale, differences = staleness(record, check.config)
    if stale:
        st.warning(
            f"**The sidebar settings differ from the configuration of "
            f"{record.title}.** The results below were computed with that "
            "run's own configuration, not with the current sidebar values. "
            "Press **Run simulation** to simulate the current settings.",
            icon=":material/history:",
        )
        if differences:
            with st.expander(f"Show the {len(differences)} changed setting(s)"):
                st.dataframe(diff_frame(differences, record.title, "Sidebar now"),
                             hide_index=True)
    else:
        st.success("Displayed results correspond to the current sidebar "
                   "settings.", icon=":material/check_circle:")

    m = result.metrics
    previous = history.previous
    pm = previous.result.metrics if previous is not None else None

    def card(column, label, key, unit, fmt="{:.3f}"):
        delta = None
        if pm is not None:
            delta = fmt.format(m[key] - pm[key]) + f" {unit} vs #{previous.number}"
        column.metric(label, fmt.format(m[key]) + f" {unit}", delta=delta,
                      delta_color="inverse")

    cols = st.columns(5)
    card(cols[0], "Max lookahead error", "lookahead_err_max_m", "m")
    card(cols[1], "Final lookahead error", "lookahead_err_final_m", "m")
    card(cols[2], "Steady-state RMS error", "lookahead_err_ss_rms_m", "m")
    card(cols[3], "Max hull-centre error", "center_err_max_m", "m")
    card(cols[4], "Thruster saturation", "saturation_any_pct", "%", "{:.1f}")


def _format_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "-"
    if value.is_integer() and abs(value) >= 1000:
        return f"{value:.0f}"
    return f"{value:.4g}"


def format_metrics_frame(frame, value_columns):
    """Numbers as compact strings (4 significant digits, '-' if undefined)."""
    frame = frame.copy()
    for column in value_columns:
        frame[column] = frame[column].map(_format_number)
    return frame


def render_metrics_tab(record):
    result = record.result
    tolerance = st.number_input(
        r"Tracking tolerance $\varepsilon$ for the success indicators [m]",
        key="eval_tolerance_m",
        value=met.DEFAULT_TOLERANCE_M, min_value=0.0, step=0.05, format="%g",
        help="Evaluation only: changing it does not require a new run.",
    )
    indicators = met.evaluate_indicators(result.metrics, tolerance)
    st.dataframe(pd.DataFrame([
        {"Status": "PASS" if row["passed"] else "FAIL",
         "Indicator": row["indicator"], "Value": row["detail"]}
        for row in indicators
    ]), hide_index=True)
    st.dataframe(format_metrics_frame(met.metrics_table(result.metrics),
                                      ["Value"]),
                 hide_index=True, height=min(850, 36 * 24))
    if result.warnings:
        st.caption("Validation warnings at run time: " + " | ".join(result.warnings))
    with st.expander("Metric definitions", icon=":material/functions:"):
        render_metric_definitions(result.config)


def render_metric_definitions(config):
    st.markdown("**Position errors** (lookahead point and hull centre)")
    equations(
        r"\boldsymbol{p}_s &= \boldsymbol{p} + L\,[\cos\psi,\ \sin\psi]^\top,"
        rf"\qquad L = {config.evaluation_lookahead_L:g}\ \mathrm{{m}}",
        r"e_s(t) &= \lVert \boldsymbol{p}_{s,ref}(t) - \boldsymbol{p}_s(t) \rVert,"
        r"\qquad e_v(t) = \lVert \dot{\boldsymbol{p}}_{s,ref}(t)"
        r" - \dot{\boldsymbol{p}}_s(t) \rVert",
        r"e_c(t) &= \lVert \boldsymbol{p}_{ref}(t) - \boldsymbol{p}(t) \rVert",
    )
    fraction = met.STEADY_STATE_FRACTION
    st.markdown(r"**Statistics** of $e \in \{e_s, e_c\}$ over the $N$ output "
                f"samples; *steady state* = final {fraction:.0%} of the run")
    equations(
        r"e_{max} &= \max_k e_k, \qquad e_{final} = e(T_{sim}),"
        r"\qquad \mathrm{RMS}(e) = \sqrt{\tfrac{1}{N}\textstyle\sum_k e_k^2}",
        rf"\Omega_{{ss}} &= \{{\, t_k \ge {1 - fraction:g}\, T_{{sim}} \,\}}",
    )
    st.markdown("**Heading and speed**")
    equations(
        r"e_{\psi} &= \operatorname{wrap}(\psi_{ref} - \psi), \qquad "
        r"\psi_{ref} = \operatorname{atan2}(\dot{y}_{ref},\, \dot{x}_{ref})",
        r"e_u &= \lVert \dot{\boldsymbol{p}}_{ref} \rVert - u",
    )
    st.markdown("**Control effort and chattering indicator** (delivered "
                r"$T_u$, $T_r$; same for $T_r$)")
    equations(
        r"\mathrm{RMS}(T_u) &= \sqrt{\tfrac{1}{N}\textstyle\sum_k T_{u,k}^2},"
        r"\qquad \mathrm{TV}(T_u) = \tfrac{1}{T_{sim}} \textstyle\sum_k"
        r" \lvert T_{u,k+1} - T_{u,k} \rvert",
    )
    st.markdown("**Saturation** - thruster $i$ is saturated when")
    equations(
        r"F_i &\in \{-F_{rev},\ F_{fwd}\} = "
        rf"\{{-{THRUSTER_REV_MAX:g},\ {THRUSTER_FWD_MAX:g}\}}\ \mathrm{{N}}"
    )
    st.markdown(r"**Indicators**: $e_s(T_{sim}) \le \varepsilon$, "
                r"$\max_{\Omega_{ss}} e_s \le \varepsilon$, and saturation "
                f"in less than {met.SATURATION_WARNING_PCT:g} % of the samples.")


def sync_comparison_defaults(history):
    latest = history.latest
    numbers = [r.number for r in history.records]
    if st.session_state.get("cmp_seen_latest") != latest.number or \
            st.session_state.get("cmp_a") not in numbers or \
            st.session_state.get("cmp_b") not in numbers:
        st.session_state.cmp_a = history.previous.number
        st.session_state.cmp_b = latest.number
        st.session_state.cmp_seen_latest = latest.number


def render_comparison_tab(history):
    if len(history) < 2:
        st.info("Run at least two simulations to compare them. Every completed "
                "run is kept (with its own configuration) for this browser "
                f"session, up to {history.max_runs} runs.")
        return
    sync_comparison_defaults(history)
    titles = {r.number: r.title for r in history.records}
    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("Run A (baseline)", options=list(titles),
                         format_func=titles.get, key="cmp_a")
    with c2:
        b = st.selectbox("Run B", options=list(titles), format_func=titles.get,
                         key="cmp_b")
    record_a, record_b = history.get(a), history.get(b)
    differences = cfg.config_diff(record_a.config, record_b.config)
    if a == b:
        st.info("Select two different runs.")
    elif not differences:
        st.info("Both runs used the same configuration (the simulator is "
                "deterministic, so their results should coincide).")
    else:
        st.markdown("**Configuration differences**")
        st.dataframe(diff_frame(differences, "Run A", "Run B"), hide_index=True)
    if (record_a.config.evaluation_lookahead_L
            != record_b.config.evaluation_lookahead_L):
        st.caption("The runs use different lookahead distances $L$: compare the "
                   "hull-centre errors rather than the lookahead-point errors.")
    table = met.comparison_table(record_a.result.metrics, record_b.result.metrics)
    st.markdown("**Performance metrics** ($B - A < 0$ means run B has a smaller "
                "value)")
    st.dataframe(format_metrics_frame(table, ["Run A", "Run B", "B - A",
                                              "Change %"]),
                 hide_index=True, height=min(850, 36 * 24))
    st.plotly_chart(figures.compare_xy(record_a, record_b))
    st.plotly_chart(figures.compare_time_series(record_a, record_b))


def render_details_tab(record, history):
    result = record.result
    st.markdown(f"**Configuration of {record.title}** (run id {result.run_id})")
    flat = result.config.flat()
    st.dataframe(pd.DataFrame({"Parameter": list(flat),
                               "Value": [str(v) for v in flat.values()]}),
                 hide_index=True)
    c1, c2 = st.columns(2)
    stem = f"usv_run{record.number}_{result.run_id}"
    c1.download_button("Download configuration (JSON)",
                       data=result.config.to_json(), file_name=f"{stem}.json",
                       mime="application/json", on_click="ignore",
                       icon=":material/download:")
    c2.download_button("Download time series (CSV)", data=result.to_csv_bytes,
                       file_name=f"{stem}.csv", mime="text/csv",
                       on_click="ignore", icon=":material/download:",
                       help="Same columns as run_simulation.save_csv, plus "
                            "disturbance and tracking-error columns.")
    with st.expander("Reproduce this run from Python"):
        st.code(
            "from dashboard.config import SimulationConfig\n"
            "from dashboard.engine import run_from_config\n\n"
            f"config = SimulationConfig.from_json(open('{stem}.json').read())\n"
            "result = run_from_config(config)\n"
            "print(result.metrics)\n",
            language="python",
        )

    st.divider()
    st.markdown("**Stored runs in this session**")
    rows = []
    for r in reversed(history.records):
        rows.append({
            "Run": r.title, "Controller": r.config.controller.kind,
            "Disturbance": r.config.disturbance.kind,
            "Duration [s]": r.config.simulation.t_final_s,
            "Max lookahead err. [m]": r.result.metrics["lookahead_err_max_m"],
            "Final lookahead err. [m]": r.result.metrics["lookahead_err_final_m"],
            "Computed": r.result.created_at,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    titles = {r.number: r.title for r in history.records}
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        number = st.selectbox("Stored run", options=list(reversed(titles)),
                              format_func=titles.get, key="history_pick",
                              label_visibility="collapsed")
    with c2:
        st.button("Load its settings into the sidebar",
                  on_click=on_load_run_config, args=(number,),
                  icon=":material/upload:")
    with c3:
        st.button("Clear stored runs", on_click=lambda: history.clear(),
                  icon=":material/delete:")


def render_xy_tab(record, history):
    """XY plot of the displayed run, with the previous run in light gray.

    Uses the existing history (``history.previous`` is the run executed just
    before the displayed one); only these two runs are drawn.
    """
    previous = history.previous
    show_previous = False
    if previous is not None:
        show_previous = st.checkbox(
            f"Overlay the previous run ({previous.title}) in light gray",
            value=True, key="xy_show_previous",
            help="Only the previous run is overlaid. Use *Compare runs* to "
                 "compare any two stored runs.",
        )
    st.plotly_chart(figures.xy_trajectory(
        record.result, previous=previous if show_previous else None))
    st.caption("Lookahead point and lookahead reference can be shown by "
               "clicking their legend entries.")
    if previous is not None:
        render_previous_summary(previous, record)


def render_previous_summary(previous, current):
    """Parameters and main metrics of the previous run next to the current."""
    st.markdown(f"**Previous run: {previous.title}** "
                f"(computed {previous.result.created_at})")
    st.caption(describe(previous.config))
    left, right = st.columns([2, 3])
    with left:
        differences = cfg.config_diff(previous.config, current.config)
        if differences:
            st.markdown(f"Parameters changed (#{previous.number} → "
                        f"#{current.number})")
            st.dataframe(diff_frame(differences, "Previous", "Current"),
                         hide_index=True)
        else:
            st.markdown("Same configuration as the current run.")
    with right:
        table = met.comparison_table(previous.result.metrics,
                                     current.result.metrics,
                                     keys=met.KEY_METRICS)
        # Compact: unit merged into the metric name.
        table["Metric"] = table["Metric"] + " [" + table["Unit"] + "]"
        table = table.drop(columns="Unit").rename(columns={
            "Run A": f"Previous #{previous.number}",
            "Run B": f"Current #{current.number}", "B - A": "Difference"})
        st.markdown("Main metrics")
        st.dataframe(format_metrics_frame(
            table, [f"Previous #{previous.number}", f"Current #{current.number}",
                    "Difference", "Change %"]),
            hide_index=True)


def render_results(history, check):
    failure = st.session_state.last_failure
    latest = history.latest
    if failure is not None:
        shown = (f" The results below still belong to {latest.title}."
                 if latest is not None else "")
        st.error(f"**The last run attempt failed; no new results were "
                 f"produced.** {failure['message']}{shown}",
                 icon=":material/error:")

    if latest is None:
        st.info(
            "Configure the experiment in the sidebar and press **Run "
            "simulation**. Widgets never start a simulation by themselves; "
            "each run uses a frozen snapshot of the settings, and its results "
            "stay tied to that snapshot."
        )
        return

    render_header(latest, check, history)
    tabs = st.tabs(["XY trajectory", "Tracking errors", "Vessel states",
                    "Control signals", "Disturbance", "Metrics",
                    "Compare runs", "Run details & export"])
    result = latest.result
    with tabs[0]:
        render_xy_tab(latest, history)
    with tabs[1]:
        st.plotly_chart(figures.tracking_errors(result))
    with tabs[2]:
        st.plotly_chart(figures.vessel_states(result))
    with tabs[3]:
        st.plotly_chart(figures.control_signals(result))
        st.caption("Commanded = output of the control law; delivered = after "
                   "per-thruster saturation in allocate_thrusters. The "
                   "allocation gives $F_1 = F_2$ (left) and $F_3 = F_4$ (right), so "
                   "those traces overlap.")
    with tabs[4]:
        st.plotly_chart(figures.disturbance_signals(result))
    with tabs[5]:
        render_metrics_tab(latest)
    with tabs[6]:
        render_comparison_tab(history)
    with tabs[7]:
        render_details_tab(latest, history)
    if len(result.data) > figures.MAX_PLOT_POINTS:
        st.caption(f"Plots are decimated to about {figures.MAX_PLOT_POINTS} "
                   "points per signal; metrics use all "
                   f"{len(result.data)} samples.")


def main():
    init_session_state()
    check = cfg.check_form(current_form())
    history = st.session_state.history

    status_box = render_sidebar(check)
    st.title("IAcquaBot USV - trajectory tracking under disturbances")
    # When a run starts, the previous results are removed from the screen
    # (not merely faded) so they cannot be mistaken for the new run's results.
    progress_area = st.container()
    results_slot = st.empty()
    if st.session_state.pending_run is not None:
        results_slot.empty()
        with progress_area:
            execute_pending_run(check, history)
    render_status(status_box, check, history)
    with results_slot.container():
        render_results(history, check)


main()
