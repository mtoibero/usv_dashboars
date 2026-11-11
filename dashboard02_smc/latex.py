"""LaTeX snippets used by the dashboard (no Streamlit imports).

Streamlit renders ``$...$`` and ``$$...$$`` with KaTeX, so every equation
and parameter list shown in the app is built here as LaTeX source.
"""

from __future__ import annotations

import math

from dashboard02_smc import config as cfg

CURRENT_TEX_COLOR = "#0050ff"
PREVIOUS_TEX_COLOR = "#9a9a9a"
CHANGED_TEX_COLOR = "#e8590c"

def num(value, digits=4):
    """Number as LaTeX (4 significant digits, scientific as a x 10^b)."""
    if isinstance(value, str):
        return rf"\text{{{value}}}"
    value = float(value)
    if not math.isfinite(value):
        return r"\text{n/a}"
    if value == 0.0:
        return "0"
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
        return rf"{float(mantissa):g}\times 10^{{{int(exponent)}}}"
    return f"{value:.{digits}g}"


def unit(text):
    """Plain unit string ('m/s²', 'N m', '%') as upright LaTeX."""
    if not text or text == "-":
        return ""
    text = (text.replace("%", r"\%").replace("²", "^2")
            .replace(" ", r"\,"))
    return rf"\mathrm{{{text}}}"


def aligned(*lines, small=True):
    """Display block with aligned lines (use '&' as the alignment point)."""
    size = r"\small" if small else ""
    return "$$\n" + size + r"\begin{aligned}" + r" \\ ".join(lines) + \
        r"\end{aligned}" + "\n$$"


def _sinusoid(amplitude, frequency, phase, bias=0.0):
    body = rf"{num(amplitude)}\sin(2\pi\cdot {num(frequency)}\,t"
    if phase:
        sign = "+" if phase > 0 else "-"
        body += rf" {sign} {num(abs(phase))}"
    body += ")"
    if bias:
        body = rf"{num(bias)} + " + body
    return body


def disturbance_lines(dist):
    """LaTeX lines describing the disturbance model (with its parameters).

    Units: [N] for dTu, dTv and [N m] for dTr.
    """
    kind = dist.kind
    if kind == "none":
        return [r"\boldsymbol\delta = \boldsymbol 0"]
    if kind == "constant":
        d_u, d_v, d_r = (num(c) for c in dist.constant)
        return [rf"\boldsymbol\delta = [{d_u}\,\mathrm{{N}},\ {d_v}\,"
                rf"\mathrm{{N}},\ {d_r}\,\mathrm{{N\,m}}]^\top"]
    if kind == "wind":
        return [r"\boldsymbol\delta = \boldsymbol\delta_{wind}\ "
                r"\text{(fixed profile)}"]
    lines = []
    if kind == "combined":
        parts = ([r"\boldsymbol\delta_{wind}"] if dist.include_wind else []) + [
            r"\boldsymbol\delta_{dir}", r"\boldsymbol\delta_{sin}"]
        lines.append(r"\boldsymbol\delta = " + " + ".join(parts))
    if kind in ("directional", "combined"):
        lines.append(rf"F = {num(dist.dir_magnitude_N)}\,\mathrm{{N}},\ "
                     rf"\beta = {num(dist.dir_direction_deg)}^\circ")
    if kind in ("sinusoidal", "combined"):
        for i, axis in enumerate(cfg.AXES):
            amplitude, bias = dist.sin_amplitude[i], dist.sin_bias[i]
            if amplitude == 0.0 and bias == 0.0:
                continue
            unit_tex = r"\mathrm{N\,m}" if axis == "r" else r"\mathrm{N}"
            lines.append(rf"\delta T_{axis} = \left[" + _sinusoid(
                amplitude, dist.sin_frequency_hz[i], dist.sin_phase_rad[i],
                bias) + rf"\right]\,{unit_tex}")
        if kind == "sinusoidal" and len(lines) == 0:
            lines.append(r"\boldsymbol\delta = \boldsymbol 0")
    return lines


def _param(symbol, value, changed, unit_text=""):
    text = rf"{symbol} = {num(value)}"
    if unit(unit_text):
        text += rf"\,{unit(unit_text)}"
    if changed:
        return rf"\textcolor{{{CHANGED_TEX_COLOR}}}{{\boldsymbol{{{text}}}}}"
    return text


def run_card(config, changed=()):
    """Aligned LaTeX block with the parameters of a run.

    ``changed`` is a collection of flat keys (see ``SimulationConfig.flat``)
    highlighted in orange.
    """
    changed = set(changed)

    def p(key, value, unit_text=""):
        return _param(cfg.PARAM_INFO[key][0], value, key in changed, unit_text)

    s, y, ref, sim = config.surge, config.yaw, config.reference, config.simulation
    dist = config.disturbance
    ref_changed = bool({"reference.amplitude_m", "reference.period_s"} & changed)
    lines = [
        r"\text{Surge: }& " + ",\\ ".join([
            p("surge.lambda_u", s.lambda_u), p("surge.k_s_u", s.k_s_u),
            p("surge.phi_u", s.phi_u), p("surge.k_pos", s.k_pos)]),
        r"\text{Yaw: }& " + ",\\ ".join([
            p("yaw.lambda_r", y.lambda_r), p("yaw.k_s_r", y.k_s_r),
            p("yaw.phi_r", y.phi_r), p("yaw.k_psi", y.k_psi)]),
        r"\text{Reference: }& " + ",\\ ".join([
            _param("f", ref.frequency_hz, ref_changed, "Hz"),
            _param("v_{max}", ref.peak_speed_mps, ref_changed, "m/s")]),
        r"& " + ",\\ ".join([
            _param("A", ref.amplitude_m, ref_changed, "m"),
            p("reference.laps", ref.laps)]),
    ]
    dist_changed = any(key.startswith("disturbance.") for key in changed)
    dist_lines = [rf"\text{{{dist.label.split(' (')[0]}}}"] if dist.active \
        else []
    dist_lines += disturbance_lines(dist) if dist.active else [r"\text{none}"]
    if dist_changed:
        dist_lines = [rf"\textcolor{{{CHANGED_TEX_COLOR}}}{{{text}}}"
                      for text in dist_lines]
    lines.append(r"\text{Disturbance: }& " + dist_lines[0])
    lines.extend("& " + text for text in dist_lines[1:])
    if sim.mode == cfg.DIGITAL:
        impl = r"\text{digital (ZOH)},\ " + p("simulation.sample_rate_hz",
                                             sim.sample_rate_hz, "Hz")
    else:
        impl = r"\text{continuous},\ " + p("simulation.output_dt_s",
                                           sim.output_dt_s, "s")
    if "simulation.mode" in changed:
        impl = rf"\textcolor{{{CHANGED_TEX_COLOR}}}{{{impl}}}"
    impl += ",\\ " + p("simulation.lookahead_L_m", sim.lookahead_L_m, "m")
    lines.append(r"\text{Simulation: }& " + impl)
    lines.append(r"\text{Initial offset: }& " + ",\\ ".join([
        p("simulation.dx0_m", sim.dx0_m, "m"),
        p("simulation.dy0_m", sim.dy0_m, "m")]))
    lines.append("& " + ",\\ ".join([
        _param(r"\Delta\psi_0", math.degrees(sim.dpsi0_rad),
               "simulation.dpsi0_rad" in changed, "deg"),
        p("simulation.du0_mps", sim.du0_mps, "m/s")]))
    return aligned(*lines)


def change_list(differences):
    """Markdown bullet list of changed parameters: '$\\lambda_u$: 1.2 → 2'."""
    lines = []
    keys = {key for key, _old, _new in differences}
    for key, old, new in differences:
        # f_s = 0 only marks "continuous"; in digital mode dt = 1/f_s.
        if key == "simulation.sample_rate_hz" and 0.0 in (old, new):
            continue
        if key == "simulation.output_dt_s" and keys & {
                "simulation.sample_rate_hz", "simulation.mode"}:
            continue
        symbol, _html, unit_text, group = cfg.PARAM_INFO.get(
            key, (rf"\text{{{key}}}", key, "", ""))
        if key == "reference.period_s":   # shown as the frequency f = 1/T
            symbol, unit_text = "f", "Hz"
            old = 1.0 / old if isinstance(old, float) and old else old
            new = 1.0 / new if isinstance(new, float) and new else new
        unit_tex = rf"\ {unit(unit_text)}" if unit(unit_text) else ""
        prefix = f"{group}: " if group else ""
        lines.append(rf"- {prefix}${symbol}$: ${num(old)}{unit_tex} "
                     rf"\rightarrow {num(new)}{unit_tex}$")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gain hints (see tuning.py).
# ---------------------------------------------------------------------------
def gain_status(hint):
    """('robust' | 'marginal' | 'violated', icon)."""
    if not hint.sliding_ok:
        return "violated", "❌"
    if hint.k_s + 1e-12 < hint.k_suggested:
        return "marginal", "⚠️"
    return "robust", "✅"


def gain_hint_lines(hint, suggested):
    """Markdown (an aligned LaTeX block plus short lines) explaining the
    switching-gain estimate.  ``suggested`` is the rounded value offered by
    the app."""
    i, z = hint.axis, hint.z_symbol
    acc = unit(hint.acc_unit)
    k = rf"k_{{s,{i}}}"
    rows = [
        rf"\bar d_{i} &= \tfrac{{\max|\delta T_{i}|}}{{{z}}} + \bar d_{{traj,{i}}}",
        rf"&= {num(hint.d_ext, 3)} + {num(hint.d_traj, 3)} = "
        rf"{num(hint.d_bar, 3)}\ {acc}",
        rf"{k} &\ge \bar d_{i} - \lambda_{i}\phi_{i} = {num(hint.k_min, 3)}"
        r"\quad\text{(min.)}",
        rf"{k} &\approx 1.25\,\bar d_{i} = {num(suggested, 3)}"
        r"\quad\text{(suggested)}",
    ]
    k_max = hint.k_max_chattering
    if k_max is not None:
        rows.append(rf"{k} &< \phi_{i}(f_s - \lambda_{i}) = {num(k_max, 3)}"
                    r"\quad(\Gamma < 1)")
    lines = [aligned(*rows), ""]

    status, icon = gain_status(hint)
    current = rf"${k} = {num(hint.k_s, 3)}$"
    if status == "violated":
        text = (rf"{icon} Current {current}: **sliding condition violated**, "
                "the state leaves the boundary layer.")
        alternatives = []
        if math.isfinite(hint.lam_min):
            alternatives.append(rf"$\lambda_{i} \ge {num(hint.lam_min, 3)}$")
        if math.isfinite(hint.phi_min_sliding):
            alternatives.append(rf"$\phi_{i} \ge {num(hint.phi_min_sliding, 3)}$")
        if alternatives:
            text += " Or keep it and raise " + " or ".join(alternatives) + "."
    elif status == "marginal":
        text = (rf"{icon} Current {current}: **marginal**, no margin for model "
                "errors.")
    else:
        margin = hint.margin_pct
        margin_text = f"{margin:.0f}%" if math.isfinite(margin) else "∞"
        text = rf"{icon} Current {current}: **robust** (margin {margin_text})."
    if k_max is not None and not hint.chattering_ok:
        text += " ⚠️ It chatters at this $f_s$."
    lines.append(text)
    if k_max is not None and not hint.window_exists:
        lines.append(rf"⚠️ No robust, chattering-free ${k}$ at $f_s$ = "
                     rf"{num(hint.sample_rate_hz, 3)} Hz: widen "
                     rf"$\phi_{i} \ge {num(hint.phi_min_window, 3)}$ or raise "
                     "$f_s$.")
    force = unit(hint.force_unit)
    lines.append(rf"Tolerated now: $|\delta T_{i}| \le "
                 rf"{num(hint.tolerable_disturbance, 3)}\ {force}$.")
    return lines


def actuator_lines(check):
    pct = check.utilization_pct
    if pct >= 100.0:
        icon, text = "❌", ("in the demanding parts of the lap the thrusters "
                           "saturate and **sliding is interrupted whatever the "
                           "gains**. Reduce the speed or the disturbance.")
    elif pct >= 80.0:
        icon, text = "⚠️", ("little headroom for the switching action: thruster "
                           "saturation is likely.")
    else:
        icon, text = "✅", "enough headroom for the switching action."
    return [rf"{icon} Thrusters: worst case ${pct:.0f}\%$ of the limit "
            rf"(trajectory + worst-sign disturbance); {text}"]
