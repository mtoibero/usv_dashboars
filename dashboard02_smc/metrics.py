"""Performance indices of one run (no Streamlit imports).

All indices are computed from the stored samples (every controller sample in
the digital implementation, every output sample in the continuous one).
Integrals use the trapezoidal rule.  Each index has a LaTeX definition in
``INDEX_INFO`` that the dashboard displays.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from dashboard02_smc import PROJECT_ROOT  # noqa: F401

from analysis_utils import is_saturated
from usv_model import THRUSTER_FWD_MAX, THRUSTER_REV_MAX

STEADY_STATE_FRACTION = 0.25   # final fraction of the run = "steady state"
HF_WINDOW_S = 1.0              # moving-average window separating chattering
SATURATION_TOL_N = 1e-6


def _trapz(y, t):
    return float(np.trapezoid(y, t)) if len(t) > 1 else float("nan")


def _rms(y):
    y = np.asarray(y, dtype=float)
    return float(np.sqrt(np.mean(y**2))) if y.size else float("nan")


def steady_state_mask(t, fraction=STEADY_STATE_FRACTION):
    t = np.asarray(t, dtype=float)
    return t >= t[-1] - fraction * (t[-1] - t[0])


def error_integrals(t, e):
    """IAE, ISE and ITAE of a signal."""
    e = np.abs(np.asarray(e, dtype=float))
    t = np.asarray(t, dtype=float)
    return _trapz(e, t), _trapz(e**2, t), _trapz((t - t[0]) * e, t)


def reaching(t, s, phi):
    """(reaching time, % of samples inside the layer after reaching).

    Reaching time = first instant with |s| <= phi (nan if never).
    """
    inside = np.abs(np.asarray(s, dtype=float)) <= phi
    if not inside.any():
        return float("nan"), 0.0
    first = int(np.argmax(inside))
    return float(t[first] - t[0]), 100.0 * float(np.mean(inside[first:]))


def total_variation_rate(y, duration):
    y = np.asarray(y, dtype=float)
    if y.size < 2 or duration <= 0.0:
        return float("nan")
    return float(np.sum(np.abs(np.diff(y))) / duration)


def high_frequency_component(t, y, window_s=HF_WINDOW_S):
    """y minus its centred moving average over ``window_s``."""
    y = np.asarray(y, dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    size = max(3, int(round(window_s / dt)) | 1)   # odd number of samples
    if size >= y.size:
        return y - y.mean()
    return y - uniform_filter1d(y, size=size, mode="nearest")


def zero_crossing_rate(y, duration):
    """Sign changes per second (exact zeros are skipped)."""
    y = np.asarray(y, dtype=float)
    signs = np.sign(y[y != 0.0])
    if signs.size < 2 or duration <= 0.0:
        return 0.0
    return float(np.count_nonzero(np.diff(signs)) / duration)


def compute_metrics(df, config, steady_state_fraction=STEADY_STATE_FRACTION):
    """Scalar performance indices of one run."""
    t = df["t"].to_numpy()
    ss = steady_state_mask(t, steady_state_fraction)
    m = {}

    e_p = df["e_p"].to_numpy()
    iae, ise, itae = error_integrals(t, e_p)
    m.update(
        e_p_rms=_rms(e_p), e_p_max=float(np.max(e_p)), e_p_iae=iae,
        e_p_itae=itae, e_p_ss_rms=_rms(e_p[ss]), e_p_final=float(e_p[-1]),
        e_c_rms=_rms(df["e_c"]), e_c_max=float(df["e_c"].max()),
    )

    for tag, e_name, s_name, phi in (
        ("u", "e_u", "s_u", config.surge.phi_u),
        ("psi", "e_psi", "s_r", config.yaw.phi_r),
    ):
        e = df[e_name].to_numpy()
        s = df[s_name].to_numpy()
        iae, ise, itae = error_integrals(t, e)
        s_tag = "u" if tag == "u" else "r"
        t_reach, in_layer = reaching(t, s, phi)
        m.update({
            f"e_{tag}_rms": _rms(e),
            f"e_{tag}_max": float(np.max(np.abs(e))),
            f"e_{tag}_iae": iae,
            f"e_{tag}_ise": ise,
            f"e_{tag}_itae": itae,
            f"e_{tag}_ss_rms": _rms(e[ss]),
            f"s_{s_tag}_rms": _rms(s),
            f"s_{s_tag}_ss_rms": _rms(s[ss]),
            f"t_reach_{s_tag}": t_reach,
            f"in_layer_{s_tag}_pct": in_layer,
            f"s_{s_tag}_zc": zero_crossing_rate(s[ss], float(
                t[ss][-1] - t[ss][0])),
        })

    # Chattering indices are evaluated over the steady-state window, so the
    # reaching transient (e.g. an initial offset) is not counted as chattering.
    t_ss = t[ss]
    duration_ss = float(t_ss[-1] - t_ss[0]) if t_ss.size > 1 else 0.0
    for name, sat_name in (("Tu", "sat_u"), ("Tr", "sat_r")):
        y = df[name].to_numpy()
        m[f"{name}_rms"] = _rms(y)
        m[f"{name}_tv"] = total_variation_rate(y[ss], duration_ss)
        m[f"{name}_hf_rms"] = _rms(high_frequency_component(t, y)[ss])
        m[f"{sat_name}_tv"] = total_variation_rate(df[sat_name].to_numpy()[ss],
                                                   duration_ss)

    forces = df[["F1", "F2", "F3", "F4"]].to_numpy()
    saturated = is_saturated(forces, THRUSTER_FWD_MAX, -THRUSTER_REV_MAX,
                             SATURATION_TOL_N)
    m["thruster_sat_pct"] = 100.0 * float(np.mean(saturated.any(axis=1)))
    m["t_final_s"] = float(t[-1])
    m["samples"] = int(len(t))
    return m


# ---------------------------------------------------------------------------
# Index catalogue: key -> (group, LaTeX symbol, label, unit, better, LaTeX
# definition).  ``better`` is "lower" or "higher" (None: informative only).
# ---------------------------------------------------------------------------
_SS = r"\Omega_{ss}"
INDEX_INFO = {
    # -- position --------------------------------------------------------
    "e_p_rms": ("Position", r"\mathrm{RMS}(\lVert e_p \rVert)",
                "RMS lookahead position error", "m", "lower",
                r"\sqrt{\tfrac{1}{N}\sum_k \lVert \boldsymbol{p}_{s,k} - "
                r"\boldsymbol{p}_{s,ref,k} \rVert^2}"),
    "e_p_max": ("Position", r"\max \lVert e_p \rVert",
                "Max lookahead position error", "m", "lower",
                r"\max_k \lVert \boldsymbol{p}_{s,k} - \boldsymbol{p}_{s,ref,k}"
                r" \rVert"),
    "e_p_iae": ("Position", r"\mathrm{IAE}(\lVert e_p \rVert)",
                "IAE lookahead position error", "m s", "lower",
                r"\int_0^{T_{sim}} \lVert e_p \rVert\, dt"),
    "e_p_itae": ("Position", r"\mathrm{ITAE}(\lVert e_p \rVert)",
                 "ITAE lookahead position error", "m s²", "lower",
                 r"\int_0^{T_{sim}} t\,\lVert e_p \rVert\, dt"),
    "e_p_ss_rms": ("Position", r"\mathrm{RMS}_{ss}(\lVert e_p \rVert)",
                   "Steady-state RMS lookahead error", "m", "lower",
                   rf"\mathrm{{RMS}}\ \text{{over}}\ {_SS}"),
    "e_p_final": ("Position", r"\lVert e_p(T_{sim}) \rVert",
                  "Final lookahead position error", "m", "lower",
                  r"\lVert e_p \rVert\ \text{at the final sample}"),
    "e_c_rms": ("Position", r"\mathrm{RMS}(\lVert e_c \rVert)",
                "RMS hull-centre error", "m", "lower",
                r"\sqrt{\tfrac{1}{N}\sum_k \lVert \boldsymbol{p}_k - "
                r"\boldsymbol{p}_{ref,k} \rVert^2}"),
    "e_c_max": ("Position", r"\max \lVert e_c \rVert",
                "Max hull-centre error", "m", "lower",
                r"\max_k \lVert \boldsymbol{p}_k - \boldsymbol{p}_{ref,k} \rVert"),
}
for _tag, _grp, _sym, _sym_k, _unit, _s, _phi in (
    ("u", "Surge SMC", "e_u", "e_{u,k}", "m", "u", r"\phi_u"),
    ("psi", "Yaw SMC", r"e_{\psi}", r"e_{\psi,k}", "rad", "r", r"\phi_r"),
):
    _s_unit = "m/s" if _s == "u" else "rad/s"
    _ch = _grp.split()[0]
    _what = ("along-course error e_u" if _s == "u" else "heading error e_ψ")
    INDEX_INFO.update({
        f"e_{_tag}_rms": (_grp, rf"\mathrm{{RMS}}({_sym})", f"{_ch}: RMS {_what}",
                          _unit, "lower",
                          rf"\sqrt{{\tfrac{{1}}{{N}}\sum_k {_sym_k}^2}}"),
        f"e_{_tag}_max": (_grp, rf"\max|{_sym}|", f"{_ch}: max |{_what}|", _unit,
                          "lower", rf"\max_k |{_sym_k}|"),
        f"e_{_tag}_iae": (_grp, rf"\mathrm{{IAE}}({_sym})", f"{_ch}: IAE {_what}",
                          f"{_unit} s", "lower",
                          rf"\int_0^{{T_{{sim}}}} |{_sym}|\, dt"),
        f"e_{_tag}_ise": (_grp, rf"\mathrm{{ISE}}({_sym})", f"{_ch}: ISE {_what}",
                          f"{_unit}² s", "lower",
                          rf"\int_0^{{T_{{sim}}}} {_sym}^2\, dt"),
        f"e_{_tag}_itae": (_grp, rf"\mathrm{{ITAE}}({_sym})", f"{_ch}: ITAE {_what}",
                           f"{_unit} s²", "lower",
                           rf"\int_0^{{T_{{sim}}}} t\,|{_sym}|\, dt"),
        f"e_{_tag}_ss_rms": (_grp, rf"\mathrm{{RMS}}_{{ss}}({_sym})",
                             f"{_ch}: steady-state RMS {_what}", _unit, "lower",
                             rf"\mathrm{{RMS}}({_sym})\ \text{{over}}\ {_SS}"),
        f"s_{_s}_rms": (_grp, rf"\mathrm{{RMS}}(s_{_s})",
                        f"{_ch}: RMS of s_{_s}", _s_unit, "lower",
                        rf"\sqrt{{\tfrac{{1}}{{N}}\sum_k s_{{{_s},k}}^2}}"),
        f"s_{_s}_ss_rms": (_grp, rf"\mathrm{{RMS}}_{{ss}}(s_{_s})",
                           f"{_ch}: steady-state RMS of s_{_s}", _s_unit,
                           "lower", rf"\mathrm{{RMS}}(s_{_s})\ \text{{over}}\ "
                           rf"{_SS}"),
        f"t_reach_{_s}": (_grp, rf"t_{{r,{_s}}}",
                          f"{_ch}: reaching time of |s_{_s}| ≤ φ_{_s}", "s",
                          "lower", rf"\min\{{t : |s_{_s}(t)| \le {_phi}\}}"),
        f"in_layer_{_s}_pct": (_grp, rf"\eta_{{BL,{_s}}}",
                               f"{_ch}: time in the boundary layer after reaching",
                               "%", "higher",
                               rf"100\,\frac{{\#\{{k : t_k \ge t_{{r,{_s}}},\ "
                               rf"|s_{{{_s},k}}| \le {_phi}\}}}}"
                               rf"{{\#\{{k : t_k \ge t_{{r,{_s}}}\}}}}"),
    })

for _name, _sym, _unit, _s, _phi in (
    ("Tu", "T_u", "N", "u", r"\phi_u"),
    ("Tr", "T_r", "N m", "r", r"\phi_r"),
):
    _ch = "Surge" if _s == "u" else "Yaw"
    INDEX_INFO.update({
        f"{_name}_tv": ("Chattering", rf"\mathrm{{TV}}({_sym})",
                        f"{_ch}: total-variation rate of {_sym}",
                        f"{_unit}/s", "lower",
                        rf"\frac{{1}}{{|\Omega_{{ss}}|}}\sum_{{k \in \Omega_{{ss}}}}"
                        rf" |{_sym}(t_{{k+1}}) - {_sym}(t_k)|"),
        f"{_name}_hf_rms": ("Chattering", rf"\mathrm{{HF}}({_sym})",
                            f"{_ch}: high-frequency RMS of {_sym}", _unit,
                            "lower",
                            rf"\mathrm{{RMS}}_{{\Omega_{{ss}}}}\left({_sym} - "
                            rf"\langle {_sym} \rangle_{{{HF_WINDOW_S:g}\,"
                            rf"\mathrm{{s}}}}\right)"),
        f"sat_{_s}_tv": ("Chattering",
                         rf"\mathrm{{TV}}(\mathrm{{sat}}(s_{_s}/{_phi}))",
                         f"{_ch}: switching activity of sat(s_{_s}/φ_{_s})",
                         "1/s", "lower",
                         rf"\frac{{1}}{{|\Omega_{{ss}}|}}\sum_{{k \in \Omega_{{ss}}}}"
                         rf" |\Delta\,\mathrm{{sat}}(s_{{{_s},k}}/{_phi})|"),
        f"s_{_s}_zc": ("Chattering", rf"\nu_{{0,{_s}}}",
                       f"{_ch}: zero-crossing rate of s_{_s}", "1/s", "lower",
                       rf"\frac{{1}}{{|\Omega_{{ss}}|}}\,\#\{{k \in "
                       rf"\Omega_{{ss}} : s_{{{_s},k}}\, s_{{{_s},k+1}} < 0\}}"),
        f"{_name}_rms": ("Effort", rf"\mathrm{{RMS}}({_sym})",
                         f"{_ch}: RMS delivered {_sym}", _unit, "lower",
                         rf"\sqrt{{\tfrac{{1}}{{N}}\sum_k {_sym}(t_k)^2}}"),
    })
INDEX_INFO["thruster_sat_pct"] = (
    "Effort", r"\eta_{sat}", "Time with a thruster saturated", "%", "lower",
    r"100\,\frac{\#\{k : \exists i,\ F_{i,k} \in \{-F_{rev}, F_{fwd}\}\}}{N}")

GROUPS = ("Position", "Surge SMC", "Yaw SMC", "Chattering", "Effort")

# Indices shown as metric cards and in the bar charts.
KEY_INDICES = (
    "e_p_rms", "e_p_max", "e_u_rms", "e_psi_rms",
    "in_layer_u_pct", "in_layer_r_pct", "t_reach_u", "t_reach_r",
    "Tu_tv", "Tr_tv", "Tu_hf_rms", "Tr_hf_rms",
    "sat_u_tv", "sat_r_tv", "s_u_zc", "s_r_zc",
    "Tu_rms", "Tr_rms", "thruster_sat_pct",
)


def index_table(current, previous=None):
    """DataFrame with one row per index (current, previous, change)."""
    rows = []
    for key, (group, _sym, label, unit, better, _defn) in INDEX_INFO.items():
        a = current.get(key, np.nan)
        row = {"Group": group, "Index": label, "Unit": unit, "Current": a}
        if previous is not None:
            b = previous.get(key, np.nan)
            row["Previous"] = b
            delta = a - b
            row["Change"] = delta
            row["Change %"] = (100.0 * delta / abs(b)
                               if b not in (0, 0.0) and math.isfinite(b)
                               else np.nan)
            row["Verdict"] = verdict(a, b, better)
        row["key"] = key
        rows.append(row)
    rows.sort(key=lambda r: GROUPS.index(r["Group"]))   # stable: keeps order
    return pd.DataFrame(rows)


def verdict(current, previous, better, rel_tol=0.01):
    """'better' / 'worse' / 'similar' for the current vs previous value.

    NaN means "never happened" (e.g. the boundary layer was never reached),
    which is the worst possible value of a "lower is better" index.
    """
    if better is None:
        return ""
    cur_ok, prev_ok = math.isfinite(current), math.isfinite(previous)
    if not cur_ok and not prev_ok:
        return "similar"
    if not cur_ok or not prev_ok:
        return "worse" if not cur_ok else "better"
    scale = max(abs(current), abs(previous), 1e-12)
    if abs(current - previous) <= rel_tol * scale:
        return "similar"
    improved = current < previous if better == "lower" else current > previous
    return "better" if improved else "worse"
