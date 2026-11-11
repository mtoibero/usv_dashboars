"""Experiment configuration of the SMC dashboard (no Streamlit imports).

Two representations are used:

* ``SimulationConfig``: an immutable (frozen) snapshot with the *effective*
  values of one experiment, in SI units.  The engine receives it and every
  stored result keeps it, so two runs with equal configs are the same
  experiment.
* A flat "form" dict with the raw widget values, including UI conveniences
  (e.g. "define the figure-eight by frequency and speed").  ``form_to_config``
  and ``config_to_form`` convert between both.

Defaults are read from the simulator (``smc_figure_eight`` signature and the
``FIGURE_EIGHT_*`` / ``TRACKING_LOOKAHEAD`` / ``DT`` constants), so they follow
the simulator if its defaults change.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
from dataclasses import dataclass

import numpy as np

from dashboard02_smc import PROJECT_ROOT  # noqa: F401  (simulator import path)

import controllers
from usv_model import PARAMS, TU_MAX

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Limits used by validation.
# ---------------------------------------------------------------------------
MAX_T_FINAL_S = 3600.0          # hard limit on simulated time
LONG_RUN_WARNING_S = 1200.0     # warn above this simulated time
MAX_OUTPUT_SAMPLES = 400_001
SAMPLE_RATE_LIMITS_HZ = (1.0, 1000.0)
OUTPUT_DT_LIMITS_S = (1e-3, 0.1)
THIN_LAYER_CONTINUOUS = 0.01    # continuous mode gets slow below this phi

SIMULATOR_DT_S = 0.02           # run_simulation.DT (not imported: that module
                                # pulls in matplotlib and the CBF machinery)

DIGITAL = "digital"
CONTINUOUS = "continuous"
SIM_MODES = (DIGITAL, CONTINUOUS)

REF_MODES = ("freq_speed", "amp_freq", "amp_speed")

AXES = ("u", "v", "r")                  # components of [dTu, dTv, dTr]
DURATION_MODES = ("laps", "manual")
INITIAL_MODES = ("reference", "offset")

# Disturbance models of disturbances.py (same set as the first dashboard).
DIST_LABELS = {
    "none": "No disturbance",
    "constant": "Constant",
    "sinusoidal": "Sinusoidal",
    "directional": "Directional force (fixed inertial direction)",
    "wind": "Wind-like (fixed profile)",
    "combined": "Combined (wind-like + directional + sinusoidal)",
}
DIST_KINDS = tuple(DIST_LABELS)
# max |component| of disturbances.wind_like_disturbance (fixed profile):
# dTu = 2 sin(.), dTv = 3 + 4 sin(.), dTr = 2.5 sin(.)
WIND_BOUNDS = (2.0, 7.0, 2.5)


def _signature_defaults(function):
    return {
        name: parameter.default
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


SMC_DEFAULTS = _signature_defaults(controllers.smc_figure_eight)


# ---------------------------------------------------------------------------
# Figure-eight helpers: x = A sin(wt), y = A sin(wt) cos(wt).
# ---------------------------------------------------------------------------
def peak_speed(amplitude, period):
    """Peak reference speed sqrt(2) A w [m/s] (reached at the crossing)."""
    return math.sqrt(2.0) * amplitude * 2.0 * math.pi / period


def amplitude_from(frequency_hz, speed):
    """Amplitude giving peak speed ``speed`` at frequency ``frequency_hz``."""
    return speed / (math.sqrt(2.0) * 2.0 * math.pi * frequency_hz)


def frequency_from(amplitude, speed):
    """Frequency [Hz] giving peak speed ``speed`` with amplitude ``amplitude``."""
    return speed / (math.sqrt(2.0) * 2.0 * math.pi * amplitude)


def mean_speed(amplitude, period, samples=2000):
    """Mean reference speed over one period [m/s]."""
    t = np.linspace(0.0, period, samples, endpoint=False)
    _, _, x_dot, y_dot, _, _ = controllers.figure_eight_reference(
        t, amplitude, period
    )
    return float(np.mean(np.hypot(x_dot, y_dot)))


def max_steady_surge_speed():
    """Surge speed at which full forward thrust balances damping [m/s]."""
    z4, z5 = PARAMS["z4"], PARAMS["z5"]
    return (-z4 + math.sqrt(z4**2 + 4.0 * z5 * TU_MAX)) / (2.0 * z5)


# ---------------------------------------------------------------------------
# Configuration dataclasses (frozen snapshots, SI units).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReferenceConfig:
    """Figure-eight of ``controllers.figure_eight_reference``."""

    amplitude_m: float
    period_s: float
    laps: float

    @property
    def frequency_hz(self):
        return 1.0 / self.period_s

    @property
    def omega_radps(self):
        return 2.0 * math.pi / self.period_s

    @property
    def peak_speed_mps(self):
        return peak_speed(self.amplitude_m, self.period_s)

    @property
    def t_final_s(self):
        return self.laps * self.period_s


@dataclass(frozen=True)
class SurgeSMCConfig:
    """Surge SMC gains (argument names of ``smc_figure_eight``)."""

    k_pos: float
    lambda_u: float
    k_s_u: float
    phi_u: float


@dataclass(frozen=True)
class YawSMCConfig:
    """Yaw SMC gains (argument names of ``smc_figure_eight``)."""

    k_psi: float
    lambda_r: float
    k_s_r: float
    phi_r: float


@dataclass(frozen=True)
class DisturbanceConfig:
    """One disturbance model of ``disturbances.py`` (body-frame, additive).

    Tuples are per axis ``(Tu, Tv, Tr)``.  Only the fields of the selected
    model are kept (``normalized``), so hidden widget values never make two
    experiments differ.
    """

    kind: str = "none"
    constant: tuple = (0.0, 0.0, 0.0)            # [N, N, N m]
    sin_amplitude: tuple = (0.0, 0.0, 0.0)       # [N, N, N m]
    sin_frequency_hz: tuple = (0.0, 0.0, 0.0)
    sin_phase_rad: tuple = (0.0, 0.0, 0.0)
    sin_bias: tuple = (0.0, 0.0, 0.0)            # [N, N, N m]
    dir_magnitude_N: float = 0.0
    dir_direction_deg: float = 0.0
    include_wind: bool = False

    def normalized(self):
        kind = self.kind
        if kind == "constant":
            return DisturbanceConfig(kind, constant=self.constant)
        if kind == "sinusoidal":
            return DisturbanceConfig(
                kind, sin_amplitude=self.sin_amplitude,
                sin_frequency_hz=self.sin_frequency_hz,
                sin_phase_rad=self.sin_phase_rad, sin_bias=self.sin_bias)
        if kind == "directional":
            return DisturbanceConfig(kind, dir_magnitude_N=self.dir_magnitude_N,
                                     dir_direction_deg=self.dir_direction_deg)
        if kind == "combined":
            return DisturbanceConfig(
                kind, sin_amplitude=self.sin_amplitude,
                sin_frequency_hz=self.sin_frequency_hz,
                sin_phase_rad=self.sin_phase_rad,
                dir_magnitude_N=self.dir_magnitude_N,
                dir_direction_deg=self.dir_direction_deg,
                include_wind=bool(self.include_wind))
        return DisturbanceConfig(kind)          # none, wind (no parameters)

    @property
    def active(self):
        return self.kind != "none"

    @property
    def label(self):
        return DIST_LABELS.get(self.kind, self.kind)

    def bounds(self):
        """Upper bounds of max_t |dTu|, |dTv|, |dTr| (worst case over time and
        heading)."""
        zero = (0.0, 0.0, 0.0)
        if self.kind == "constant":
            return tuple(abs(c) for c in self.constant)
        if self.kind == "sinusoidal":
            return tuple(abs(b) + abs(a)
                         for a, b in zip(self.sin_amplitude, self.sin_bias))
        directional = (abs(self.dir_magnitude_N), abs(self.dir_magnitude_N), 0.0)
        if self.kind == "directional":
            return directional
        if self.kind == "wind":
            return WIND_BOUNDS
        if self.kind == "combined":
            wind = WIND_BOUNDS if self.include_wind else zero
            return tuple(w + d + abs(a) for w, d, a in
                         zip(wind, directional, self.sin_amplitude))
        return zero


@dataclass(frozen=True)
class SimulationSettings:
    """Controller implementation, lookahead distance and initial offset
    (zero when the vessel starts on the reference)."""

    mode: str                  # "digital" (sampled + ZOH) or "continuous"
    sample_rate_hz: float      # digital: controller rate; continuous: 0
    output_dt_s: float         # effective spacing of the stored samples
    lookahead_L_m: float
    dx0_m: float
    dy0_m: float
    dpsi0_rad: float
    du0_mps: float

    @property
    def sample_period_s(self):
        return 1.0 / self.sample_rate_hz if self.mode == DIGITAL else None


@dataclass(frozen=True)
class SimulationConfig:
    """Complete, immutable description of one experiment."""

    reference: ReferenceConfig
    surge: SurgeSMCConfig
    yaw: YawSMCConfig
    disturbance: DisturbanceConfig
    simulation: SimulationSettings

    @property
    def t_final_s(self):
        return self.reference.t_final_s

    def initial_state(self):
        """Reference state at t = 0 plus the configured offsets."""
        ref, sim = self.reference, self.simulation
        state = np.array(
            controllers.figure_eight_initial_state(ref.amplitude_m, ref.period_s),
            dtype=float,
        )
        state[0] += sim.dx0_m
        state[1] += sim.dy0_m
        state[2] += sim.dpsi0_rad
        state[3] += sim.du0_mps
        return state

    def smc_kwargs(self):
        """Keyword arguments for ``controllers.smc_figure_eight``."""
        return dict(
            k_pos=self.surge.k_pos,
            k_psi=self.yaw.k_psi,
            lambda_u=self.surge.lambda_u,
            lambda_r=self.yaw.lambda_r,
            k_s_u=self.surge.k_s_u,
            k_s_r=self.yaw.k_s_r,
            phi_u=self.surge.phi_u,
            phi_r=self.yaw.phi_r,
            lookahead_L=self.simulation.lookahead_L_m,
            amplitude=self.reference.amplitude_m,
            period=self.reference.period_s,
        )

    # -- serialization -----------------------------------------------------
    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, **dataclasses.asdict(self)}

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data):
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported configuration schema {version}.")
        dist = {key: tuple(float(v) for v in value)
                if isinstance(value, (list, tuple)) else value
                for key, value in data["disturbance"].items()}
        return cls(
            reference=ReferenceConfig(**data["reference"]),
            surge=SurgeSMCConfig(**data["surge"]),
            yaw=YawSMCConfig(**data["yaw"]),
            disturbance=DisturbanceConfig(**dist),
            simulation=SimulationSettings(**data["simulation"]),
        )

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def flat(self):
        """Flat ``{"section.field": value}`` view (tables, diffs)."""
        flat = {}

        def walk(prefix, value):
            if isinstance(value, dict):
                for key, item in value.items():
                    walk(f"{prefix}.{key}" if prefix else key, item)
            elif isinstance(value, (list, tuple)):
                for axis, item in zip(AXES, value):
                    flat[f"{prefix}[{axis}]"] = item
            else:
                flat[prefix] = value

        data = dataclasses.asdict(self)
        walk("", data)
        return flat


# ---------------------------------------------------------------------------
# Display metadata: flat key -> (LaTeX symbol, HTML symbol, unit, group).
# Used for legends, run cards and "what changed" lists.
# ---------------------------------------------------------------------------
PARAM_INFO = {
    "reference.amplitude_m": (r"A", "A", "m", "Reference"),
    "reference.period_s": (r"T", "T", "s", "Reference"),
    "reference.laps": (r"N_{laps}", "N<sub>laps</sub>", "-", "Reference"),
    "surge.k_pos": (r"k_{pos}", "k<sub>pos</sub>", "1/s", "Surge SMC"),
    "surge.lambda_u": (r"\lambda_u", "λ<sub>u</sub>", "1/s", "Surge SMC"),
    "surge.k_s_u": (r"k_{s,u}", "k<sub>s,u</sub>", "m/s²", "Surge SMC"),
    "surge.phi_u": (r"\phi_u", "φ<sub>u</sub>", "m/s", "Surge SMC"),
    "yaw.k_psi": (r"k_{\psi}", "k<sub>ψ</sub>", "1/s", "Yaw SMC"),
    "yaw.lambda_r": (r"\lambda_r", "λ<sub>r</sub>", "1/s", "Yaw SMC"),
    "yaw.k_s_r": (r"k_{s,r}", "k<sub>s,r</sub>", "rad/s²", "Yaw SMC"),
    "yaw.phi_r": (r"\phi_r", "φ<sub>r</sub>", "rad/s", "Yaw SMC"),
    "simulation.mode": (r"\text{implementation}", "implementation", "",
                        "Simulation"),
    "simulation.sample_rate_hz": (r"f_s", "f<sub>s</sub>", "Hz", "Simulation"),
    "simulation.output_dt_s": (r"\Delta t", "Δt", "s", "Simulation"),
    "simulation.lookahead_L_m": (r"L", "L", "m", "Simulation"),
    "simulation.dx0_m": (r"\Delta x_0", "Δx<sub>0</sub>", "m", "Initial state"),
    "simulation.dy0_m": (r"\Delta y_0", "Δy<sub>0</sub>", "m", "Initial state"),
    "simulation.dpsi0_rad": (r"\Delta \psi_0", "Δψ<sub>0</sub>", "rad",
                             "Initial state"),
    "simulation.du0_mps": (r"\Delta u_0", "Δu<sub>0</sub>", "m/s",
                           "Initial state"),
}
PARAM_INFO.update({
    "disturbance.kind": (r"\text{model}", "model", "", "Disturbance"),
    "disturbance.dir_magnitude_N": (r"F", "F", "N", "Disturbance"),
    "disturbance.dir_direction_deg": (r"\beta", "β", "deg", "Disturbance"),
    "disturbance.include_wind": (r"\text{wind-like}", "wind-like", "",
                                 "Disturbance"),
})
for _axis in AXES:
    _unit = "N m" if _axis == "r" else "N"
    PARAM_INFO.update({
        f"disturbance.constant[{_axis}]": (
            rf"\delta T_{_axis}", f"δT<sub>{_axis}</sub>", _unit, "Disturbance"),
        f"disturbance.sin_amplitude[{_axis}]": (
            rf"A_{_axis}", f"A<sub>{_axis}</sub>", _unit, "Disturbance"),
        f"disturbance.sin_frequency_hz[{_axis}]": (
            rf"f_{_axis}", f"f<sub>{_axis}</sub>", "Hz", "Disturbance"),
        f"disturbance.sin_phase_rad[{_axis}]": (
            rf"\varphi_{_axis}", f"ϕ<sub>{_axis}</sub>", "rad", "Disturbance"),
        f"disturbance.sin_bias[{_axis}]": (
            rf"b_{_axis}", f"b<sub>{_axis}</sub>", _unit, "Disturbance"),
    })

SURGE_KEYS = ("surge.lambda_u", "surge.k_s_u", "surge.phi_u", "surge.k_pos")
YAW_KEYS = ("yaw.lambda_r", "yaw.k_s_r", "yaw.phi_r", "yaw.k_psi")


def format_value(value):
    """Compact numeric formatting used in labels (4 significant digits)."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.4g}"


# ---------------------------------------------------------------------------
# Config comparison.
# ---------------------------------------------------------------------------
def _same_value(a, b):
    # Floats are compared with a tiny tolerance so that unit round trips
    # (deg <-> rad, frequency <-> period) are not reported as changes.
    numeric = (int, float, np.floating, np.integer)
    if (isinstance(a, numeric) and isinstance(b, numeric)
            and not isinstance(a, bool) and not isinstance(b, bool)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def config_diff(old, new):
    """List of ``(flat key, old value, new value)`` that differ."""
    old_flat = {} if old is None else old.flat()
    new_flat = {} if new is None else new.flat()
    keys = list(old_flat) + [k for k in new_flat if k not in old_flat]
    return [
        (key, old_flat.get(key, "—"), new_flat.get(key, "—"))
        for key in keys
        if not _same_value(old_flat.get(key, "—"), new_flat.get(key, "—"))
    ]


def same_experiment(a, b):
    if a is None or b is None:
        return False
    return not config_diff(a, b)


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    errors: list = dataclasses.field(default_factory=list)
    warnings: list = dataclasses.field(default_factory=list)

    @property
    def ok(self):
        return not self.errors


class ConfigValidationError(ValueError):
    """Raised when a configuration cannot be simulated."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("Invalid configuration: " + "; ".join(self.errors))


def _finite(value):
    return isinstance(value, (int, float, np.floating, np.integer)) and \
        not isinstance(value, bool) and math.isfinite(float(value))


def validate_config(config):
    """Check a ``SimulationConfig``; returns a ``ValidationReport``."""
    report = ValidationReport()
    errors, warnings = report.errors, report.warnings

    for key, value in config.flat().items():
        if isinstance(value, (str, bool)):
            continue
        if not _finite(value):
            symbol = PARAM_INFO.get(key, (key,))[0]
            errors.append(f"${symbol}$ must be a finite number.")
    if errors:
        return report

    ref, surge, yaw = config.reference, config.surge, config.yaw
    sim, dist = config.simulation, config.disturbance

    # Reference.
    if ref.amplitude_m <= 0.0:
        errors.append("Figure-eight amplitude $A$ must be > 0 m.")
    if ref.period_s <= 0.0:
        errors.append("Figure-eight frequency $f$ must be > 0 Hz.")
    if ref.laps <= 0.0:
        errors.append("Number of laps $N_{laps}$ must be > 0.")

    # Controller gains.
    for section, name, symbol in (
        (surge, "k_pos", r"k_{pos}"), (surge, "lambda_u", r"\lambda_u"),
        (surge, "k_s_u", r"k_{s,u}"), (yaw, "k_psi", r"k_{\psi}"),
        (yaw, "lambda_r", r"\lambda_r"), (yaw, "k_s_r", r"k_{s,r}"),
    ):
        if getattr(section, name) < 0.0:
            errors.append(f"${symbol}$ must be $\\ge 0$.")
    for value, symbol in ((surge.phi_u, r"\phi_u"), (yaw.phi_r, r"\phi_r")):
        if value <= 0.0:
            errors.append(f"Boundary-layer half-width ${symbol}$ must be > 0 "
                          r"(it divides $s$ in $\mathrm{sat}(s/\phi)$).")
    if sim.lookahead_L_m < 0.0:
        errors.append("Lookahead distance $L$ must be $\\ge 0$ m.")

    # Simulation settings.
    if sim.mode not in SIM_MODES:
        errors.append(f"Unknown implementation '{sim.mode}'.")
    elif sim.mode == DIGITAL:
        lo, hi = SAMPLE_RATE_LIMITS_HZ
        if not lo <= sim.sample_rate_hz <= hi:
            errors.append(f"Controller sample rate $f_s$ must be between "
                          f"{lo:g} and {hi:g} Hz.")
    else:
        lo, hi = OUTPUT_DT_LIMITS_S
        if not lo <= sim.output_dt_s <= hi:
            errors.append(rf"Output interval $\Delta t$ must be between {lo:g} "
                          f"and {hi:g} s.")

    if dist.kind not in DIST_KINDS:
        errors.append(f"Unknown disturbance model '{dist.kind}'.")
    for axis, frequency in zip(AXES, dist.sin_frequency_hz):
        if frequency < 0.0:
            errors.append(f"Disturbance frequency $f_{axis}$ must be "
                          "$\\ge 0$ Hz.")
    if dist.dir_magnitude_N < 0.0:
        errors.append("Directional force magnitude $F$ must be $\\ge 0$ N.")
    if errors:
        return report

    t_final = ref.t_final_s
    if t_final > MAX_T_FINAL_S:
        errors.append(rf"Simulated time $T_{{sim}}$ = "
                      f"{t_final:.0f} s exceeds the limit of "
                      f"{MAX_T_FINAL_S:.0f} s. Reduce $N_{{laps}}$ or "
                      r"$T_{sim}$, or raise $f$.")
    elif t_final > LONG_RUN_WARNING_S:
        warnings.append(rf"Long simulation: $T_{{sim}}$ = {t_final:.0f} s.")
    if t_final / sim.output_dt_s + 1 > MAX_OUTPUT_SAMPLES:
        errors.append(f"Too many output samples "
                      f"({t_final / sim.output_dt_s:.0f}). Lower $f_s$ "
                      r"(or raise $\Delta t$) or shorten the run.")
    if errors:
        return report

    # Physically questionable, but allowed.
    v_max = max_steady_surge_speed()
    if ref.peak_speed_mps > v_max:
        warnings.append(
            rf"Peak reference speed $v_{{max}}$ = {ref.peak_speed_mps:.2f} m/s "
            f"exceeds the maximum steady surge speed of the thrusters "
            f"({v_max:.2f} m/s): thruster saturation is expected."
        )
    if sim.mode == CONTINUOUS and min(surge.phi_u, yaw.phi_r) < \
            THIN_LAYER_CONTINUOUS:
        warnings.append(
            r"Continuous implementation with a thin boundary layer "
            rf"($\phi < {THIN_LAYER_CONTINUOUS:g}$): the adaptive integrator "
            "becomes very slow, and an ideal continuous controller shows no "
            "chattering. Use the digital implementation to study chattering."
        )
    for check in design_checks(config):
        if check["kind"] == "sliding" and not check["ok"]:
            warnings.append(check["warning"])
    from dashboard02_smc import tuning   # tuning imports this module
    actuators = tuning.actuator_check(config)
    if actuators.utilization_pct >= 100.0:
        warnings.append(
            f"Thrusters: the trajectory plus the worst-sign disturbance need "
            f"{actuators.utilization_pct:.0f}% of a thruster's limit. Sliding "
            "will be interrupted by saturation whatever the gains."
        )
    return report


# ---------------------------------------------------------------------------
# Design checks (approximate SMC theory shown next to the inputs).
# ---------------------------------------------------------------------------
def design_checks(config):
    """Approximate SMC design indicators per channel.

    Returns a list of dicts with ``channel``, ``kind``, ``ok`` (True/False/None),
    ``latex`` (explanation) and, for failed checks, ``warning``.

    * ``sliding``: sliding condition k_s + lambda phi >= d_bar, where d_bar is
      the matched uncertainty: disturbance / z plus the trajectory-induced part
      (see ``tuning.py``);
    * ``steady``: for a constant disturbance d = delta T / z the equilibrium
      inside the layer is s = d / (lambda + k_s/phi), and e ~ s / k;
    * ``discrete``: a digital controller with ZOH and period Ts behaves inside
      the layer like s[k+1] ~ (1 - Gamma) s[k], Gamma = Ts (lambda + k_s/phi):
      monotone for Gamma < 1, alternating sign (chattering) for 1 < Gamma < 2,
      unstable inside the layer (limit-cycle chattering) for Gamma > 2.
      Simulations of this vessel confirm a sharp chattering onset at Gamma ~ 1;
    * ``surface``: time constant 1/k of the error on the surface.

    These ignore thruster saturation and the remaining model approximations,
    so they are guides, not guarantees.
    """
    from dashboard02_smc import tuning   # tuning imports this module

    checks = []
    sim, dist = config.simulation, config.disturbance
    hints = tuning.gain_hints(config)
    bound_u, _bound_v, bound_r = dist.bounds()
    channels = (
        ("surge", "u", config.surge.lambda_u, config.surge.k_s_u,
         config.surge.phi_u, config.surge.k_pos, PARAMS["z1"], "z_1", "m/s",
         bound_u),
        ("yaw", "r", config.yaw.lambda_r, config.yaw.k_s_r, config.yaw.phi_r,
         config.yaw.k_psi, PARAMS["z3"], "z_3", "rad/s", bound_r),
    )
    for (channel, i, lam, k_s, phi, k, z, z_sym, s_unit,
         bound) in channels:
        if phi <= 0.0:
            continue
        hint = hints[channel]
        ok = hint.sliding_ok
        relation = r"\ge" if ok else "<"
        checks.append({
            "channel": channel, "kind": "sliding", "ok": ok,
            "latex": (
                rf"k_{{s,{i}}} + \lambda_{i}\phi_{i} = "
                rf"{k_s + lam * phi:.3g} \;{relation}\; \bar d_{i} = "
                rf"{hint.d_bar:.3g}"
            ),
            "warning": (
                rf"{channel.capitalize()} SMC: sliding condition violated, "
                rf"$k_{{s,{i}}} + \lambda_{i}\phi_{i}$ = {k_s + lam * phi:.3g} "
                rf"< $\bar d_{i}$ = {hint.d_bar:.3g} (disturbance + trajectory). "
                rf"Use $k_{{s,{i}}} \ge$ {hint.k_min:.3g} (suggested "
                rf"{tuning.suggested_value(hint.k_suggested):.3g}); see the gain "
                "hint under *Disturbances*."
            ),
        })
        gain_in_layer = lam + k_s / phi
        d_ext = bound / z
        if d_ext > 0.0:
            s_ss = d_ext / gain_in_layer if gain_in_layer > 0 else math.inf
            e_ss = s_ss / k if k > 0 else math.inf
            checks.append({
                "channel": channel, "kind": "steady", "ok": s_ss <= phi,
                "latex": (
                    rf"|s_{i}|_{{ss}} \approx \dfrac{{\max|\delta T_{i}|/{z_sym}}}"
                    rf"{{\lambda_{i} + k_{{s,{i}}}/\phi_{i}}} = {s_ss:.3g}\ "
                    rf"\mathrm{{{s_unit}}}"
                    + (rf",\quad |e|_{{ss}} \approx |s_{i}|_{{ss}}/k = "
                       rf"{e_ss:.3g}" if math.isfinite(e_ss) else "")
                ),
            })
        if sim.mode == DIGITAL and sim.sample_rate_hz > 0:
            gamma = gain_in_layer / sim.sample_rate_hz
            if gamma < 1.0:
                verdict, ok = r"\text{no chattering expected}", True
            elif gamma < 2.0:
                verdict, ok = r"\text{oscillatory in the layer: chattering}", False
            else:
                verdict, ok = (r"\text{unstable in the layer: limit-cycle "
                               r"chattering}", False)
            checks.append({
                "channel": channel, "kind": "discrete", "ok": ok,
                "gamma": gamma,
                "latex": (
                    rf"\Gamma_{i} = T_s\left(\lambda_{i} + \dfrac{{k_{{s,{i}}}}}"
                    rf"{{\phi_{i}}}\right) = {gamma:.3g}\ \Rightarrow\ {verdict}"
                ),
            })
        if k > 0:
            checks.append({
                "channel": channel, "kind": "surface", "ok": None,
                "latex": (
                    rf"s_{i} = 0 \Rightarrow e(t) = e(0)\,e^{{-t/\tau}},\ "
                    rf"\tau = 1/k = {1.0 / k:.3g}\ \mathrm{{s}}"
                ),
            })
    return checks


# ---------------------------------------------------------------------------
# Form <-> config.
# ---------------------------------------------------------------------------
_DEFAULT_PERIOD = controllers.FIGURE_EIGHT_PERIOD
_DEFAULT_AMPLITUDE = controllers.FIGURE_EIGHT_AMPLITUDE

# Disturbance defaults = the modes of run_simulation.DISTURBANCE_OPTIONS
# (same values as the first dashboard); checked by the tests.
_CONSTANT_DEFAULT = (0.0, 0.0, 15.0)
_PURE_SIN_DEFAULT = {            # disturbances.pure_sinusoidal_disturbance
    "amplitude": (3.0, 4.0, 2.0),
    "frequency": (0.035, 0.050, 0.025),
    "phase": (0.0, 0.8, 1.4),
}
_DIRECTIONAL_DEFAULT = (25.0, 45.0)
_COMBINED_DEFAULT = {
    "directional": (7.0, 64.0),
    "amplitude": (6.0, 3.0, 1.5),
    "frequency": (0.04, 0.06, 0.03),
    "phase": (0.0, 0.5, 1.0),
}


def default_form():
    """Widget values equivalent to the simulator defaults."""
    form = {
        "ref_mode": "freq_speed",
        "ref_amplitude_m": float(_DEFAULT_AMPLITUDE),
        "ref_frequency_hz": 1.0 / _DEFAULT_PERIOD,
        "ref_peak_speed_mps": peak_speed(_DEFAULT_AMPLITUDE, _DEFAULT_PERIOD),
        "ref_laps": float(controllers.FIGURE_EIGHT_LAPS),
        "surge_k_pos": float(SMC_DEFAULTS["k_pos"]),
        "surge_lambda_u": float(SMC_DEFAULTS["lambda_u"]),
        "surge_k_s_u": float(SMC_DEFAULTS["k_s_u"]),
        "surge_phi_u": float(SMC_DEFAULTS["phi_u"]),
        "yaw_k_psi": float(SMC_DEFAULTS["k_psi"]),
        "yaw_lambda_r": float(SMC_DEFAULTS["lambda_r"]),
        "yaw_k_s_r": float(SMC_DEFAULTS["k_s_r"]),
        "yaw_phi_r": float(SMC_DEFAULTS["phi_r"]),
        "sim_duration_mode": "laps",
        "sim_t_final_s": float(controllers.FIGURE_EIGHT_LAPS
                               * _DEFAULT_PERIOD),
        "sim_mode": DIGITAL,
        "sim_sample_rate_hz": 1.0 / SIMULATOR_DT_S,
        "sim_output_dt_s": SIMULATOR_DT_S,
        "sim_lookahead_L_m": float(SMC_DEFAULTS["lookahead_L"]),
        "sim_initial_mode": "reference",
        "init_dx_m": 0.0,
        "init_dy_m": 0.0,
        "init_dpsi_deg": 0.0,
        "init_du_mps": 0.0,
        "dist_kind": "none",
        "dist_const_dTu": _CONSTANT_DEFAULT[0],
        "dist_const_dTv": _CONSTANT_DEFAULT[1],
        "dist_const_dTr": _CONSTANT_DEFAULT[2],
        "dist_dir_magnitude_N": _DIRECTIONAL_DEFAULT[0],
        "dist_dir_direction_deg": _DIRECTIONAL_DEFAULT[1],
        "dist_comb_wind": True,
        "dist_comb_dir_magnitude_N": _COMBINED_DEFAULT["directional"][0],
        "dist_comb_dir_direction_deg": _COMBINED_DEFAULT["directional"][1],
    }
    for i, axis in enumerate(AXES):
        form[f"dist_sin_amp_{axis}"] = _PURE_SIN_DEFAULT["amplitude"][i]
        form[f"dist_sin_freq_{axis}"] = _PURE_SIN_DEFAULT["frequency"][i]
        form[f"dist_sin_phase_{axis}"] = _PURE_SIN_DEFAULT["phase"][i]
        form[f"dist_sin_bias_{axis}"] = 0.0
        form[f"dist_comb_amp_{axis}"] = _COMBINED_DEFAULT["amplitude"][i]
        form[f"dist_comb_freq_{axis}"] = _COMBINED_DEFAULT["frequency"][i]
        form[f"dist_comb_phase_{axis}"] = _COMBINED_DEFAULT["phase"][i]
    return form


FORM_KEYS = tuple(default_form())


def _f(form, key):
    try:
        return float(form[key])
    except (TypeError, ValueError):
        return float("nan")


def _triple(form, prefix):
    return tuple(_f(form, f"{prefix}_{axis}") for axis in AXES)


def form_disturbance(form):
    """``DisturbanceConfig`` of the disturbance widgets."""
    kind = form["dist_kind"]
    if kind not in DIST_KINDS:
        raise ConfigValidationError([f"Unknown disturbance model '{kind}'."])
    if kind == "combined":
        magnitude = _f(form, "dist_comb_dir_magnitude_N")
        direction = _f(form, "dist_comb_dir_direction_deg")
        amplitude = _triple(form, "dist_comb_amp")
        frequency = _triple(form, "dist_comb_freq")
        phase = _triple(form, "dist_comb_phase")
    else:
        magnitude = _f(form, "dist_dir_magnitude_N")
        direction = _f(form, "dist_dir_direction_deg")
        amplitude = _triple(form, "dist_sin_amp")
        frequency = _triple(form, "dist_sin_freq")
        phase = _triple(form, "dist_sin_phase")
    return DisturbanceConfig(
        kind=kind,
        constant=(_f(form, "dist_const_dTu"), _f(form, "dist_const_dTv"),
                  _f(form, "dist_const_dTr")),
        sin_amplitude=amplitude, sin_frequency_hz=frequency,
        sin_phase_rad=phase, sin_bias=_triple(form, "dist_sin_bias"),
        dir_magnitude_N=magnitude, dir_direction_deg=direction,
        include_wind=bool(form["dist_comb_wind"]),
    ).normalized()


_REF_INPUTS = {
    "ref_amplitude_m": "amplitude $A$",
    "ref_frequency_hz": "frequency $f$",
    "ref_peak_speed_mps": "peak speed $v_{max}$",
}
_REF_MODE_INPUTS = {
    "freq_speed": ("ref_frequency_hz", "ref_peak_speed_mps"),
    "amp_freq": ("ref_amplitude_m", "ref_frequency_hz"),
    "amp_speed": ("ref_amplitude_m", "ref_peak_speed_mps"),
}


def resolve_reference(form):
    """(amplitude, frequency, peak speed) implied by the form; may be nan."""
    mode = form["ref_mode"]
    amplitude = _f(form, "ref_amplitude_m")
    frequency = _f(form, "ref_frequency_hz")
    speed = _f(form, "ref_peak_speed_mps")
    try:
        if mode == "freq_speed":
            amplitude = amplitude_from(frequency, speed)
        elif mode == "amp_freq":
            speed = peak_speed(amplitude, 1.0 / frequency)
        elif mode == "amp_speed":
            frequency = frequency_from(amplitude, speed)
        else:
            raise ValueError(mode)
    except (ZeroDivisionError, ValueError, OverflowError):
        return float("nan"), float("nan"), float("nan")

    # The form stores all three values and the UI keeps them consistent; if
    # the derived one agrees with the stored one up to round-off, keep the
    # stored value (so the defaults give exactly A = 15 m, T = 140 s).
    def snap(derived, key):
        stored = _f(form, key)
        if math.isfinite(derived) and math.isclose(derived, stored,
                                                   rel_tol=1e-12):
            return stored
        return derived

    return (snap(amplitude, "ref_amplitude_m"),
            snap(frequency, "ref_frequency_hz"),
            snap(speed, "ref_peak_speed_mps"))


def form_to_config(form):
    """Build a ``SimulationConfig`` from widget values.

    Raises ``ConfigValidationError`` when the values cannot define a config
    (e.g. a zero frequency); range checks are done by ``validate_config``.
    """
    if form["ref_mode"] not in REF_MODES:
        raise ConfigValidationError(
            [f"Unknown reference mode '{form['ref_mode']}'."])
    errors = [
        f"Reference {_REF_INPUTS[key]} must be > 0."
        for key in _REF_MODE_INPUTS[form["ref_mode"]]
        if not _f(form, key) > 0.0
    ]
    if errors:
        raise ConfigValidationError(errors)
    amplitude, frequency, _ = resolve_reference(form)
    if form["sim_duration_mode"] == "manual":
        t_final = _f(form, "sim_t_final_s")
        if not t_final > 0.0:
            raise ConfigValidationError(
                [r"Total simulation time $T_{sim}$ must be > 0 s."])
        laps = t_final * frequency
    else:
        laps = _f(form, "ref_laps")
    offset = form["sim_initial_mode"] == "offset"

    mode = form["sim_mode"]
    if mode == DIGITAL:
        rate = _f(form, "sim_sample_rate_hz")
        if not rate > 0.0:
            raise ConfigValidationError(
                ["Controller sample rate $f_s$ must be > 0 Hz."])
        output_dt = 1.0 / rate
    else:
        rate = 0.0
        output_dt = _f(form, "sim_output_dt_s")
        if not output_dt > 0.0:
            raise ConfigValidationError([r"Output interval $\Delta t$ must be "
                                         "> 0 s."])

    return SimulationConfig(
        reference=ReferenceConfig(
            amplitude_m=float(amplitude),
            period_s=float(1.0 / frequency),
            laps=float(laps),
        ),
        surge=SurgeSMCConfig(
            k_pos=_f(form, "surge_k_pos"),
            lambda_u=_f(form, "surge_lambda_u"),
            k_s_u=_f(form, "surge_k_s_u"),
            phi_u=_f(form, "surge_phi_u"),
        ),
        yaw=YawSMCConfig(
            k_psi=_f(form, "yaw_k_psi"),
            lambda_r=_f(form, "yaw_lambda_r"),
            k_s_r=_f(form, "yaw_k_s_r"),
            phi_r=_f(form, "yaw_phi_r"),
        ),
        disturbance=form_disturbance(form),
        simulation=SimulationSettings(
            mode=mode,
            sample_rate_hz=float(rate),
            output_dt_s=float(output_dt),
            lookahead_L_m=_f(form, "sim_lookahead_L_m"),
            dx0_m=_f(form, "init_dx_m") if offset else 0.0,
            dy0_m=_f(form, "init_dy_m") if offset else 0.0,
            dpsi0_rad=math.radians(_f(form, "init_dpsi_deg")) if offset else 0.0,
            du0_mps=_f(form, "init_du_mps") if offset else 0.0,
        ),
    )


@dataclass
class FormCheck:
    """Result of ``check_form``: config (None if not buildable) + report."""

    config: SimulationConfig | None
    report: ValidationReport


def check_form(form):
    """Convert and validate widget values in one step."""
    try:
        config = form_to_config(form)
    except ConfigValidationError as exc:
        return FormCheck(None, ValidationReport(errors=list(exc.errors)))
    return FormCheck(config, validate_config(config))


def config_to_form(config, base=None):
    """Widget values reproducing ``config`` (inverse of ``form_to_config``).

    The reference mode of ``base`` is kept, and all three reference fields
    are filled consistently.  Values of hidden widgets (e.g. the sinusoid of
    a channel set to "none") keep their ``base`` values.
    """
    form = dict(default_form() if base is None else base)
    ref, sim = config.reference, config.simulation
    form["ref_amplitude_m"] = ref.amplitude_m
    form["ref_frequency_hz"] = ref.frequency_hz
    form["ref_peak_speed_mps"] = ref.peak_speed_mps
    form["ref_laps"] = ref.laps
    for name in ("k_pos", "lambda_u", "k_s_u", "phi_u"):
        form[f"surge_{name}"] = getattr(config.surge, name)
    for name in ("k_psi", "lambda_r", "k_s_r", "phi_r"):
        form[f"yaw_{name}"] = getattr(config.yaw, name)
    dist = config.disturbance
    form["dist_kind"] = dist.kind
    if dist.kind == "constant":
        (form["dist_const_dTu"], form["dist_const_dTv"],
         form["dist_const_dTr"]) = dist.constant
    elif dist.kind in ("sinusoidal", "combined"):
        prefix = "dist_sin" if dist.kind == "sinusoidal" else "dist_comb"
        for i, axis in enumerate(AXES):
            form[f"{prefix}_amp_{axis}"] = dist.sin_amplitude[i]
            form[f"{prefix}_freq_{axis}"] = dist.sin_frequency_hz[i]
            form[f"{prefix}_phase_{axis}"] = dist.sin_phase_rad[i]
            if dist.kind == "sinusoidal":
                form[f"dist_sin_bias_{axis}"] = dist.sin_bias[i]
    if dist.kind == "directional":
        form["dist_dir_magnitude_N"] = dist.dir_magnitude_N
        form["dist_dir_direction_deg"] = dist.dir_direction_deg
    elif dist.kind == "combined":
        form["dist_comb_wind"] = dist.include_wind
        form["dist_comb_dir_magnitude_N"] = dist.dir_magnitude_N
        form["dist_comb_dir_direction_deg"] = dist.dir_direction_deg
    form["sim_mode"] = sim.mode
    if sim.mode == DIGITAL:
        form["sim_sample_rate_hz"] = sim.sample_rate_hz
    else:
        form["sim_output_dt_s"] = sim.output_dt_s
    form["sim_lookahead_L_m"] = sim.lookahead_L_m
    offsets = (sim.dx0_m, sim.dy0_m, sim.dpsi0_rad, sim.du0_mps)
    form["sim_initial_mode"] = "offset" if any(offsets) else "reference"
    form["init_dx_m"] = sim.dx0_m
    form["init_dy_m"] = sim.dy0_m
    form["init_dpsi_deg"] = math.degrees(sim.dpsi0_rad)
    form["init_du_mps"] = sim.du0_mps
    t_final = config.t_final_s
    form["sim_t_final_s"] = t_final
    form["sim_duration_mode"] = (
        "laps" if math.isclose(ref.laps, round(ref.laps * 4) / 4,
                               rel_tol=0, abs_tol=1e-9)
        else "manual")
    return {key: (float(value) if isinstance(value, (int, float, np.floating))
                  and not isinstance(value, bool) else value)
            for key, value in form.items()}


def default_config():
    return form_to_config(default_form())


# ---------------------------------------------------------------------------
# Scenario presets (form overrides applied on top of the defaults).
# ---------------------------------------------------------------------------
PRESETS = {
    "defaults": (
        "Simulator defaults",
        "Gains of smc_figure_eight, reference A = 15 m, T = 140 s, "
        "no disturbance.",
        {},
    ),
    "reaching": (
        "Reaching phase (initial offset)",
        "Starts 3 m to the side with 30° heading error: the phase plots show "
        "the reaching phase and the sliding phase.",
        {"sim_initial_mode": "offset", "init_dy_m": 3.0,
         "init_dpsi_deg": 30.0},
    ),
    "chattering": (
        "Thin boundary layer (chattering)",
        "phi_u = phi_r = 0.005 with a 20 Hz digital controller: "
        "Gamma >> 2, the control chatters.",
        {"surge_phi_u": 0.005, "yaw_phi_r": 0.005, "sim_sample_rate_hz": 20.0},
    ),
    "constant_dist": (
        "Constant disturbance rejection",
        "Constant matched disturbances dTu = 15 N and dTr = 8 N m.",
        {"dist_kind": "constant", "dist_const_dTu": 15.0,
         "dist_const_dTv": 0.0, "dist_const_dTr": 8.0},
    ),
    "weak_switching": (
        "Switching gain below the disturbance bound",
        "Sinusoidal yaw disturbance (15 N m) with k_s,r = 0.05 and "
        "lambda_r = 0.5: the sliding variable leaves the boundary layer.",
        {"dist_kind": "sinusoidal", "dist_sin_amp_u": 0.0,
         "dist_sin_amp_v": 0.0, "dist_sin_amp_r": 15.0,
         "dist_sin_freq_r": 0.05, "dist_sin_phase_r": 0.0,
         "yaw_k_s_r": 0.05, "yaw_lambda_r": 0.5},
    ),
    "combined": (
        "Combined disturbance (wind + current + waves)",
        "The 'combined' mode of run_simulation.py with the default gains.",
        {"dist_kind": "combined"},
    ),
    "fast_reference": (
        "Fast reference (thruster saturation)",
        "Peak reference speed 1.6 m/s at the default frequency.",
        {"ref_mode": "freq_speed", "ref_peak_speed_mps": 1.6},
    ),
}


def preset_form(name, base=None):
    """Form with the preset ``name`` applied on top of the defaults.

    Display-only preferences in ``base`` (the reference input mode) are kept
    unless the preset sets them.
    """
    form = default_form()
    if base is not None:
        form["ref_mode"] = base.get("ref_mode", form["ref_mode"])
    form.update(PRESETS[name][2])
    return form
