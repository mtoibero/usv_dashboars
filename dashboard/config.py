"""Experiment configuration for the dashboard (no Streamlit imports).

Two representations are used:

* ``SimulationConfig``: an immutable (frozen) snapshot containing only the
  *effective* values that define a simulation (SI units, resolved duration
  and initial state).  It is what the engine receives and what every stored
  result keeps, so two runs with equal configs are the same experiment.
* A flat "form" dict: the raw values of the dashboard widgets, including UI
  conveniences such as "duration from laps" or "define timing by peak speed".
  ``form_to_config`` / ``config_to_form`` convert between both.

Default values are read from the existing simulator (function signatures and
module constants) so they follow the simulator if its defaults change.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
from dataclasses import dataclass
from typing import ClassVar, Union

import numpy as np

from dashboard import PROJECT_ROOT  # noqa: F401  (ensures simulator import path)

import controllers
import run_simulation
from usv_model import PARAMS, TR_MAX, TU_MAX

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Limits used by validation (protect the session from unusable requests).
# ---------------------------------------------------------------------------
MAX_T_FINAL_S = 7200.0          # hard limit on simulated time
LONG_RUN_WARNING_S = 1200.0     # warn above this simulated time
MAX_OUTPUT_SAMPLES = 200_001    # t_final / dt + 1
THIN_BOUNDARY_LAYER = 0.01      # warn below this phi (slow, chattering solve)


def _signature_defaults(function):
    return {
        name: parameter.default
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


_SMC_DEFAULTS = _signature_defaults(controllers.smc_figure_eight)
_FEEDBACK_DEFAULTS = _signature_defaults(controllers.feedback_figure_eight)


# ---------------------------------------------------------------------------
# Reference helpers (figure-eight: x = A sin(wt), y = A sin(wt) cos(wt)).
# ---------------------------------------------------------------------------
def figure_eight_peak_speed(amplitude, period):
    """Peak reference speed A*w*sqrt(2) [m/s] (reached at the crossing)."""
    return amplitude * (2.0 * math.pi / period) * math.sqrt(2.0)


def figure_eight_period_from_peak_speed(amplitude, peak_speed):
    """Inverse of ``figure_eight_peak_speed``."""
    return 2.0 * math.pi * math.sqrt(2.0) * amplitude / peak_speed


def figure_eight_mean_speed(amplitude, period, samples=2000):
    """Mean reference speed over one period [m/s]."""
    t = np.linspace(0.0, period, samples, endpoint=False)
    _, _, x_dot, y_dot, _, _ = controllers.figure_eight_reference(
        t, amplitude, period
    )
    return float(np.mean(np.hypot(x_dot, y_dot)))


def max_steady_surge_speed():
    """Surge speed at which full forward thrust balances the damping [m/s].

    Solves z4*u + z5*u^2 = TU_MAX with the identified parameters.
    """
    z4, z5 = PARAMS["z4"], PARAMS["z5"]
    return (-z4 + math.sqrt(z4**2 + 4.0 * z5 * TU_MAX)) / (2.0 * z5)


# ---------------------------------------------------------------------------
# Configuration dataclasses (frozen snapshots).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SimulationSettings:
    """Duration, output sampling and initial state (SI units)."""

    t_final_s: float
    dt_s: float
    x0_m: float
    y0_m: float
    psi0_rad: float
    u0_mps: float
    v0_mps: float
    r0_radps: float

    def initial_state(self):
        return np.array([
            self.x0_m, self.y0_m, self.psi0_rad,
            self.u0_mps, self.v0_mps, self.r0_radps,
        ], dtype=float)


@dataclass(frozen=True)
class FigureEightReference:
    """Figure-eight reference of ``controllers.figure_eight_reference``."""

    kind: ClassVar[str] = "figure_eight"
    label: ClassVar[str] = "Figure-eight"

    amplitude_m: float
    period_s: float

    @property
    def peak_speed_mps(self):
        return figure_eight_peak_speed(self.amplitude_m, self.period_s)


@dataclass(frozen=True)
class SMCConfig:
    """Arguments of ``controllers.smc_figure_eight`` (same names)."""

    kind: ClassVar[str] = "smc"
    label: ClassVar[str] = "Sliding mode controller (SMC, lookahead point)"

    lookahead_L: float
    k_pos: float
    k_psi: float
    lambda_u: float
    lambda_r: float
    k_s_u: float
    k_s_r: float
    phi_u: float
    phi_r: float


@dataclass(frozen=True)
class FeedbackConfig:
    """Arguments of ``controllers.feedback_figure_eight`` (same names).

    With default gains this is exactly ``controllers.tau_figure_eight``.
    """

    kind: ClassVar[str] = "feedback"
    label: ClassVar[str] = "Simple feedback controller (tau_figure_eight)"

    k_pos: float
    k_u: float
    k_psi: float
    k_r: float


@dataclass(frozen=True)
class NoDisturbance:
    kind: ClassVar[str] = "none"
    label: ClassVar[str] = "No disturbance"


@dataclass(frozen=True)
class ConstantDisturbance:
    """``disturbances.constant_disturbance`` (body frame, additive)."""

    kind: ClassVar[str] = "constant"
    label: ClassVar[str] = "Constant"

    dTu: float
    dTv: float
    dTr: float


@dataclass(frozen=True)
class SinusoidalDisturbance:
    """``disturbances.sinusoidal_disturbance``; tuples are (Tu, Tv, Tr)."""

    kind: ClassVar[str] = "sinusoidal"
    label: ClassVar[str] = "Sinusoidal"

    amplitude: tuple
    frequency_hz: tuple
    phase_rad: tuple
    bias: tuple


@dataclass(frozen=True)
class DirectionalDisturbance:
    """``disturbances.directional_force_disturbance`` (inertial force)."""

    kind: ClassVar[str] = "directional"
    label: ClassVar[str] = "Directional force (fixed inertial direction)"

    magnitude_N: float
    direction_deg: float


@dataclass(frozen=True)
class WindLikeDisturbance:
    """``disturbances.wind_like_disturbance`` (fixed profile, no parameters)."""

    kind: ClassVar[str] = "wind"
    label: ClassVar[str] = "Wind-like (fixed profile)"


@dataclass(frozen=True)
class CombinedDisturbance:
    """``combined_disturbance`` of wind-like + directional + sinusoidal.

    Defaults reproduce the ``"combined"`` mode of ``run_simulation.py``.
    """

    kind: ClassVar[str] = "combined"
    label: ClassVar[str] = "Combined (wind-like + directional + sinusoidal)"

    include_wind: bool
    directional_magnitude_N: float
    directional_direction_deg: float
    sin_amplitude: tuple
    sin_frequency_hz: tuple
    sin_phase_rad: tuple


REFERENCE_TYPES = {cls.kind: cls for cls in (FigureEightReference,)}
CONTROLLER_TYPES = {cls.kind: cls for cls in (SMCConfig, FeedbackConfig)}
DISTURBANCE_TYPES = {
    cls.kind: cls
    for cls in (
        NoDisturbance,
        ConstantDisturbance,
        SinusoidalDisturbance,
        DirectionalDisturbance,
        WindLikeDisturbance,
        CombinedDisturbance,
    )
}

ReferenceConfig = FigureEightReference
ControllerConfig = Union[SMCConfig, FeedbackConfig]
DisturbanceConfig = Union[
    NoDisturbance,
    ConstantDisturbance,
    SinusoidalDisturbance,
    DirectionalDisturbance,
    WindLikeDisturbance,
    CombinedDisturbance,
]


def _section_to_dict(section):
    data = {"kind": section.kind} if hasattr(section, "kind") else {}
    for field in dataclasses.fields(section):
        value = getattr(section, field.name)
        data[field.name] = list(value) if isinstance(value, tuple) else value
    return data


def _section_from_dict(types, data):
    data = dict(data)
    cls = types[data.pop("kind")]
    kwargs = {}
    for field in dataclasses.fields(cls):
        value = data[field.name]
        kwargs[field.name] = tuple(float(v) for v in value) if isinstance(
            value, (list, tuple)
        ) else value
    return cls(**kwargs)


@dataclass(frozen=True)
class SimulationConfig:
    """Complete, immutable description of one dashboard experiment."""

    simulation: SimulationSettings
    reference: ReferenceConfig
    controller: ControllerConfig
    disturbance: DisturbanceConfig

    @property
    def laps(self):
        """Number of reference laps covered by the simulated time."""
        return self.simulation.t_final_s / self.reference.period_s

    @property
    def evaluation_lookahead_L(self):
        """Lookahead distance used to evaluate the lookahead-point error.

        SMC: its own lookahead; other controllers: simulator default.
        """
        return getattr(
            self.controller, "lookahead_L", controllers.TRACKING_LOOKAHEAD
        )

    # -- serialization -----------------------------------------------------
    def to_dict(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "simulation": _section_to_dict(self.simulation),
            "reference": _section_to_dict(self.reference),
            "controller": _section_to_dict(self.controller),
            "disturbance": _section_to_dict(self.disturbance),
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data):
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported configuration schema {version}.")
        return cls(
            simulation=SimulationSettings(**data["simulation"]),
            reference=_section_from_dict(REFERENCE_TYPES, data["reference"]),
            controller=_section_from_dict(CONTROLLER_TYPES, data["controller"]),
            disturbance=_section_from_dict(
                DISTURBANCE_TYPES, data["disturbance"]
            ),
        )

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def flat(self):
        """Flat ``{"section.field": value}`` view (for tables and diffs)."""
        flat = {}
        for section, values in self.to_dict().items():
            if section == "schema_version":
                continue
            for name, value in values.items():
                if isinstance(value, list):
                    for axis, item in zip(("Tu", "Tv", "Tr"), value):
                        flat[f"{section}.{name}[{axis}]"] = item
                else:
                    flat[f"{section}.{name}"] = value
        return flat


def config_diff(old, new):
    """List of ``(parameter, old value, new value)`` that differ."""
    old_flat = {} if old is None else old.flat()
    new_flat = {} if new is None else new.flat()
    keys = list(old_flat) + [k for k in new_flat if k not in old_flat]
    return [
        (key, old_flat.get(key, "—"), new_flat.get(key, "—"))
        for key in keys
        if not _same_value(old_flat.get(key, "—"), new_flat.get(key, "—"))
    ]


def _same_value(a, b):
    # Floats are compared with a tiny tolerance so that unit round trips
    # (e.g. rad -> deg -> rad when restoring a run) are not reported as
    # parameter changes.
    numeric = (int, float, np.floating, np.integer)
    if (isinstance(a, numeric) and isinstance(b, numeric)
            and not isinstance(a, bool) and not isinstance(b, bool)):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def same_experiment(a, b):
    """True if two configs describe the same experiment (see config_diff)."""
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

    def extend(self, other):
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


class ConfigValidationError(ValueError):
    """Raised when a configuration cannot be simulated."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("Invalid configuration: " + "; ".join(self.errors))


def _numbers(section):
    for field in dataclasses.fields(section):
        value = getattr(section, field.name)
        values = value if isinstance(value, tuple) else (value,)
        for item in values:
            yield field.name, item


def validate_config(config):
    """Check a ``SimulationConfig``; returns a ``ValidationReport``."""
    report = ValidationReport()
    errors, warnings = report.errors, report.warnings

    # Every numeric value must be a finite number.
    for section_name in ("simulation", "reference", "controller", "disturbance"):
        section = getattr(config, section_name)
        for name, value in _numbers(section):
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float, np.floating, np.integer)):
                errors.append(f"{section_name}.{name}: not a number ({value!r}).")
            elif not math.isfinite(float(value)):
                errors.append(f"{section_name}.{name}: must be finite.")
    if errors:
        return report

    sim = config.simulation
    if sim.t_final_s <= 0.0:
        errors.append(r"Total simulation time $T_{sim}$ must be > 0 s.")
    elif sim.t_final_s > MAX_T_FINAL_S:
        errors.append(
            f"Total simulation time $T_{{sim}}$ = {sim.t_final_s:g} s exceeds "
            f"the dashboard limit of {MAX_T_FINAL_S:g} s."
        )
    elif sim.t_final_s > LONG_RUN_WARNING_S:
        warnings.append(
            f"Long simulation ($T_{{sim}}$ = {sim.t_final_s:g} s): the run may "
            "take several "
            "minutes."
        )

    if sim.dt_s <= 0.0:
        errors.append(r"Output sampling interval $\Delta t$ must be > 0 s.")
    elif sim.t_final_s > 0.0:
        if sim.dt_s >= sim.t_final_s:
            errors.append(
                r"Output sampling interval $\Delta t$ must be smaller than the "
                r"total simulation time $T_{sim}$."
            )
        else:
            samples = int(sim.t_final_s / sim.dt_s) + 1
            if samples > MAX_OUTPUT_SAMPLES:
                errors.append(
                    rf"$T_{{sim}}/\Delta t$ gives {samples:,} output samples "
                    rf"(limit {MAX_OUTPUT_SAMPLES:,}); increase $\Delta t$."
                )
            ratio = sim.t_final_s / sim.dt_s
            if abs(ratio - round(ratio)) > 1e-6:
                actual = sim.t_final_s / (int(ratio))
                warnings.append(
                    r"$T_{sim}$ is not a multiple of $\Delta t$: the simulator "
                    "uses "
                    f"{int(ratio) + 1} evenly spaced samples "
                    f"(actual spacing {actual:.5g} s)."
                )

    u_max = max_steady_surge_speed()
    if abs(sim.u0_mps) > u_max:
        warnings.append(
            f"Initial surge speed $u_0$ = {sim.u0_mps:g} m/s exceeds the "
            rf"maximum steady speed under full thrust ($\approx$ {u_max:.2f} m/s)."
        )

    ref = config.reference
    if ref.amplitude_m <= 0.0:
        errors.append("Trajectory amplitude $A$ must be > 0 m.")
    if ref.period_s <= 0.0:
        errors.append("Trajectory period $T$ must be > 0 s.")
    if ref.amplitude_m > 0.0 and ref.period_s > 0.0:
        if ref.peak_speed_mps > u_max:
            warnings.append(
                f"Peak reference speed $v_{{peak}}$ = {ref.peak_speed_mps:.2f} m/s "
                rf"exceeds the maximum steady surge speed ($\approx$ {u_max:.2f} "
                f"m/s with "
                f"{TU_MAX:.1f} N total thrust): expect actuator saturation "
                "and growing tracking errors."
            )

    ctrl = config.controller
    if isinstance(ctrl, SMCConfig):
        if ctrl.lookahead_L <= 0.0:
            errors.append("SMC lookahead distance $L$ must be > 0 m.")
        for name, label in (
            ("k_pos", r"Position gain $k_{pos}$"),
            ("k_psi", r"Heading gain $k_{\psi}$"),
            ("lambda_u", r"Surge reaching gain $\lambda_u$"),
            ("lambda_r", r"Yaw-rate reaching gain $\lambda_r$"),
            ("k_s_u", r"Surge switching gain $k_{s,u}$"),
            ("k_s_r", r"Yaw switching gain $k_{s,r}$"),
        ):
            if getattr(ctrl, name) < 0.0:
                errors.append(f"{label} must be $\\geq$ 0.")
        for name, label in (
            ("phi_u", r"Surge boundary layer $\phi_u$"),
            ("phi_r", r"Yaw-rate boundary layer $\phi_r$"),
        ):
            value = getattr(ctrl, name)
            axis = name[-1]
            if value <= 0.0:
                errors.append(
                    f"{label} must be > 0 (it divides the sliding variable: "
                    rf"$\mathrm{{sat}}(s_{axis}/\phi_{axis})$)."
                )
            elif value < THIN_BOUNDARY_LAYER:
                warnings.append(
                    f"{label} = {value:g} is very thin: near-discontinuous "
                    "switching can make the adaptive integrator very slow."
                )
        if ctrl.k_pos == 0.0:
            warnings.append(
                "$k_{pos} = 0$: no position feedback, position errors will not "
                "be corrected."
            )
    elif isinstance(ctrl, FeedbackConfig):
        for name, symbol in (("k_pos", "k_{pos}"), ("k_u", "k_u"),
                             ("k_psi", r"k_{\psi}"), ("k_r", "k_r")):
            if getattr(ctrl, name) < 0.0:
                errors.append(f"Feedback gain ${symbol}$ must be $\\geq$ 0.")
    else:  # pragma: no cover - defensive
        errors.append(f"Unknown controller {type(ctrl).__name__}.")

    dist = config.disturbance
    if isinstance(dist, ConstantDisturbance):
        if abs(dist.dTu) >= TU_MAX:
            warnings.append(
                rf"$|\delta T_u|$ = {abs(dist.dTu):g} N exceeds the surge force "
                f"authority ({TU_MAX:.1f} N)."
            )
        if abs(dist.dTr) >= TR_MAX:
            warnings.append(
                rf"$|\delta T_r|$ = {abs(dist.dTr):g} N m exceeds the yaw moment "
                f"authority ({TR_MAX:.1f} N m)."
            )
    if isinstance(dist, SinusoidalDisturbance):
        if any(f < 0.0 for f in dist.frequency_hz):
            errors.append(r"Sinusoidal disturbance frequencies $f_i$ must be $\geq$ 0 Hz.")
    if isinstance(dist, DirectionalDisturbance):
        if dist.magnitude_N < 0.0:
            errors.append(r"Directional disturbance magnitude $F$ must be $\geq$ 0 N.")
    if isinstance(dist, CombinedDisturbance):
        if dist.directional_magnitude_N < 0.0:
            errors.append(r"Directional component magnitude $F$ must be $\geq$ 0 N.")
        if any(f < 0.0 for f in dist.sin_frequency_hz):
            errors.append(r"Sinusoidal component frequencies $f_i$ must be $\geq$ 0 Hz.")
    for name, value in _numbers(dist):
        if "frequency" in name and value > 0.0 and sim.dt_s > 0.0:
            if value > 0.5 / sim.dt_s:
                warnings.append(
                    f"Disturbance frequency {value:g} Hz is above the output "
                    rf"Nyquist frequency $1/(2\Delta t)$ = {0.5 / sim.dt_s:g} Hz; "
                    "plots will "
                    "alias it (the integration itself is unaffected)."
                )
                break

    return report


# ---------------------------------------------------------------------------
# Form (widget) state.
# ---------------------------------------------------------------------------
AXES = ("u", "v", "r")  # component suffixes for (Tu, Tv, Tr)

# Defaults of run_simulation.DISTURBANCE_OPTIONS (closures cannot be
# introspected, so the literal values are mirrored here; a unit test checks
# that both produce identical disturbances).
_CONSTANT_DEFAULT = (0.0, 0.0, 15.0)
_PURE_SIN_DEFAULT = {  # disturbances.pure_sinusoidal_disturbance
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
    """Widget defaults, taken from the current simulator configuration."""
    amplitude = float(controllers.FIGURE_EIGHT_AMPLITUDE)
    period = float(controllers.FIGURE_EIGHT_PERIOD)
    x0, y0, psi0, u0, v0, r0 = (
        float(v) for v in controllers.figure_eight_initial_state()
    )
    form = {
        # Simulation settings
        "sim_duration_mode": "laps",
        "sim_laps": float(controllers.FIGURE_EIGHT_LAPS),
        "sim_t_final_s": period * float(controllers.FIGURE_EIGHT_LAPS),
        "sim_dt_s": float(run_simulation.DT),
        "sim_initial_mode": "reference",
        "sim_x0_m": x0,
        "sim_y0_m": y0,
        "sim_psi0_deg": math.degrees(psi0),
        "sim_u0_mps": u0,
        "sim_v0_mps": v0,
        "sim_r0_degps": math.degrees(r0),
        # Reference
        "ref_kind": FigureEightReference.kind,
        "ref_timing_mode": "period",
        "ref_amplitude_m": amplitude,
        "ref_period_s": period,
        "ref_peak_speed_mps": figure_eight_peak_speed(amplitude, period),
        # Controller
        "ctrl_kind": SMCConfig.kind,
        "smc_lookahead_L": float(_SMC_DEFAULTS["lookahead_L"]),
        "fb_k_pos": float(_FEEDBACK_DEFAULTS["k_pos"]),
        "fb_k_u": float(_FEEDBACK_DEFAULTS["k_u"]),
        "fb_k_psi": float(_FEEDBACK_DEFAULTS["k_psi"]),
        "fb_k_r": float(_FEEDBACK_DEFAULTS["k_r"]),
        # Disturbance
        "dist_kind": run_simulation.DISTURBANCE_MODE
        if run_simulation.DISTURBANCE_MODE in DISTURBANCE_TYPES
        else NoDisturbance.kind,
        "dist_const_dTu": _CONSTANT_DEFAULT[0],
        "dist_const_dTv": _CONSTANT_DEFAULT[1],
        "dist_const_dTr": _CONSTANT_DEFAULT[2],
        "dist_dir_magnitude_N": _DIRECTIONAL_DEFAULT[0],
        "dist_dir_direction_deg": _DIRECTIONAL_DEFAULT[1],
        "dist_comb_wind": True,
        "dist_comb_dir_magnitude_N": _COMBINED_DEFAULT["directional"][0],
        "dist_comb_dir_direction_deg": _COMBINED_DEFAULT["directional"][1],
    }
    for name in ("k_pos", "k_psi", "lambda_u", "lambda_r",
                 "k_s_u", "k_s_r", "phi_u", "phi_r"):
        form[f"smc_{name}"] = float(_SMC_DEFAULTS[name])
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
    return float(form[key])


def _triple(form, prefix):
    return tuple(_f(form, f"{prefix}_{axis}") for axis in AXES)


def form_to_config(form):
    """Build a ``SimulationConfig`` from widget values.

    Raises ``ConfigValidationError`` for inputs that cannot be converted
    (e.g. a zero peak speed or a non-positive number of laps).
    """
    errors = []
    amplitude = _f(form, "ref_amplitude_m")

    if form["ref_timing_mode"] == "peak_speed":
        peak_speed = _f(form, "ref_peak_speed_mps")
        if not peak_speed > 0.0 or not math.isfinite(peak_speed):
            errors.append("Peak reference speed $v_{peak}$ must be > 0 m/s.")
            period = float("nan")
        else:
            period = figure_eight_period_from_peak_speed(amplitude, peak_speed)
    else:
        period = _f(form, "ref_period_s")

    if form["sim_duration_mode"] == "laps":
        laps = _f(form, "sim_laps")
        if not laps > 0.0 or not math.isfinite(laps):
            errors.append("Number of laps $N_{laps}$ must be > 0.")
        t_final = laps * period
    else:
        t_final = _f(form, "sim_t_final_s")

    if errors:
        raise ConfigValidationError(errors)

    if form["sim_initial_mode"] == "reference":
        if amplitude > 0.0 and period > 0.0:
            x0, y0, psi0, u0, v0, r0 = (
                float(v)
                for v in controllers.figure_eight_initial_state(amplitude, period)
            )
        else:  # invalid reference; validation will report it
            x0 = y0 = psi0 = u0 = v0 = r0 = 0.0
    else:
        x0, y0 = _f(form, "sim_x0_m"), _f(form, "sim_y0_m")
        psi0 = math.radians(_f(form, "sim_psi0_deg"))
        u0, v0 = _f(form, "sim_u0_mps"), _f(form, "sim_v0_mps")
        r0 = math.radians(_f(form, "sim_r0_degps"))

    simulation = SimulationSettings(
        t_final_s=float(t_final), dt_s=_f(form, "sim_dt_s"),
        x0_m=x0, y0_m=y0, psi0_rad=psi0, u0_mps=u0, v0_mps=v0, r0_radps=r0,
    )
    reference = FigureEightReference(amplitude_m=amplitude, period_s=float(period))

    kind = form["ctrl_kind"]
    if kind == SMCConfig.kind:
        controller = SMCConfig(
            lookahead_L=_f(form, "smc_lookahead_L"),
            **{
                name: _f(form, f"smc_{name}")
                for name in ("k_pos", "k_psi", "lambda_u", "lambda_r",
                             "k_s_u", "k_s_r", "phi_u", "phi_r")
            },
        )
    elif kind == FeedbackConfig.kind:
        controller = FeedbackConfig(
            **{name: _f(form, f"fb_{name}")
               for name in ("k_pos", "k_u", "k_psi", "k_r")}
        )
    else:
        raise ConfigValidationError([f"Unknown controller '{kind}'."])

    kind = form["dist_kind"]
    if kind == NoDisturbance.kind:
        disturbance = NoDisturbance()
    elif kind == ConstantDisturbance.kind:
        disturbance = ConstantDisturbance(
            dTu=_f(form, "dist_const_dTu"),
            dTv=_f(form, "dist_const_dTv"),
            dTr=_f(form, "dist_const_dTr"),
        )
    elif kind == SinusoidalDisturbance.kind:
        disturbance = SinusoidalDisturbance(
            amplitude=_triple(form, "dist_sin_amp"),
            frequency_hz=_triple(form, "dist_sin_freq"),
            phase_rad=_triple(form, "dist_sin_phase"),
            bias=_triple(form, "dist_sin_bias"),
        )
    elif kind == DirectionalDisturbance.kind:
        disturbance = DirectionalDisturbance(
            magnitude_N=_f(form, "dist_dir_magnitude_N"),
            direction_deg=_f(form, "dist_dir_direction_deg"),
        )
    elif kind == WindLikeDisturbance.kind:
        disturbance = WindLikeDisturbance()
    elif kind == CombinedDisturbance.kind:
        disturbance = CombinedDisturbance(
            include_wind=bool(form["dist_comb_wind"]),
            directional_magnitude_N=_f(form, "dist_comb_dir_magnitude_N"),
            directional_direction_deg=_f(form, "dist_comb_dir_direction_deg"),
            sin_amplitude=_triple(form, "dist_comb_amp"),
            sin_frequency_hz=_triple(form, "dist_comb_freq"),
            sin_phase_rad=_triple(form, "dist_comb_phase"),
        )
    else:
        raise ConfigValidationError([f"Unknown disturbance '{kind}'."])

    return SimulationConfig(simulation, reference, controller, disturbance)


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

    Widgets of non-selected controllers/disturbances keep ``base`` values.
    """
    form = dict(default_form() if base is None else base)
    sim, ref = config.simulation, config.reference

    form["ref_kind"] = ref.kind
    form["ref_timing_mode"] = "period"
    form["ref_amplitude_m"] = ref.amplitude_m
    form["ref_period_s"] = ref.period_s
    form["ref_peak_speed_mps"] = ref.peak_speed_mps

    form["sim_dt_s"] = sim.dt_s
    laps = sim.t_final_s / ref.period_s
    form["sim_laps"] = laps
    form["sim_t_final_s"] = sim.t_final_s
    form["sim_duration_mode"] = (
        "laps" if laps * ref.period_s == sim.t_final_s else "manual"
    )

    reference_state = controllers.figure_eight_initial_state(
        ref.amplitude_m, ref.period_s
    )
    form["sim_initial_mode"] = (
        "reference"
        if np.array_equal(reference_state, sim.initial_state())
        else "custom"
    )
    form["sim_x0_m"], form["sim_y0_m"] = sim.x0_m, sim.y0_m
    form["sim_psi0_deg"] = math.degrees(sim.psi0_rad)
    form["sim_u0_mps"], form["sim_v0_mps"] = sim.u0_mps, sim.v0_mps
    form["sim_r0_degps"] = math.degrees(sim.r0_radps)

    ctrl = config.controller
    form["ctrl_kind"] = ctrl.kind
    prefix = "smc" if isinstance(ctrl, SMCConfig) else "fb"
    for field in dataclasses.fields(ctrl):
        form[f"{prefix}_{field.name}"] = float(getattr(ctrl, field.name))

    dist = config.disturbance
    form["dist_kind"] = dist.kind
    if isinstance(dist, ConstantDisturbance):
        form.update(dist_const_dTu=dist.dTu, dist_const_dTv=dist.dTv,
                    dist_const_dTr=dist.dTr)
    elif isinstance(dist, SinusoidalDisturbance):
        for i, axis in enumerate(AXES):
            form[f"dist_sin_amp_{axis}"] = dist.amplitude[i]
            form[f"dist_sin_freq_{axis}"] = dist.frequency_hz[i]
            form[f"dist_sin_phase_{axis}"] = dist.phase_rad[i]
            form[f"dist_sin_bias_{axis}"] = dist.bias[i]
    elif isinstance(dist, DirectionalDisturbance):
        form["dist_dir_magnitude_N"] = dist.magnitude_N
        form["dist_dir_direction_deg"] = dist.direction_deg
    elif isinstance(dist, CombinedDisturbance):
        form["dist_comb_wind"] = dist.include_wind
        form["dist_comb_dir_magnitude_N"] = dist.directional_magnitude_N
        form["dist_comb_dir_direction_deg"] = dist.directional_direction_deg
        for i, axis in enumerate(AXES):
            form[f"dist_comb_amp_{axis}"] = dist.sin_amplitude[i]
            form[f"dist_comb_freq_{axis}"] = dist.sin_frequency_hz[i]
            form[f"dist_comb_phase_{axis}"] = dist.sin_phase_rad[i]

    return {key: (float(value) if isinstance(value, (int, float, np.floating))
                  and not isinstance(value, bool) else value)
            for key, value in form.items()}


def default_config():
    """``SimulationConfig`` equivalent to the simulator defaults."""
    return form_to_config(default_form())


def form_period(form):
    """Reference period [s] implied by the widget values (None if invalid)."""
    try:
        amplitude = float(form["ref_amplitude_m"])
        if form["ref_timing_mode"] == "peak_speed":
            speed = float(form["ref_peak_speed_mps"])
            period = figure_eight_period_from_peak_speed(amplitude, speed)
        else:
            period = float(form["ref_period_s"])
    except (ZeroDivisionError, ValueError, TypeError):
        return None
    return period if math.isfinite(period) and period > 0.0 else None
