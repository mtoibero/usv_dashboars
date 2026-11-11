"""Gain estimates for the sliding mode controllers (no Streamlit imports).

With the model inversion of ``controllers.smc_figure_eight`` each sliding
variable obeys

    s_dot = -lambda s - k_s sat(s/phi) + d(t)

where ``d`` is the *matched uncertainty* in acceleration units:

* the external disturbance of the same channel, ``delta T_u / z1`` (surge) or
  ``delta T_r / z3`` (yaw), bounded with ``DisturbanceConfig.bounds`` (worst
  case over time and heading), and
* what the controller itself neglects along the trajectory: the approximate
  feed-forward ``u_dot_cmd`` (surge) and the missing ``r_dot_cmd`` feed-forward
  (yaw).  This part does not depend on the disturbance but on the reference
  (it grows roughly with the square of its frequency).

Outside the boundary layer (|s| >= phi) the sliding condition
``s s_dot <= -eta |s|`` holds when

    k_s + lambda phi >= d_bar + eta,        d_bar = sup |d(t)|.

With a digital controller (sample period Ts) the boundary layer does not
chatter when ``Ts (lambda + k_s/phi) < 1``, i.e. ``k_s < phi (f_s - lambda)``.
Both conditions leave a window for ``k_s``, which exists only if the layer is
wide enough.

The trajectory part is evaluated with the controller's own code
(``controller.signals``) on states that lie exactly on the reference, so the
estimate follows the controller if it changes.  The estimates ignore thruster
saturation (checked separately by ``actuator_check``) and the remaining
modelling approximations; this is why the suggested gain adds a margin.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

from dashboard02_smc import PROJECT_ROOT  # noqa: F401
from dashboard02_smc import config as cfg

import controllers
from usv_model import PARAMS, THRUSTER_ARM, THRUSTER_FWD_MAX, THRUSTER_REV_MAX

MARGIN = 0.25            # eta = 25 % of d_bar in the suggested gain
SAMPLES_PER_PERIOD = 1200


# ---------------------------------------------------------------------------
# Trajectory-induced matched uncertainty.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrajectoryLoad:
    d_traj_u: float          # max |u_dot_cmd (controller) - d u_cmd/dt|  [m/s^2]
    d_traj_r: float          # max |d r_cmd / dt| on the surface          [rad/s^2]
    t: np.ndarray
    u_cmd: np.ndarray        # commanded surge speed along the reference
    r_cmd: np.ndarray        # commanded yaw rate along the reference


def _periodic_derivative(y, dt):
    return (np.roll(y, -1) - np.roll(y, 1)) / (2.0 * dt)


@functools.lru_cache(maxsize=64)
def trajectory_load(amplitude, period, lookahead_L, n=SAMPLES_PER_PERIOD):
    """Matched uncertainty created by the reference itself (one period).

    The vessel is placed exactly on the reference (p_s = p_s,ref,
    p_s_dot = p_s,ref_dot, psi = psi_cmd), which is where it sits while
    sliding, and ``controller.signals`` gives the commands.
    """
    controller = controllers.smc_figure_eight(
        lookahead_L=lookahead_L, amplitude=amplitude, period=period)
    t = np.linspace(0.0, period, n, endpoint=False)
    p_s_ref, p_s_dot_ref, _, _ = controllers.figure_eight_lookahead_reference(
        t, lookahead_L, amplitude=amplitude, period=period)
    psi = np.arctan2(p_s_dot_ref[:, 1], p_s_dot_ref[:, 0])
    c, s = np.cos(psi), np.sin(psi)
    x = p_s_ref[:, 0] - lookahead_L * c
    y = p_s_ref[:, 1] - lookahead_L * s
    # Body velocity of the lookahead point: [u, v + L r] = R(psi)^T p_s_dot
    u = c * p_s_dot_ref[:, 0] + s * p_s_dot_ref[:, 1]
    v = -s * p_s_dot_ref[:, 0] + c * p_s_dot_ref[:, 1]      # with r = 0
    signals = [controller.signals(ti, np.array([xi, yi, pi, ui, vi, 0.0]))
               for ti, xi, yi, pi, ui, vi in zip(t, x, y, psi, u, v)]
    u_cmd = np.array([sg["speed_cmd"] for sg in signals])
    u_dot_ctrl = np.array([sg["speed_dot_cmd"] for sg in signals])
    r_cmd = np.array([sg["r_cmd"] for sg in signals])
    dt = t[1] - t[0]
    d_u = np.max(np.abs(u_dot_ctrl - _periodic_derivative(u_cmd, dt)))
    d_r = np.max(np.abs(_periodic_derivative(r_cmd, dt)))
    return TrajectoryLoad(float(d_u), float(d_r), t, u_cmd, r_cmd)


def load_for(config):
    ref = config.reference
    return trajectory_load(float(ref.amplitude_m), float(ref.period_s),
                           float(config.simulation.lookahead_L_m))


# ---------------------------------------------------------------------------
# Gain hint per channel.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GainHint:
    channel: str             # "surge" or "yaw"
    axis: str                # "u" or "r"
    z: float                 # inertia of the channel (z1 or z3)
    z_symbol: str
    acc_unit: str            # unit of d, k_s: "m/s^2" or "rad/s^2"
    s_unit: str              # unit of s, phi
    force_unit: str          # "N" or "N m"
    lam: float
    k_s: float
    phi: float
    d_ext: float             # external disturbance bound / z
    d_traj: float            # trajectory-induced part
    sample_rate_hz: float | None

    @property
    def d_bar(self):
        return self.d_ext + self.d_traj

    # -- sliding condition: k_s + lambda phi >= d_bar (+ eta) ----------------
    @property
    def k_min(self):
        """Smallest k_s meeting the sliding condition (eta = 0)."""
        return max(0.0, self.d_bar - self.lam * self.phi)

    @property
    def k_suggested(self):
        """k_s = d_bar + eta with eta = MARGIN d_bar, without relying on the
        help of the linear term (classic choice k = F + eta)."""
        return (1.0 + MARGIN) * self.d_bar

    @property
    def sliding_ok(self):
        return self.k_s + self.lam * self.phi >= self.d_bar

    @property
    def margin_pct(self):
        """Robustness margin of the current gains (k_s + lambda phi)/d_bar - 1."""
        if self.d_bar <= 0.0:
            return math.inf
        return 100.0 * ((self.k_s + self.lam * self.phi) / self.d_bar - 1.0)

    @property
    def lam_min(self):
        """Smallest lambda meeting the sliding condition with the current k_s."""
        if self.phi <= 0.0:
            return math.inf
        return max(0.0, (self.d_bar - self.k_s) / self.phi)

    @property
    def phi_min_sliding(self):
        """Smallest phi meeting the sliding condition with the current k_s, lambda."""
        if self.k_s >= self.d_bar:
            return 0.0
        if self.lam <= 0.0:
            return math.inf
        return (self.d_bar - self.k_s) / self.lam

    @property
    def tolerable_disturbance(self):
        """Largest |delta T| [N or N m] the current gains tolerate (eta = 0)."""
        return max(0.0, self.z * (self.k_s + self.lam * self.phi - self.d_traj))

    # -- digital implementation: no chattering for Gamma < 1 ------------------
    @property
    def k_max_chattering(self):
        """Largest k_s with Gamma = Ts (lambda + k_s/phi) < 1 (None: continuous)."""
        if self.sample_rate_hz is None:
            return None
        return max(0.0, self.phi * (self.sample_rate_hz - self.lam))

    @property
    def chattering_ok(self):
        k_max = self.k_max_chattering
        return k_max is None or self.k_s < k_max

    @property
    def phi_min_window(self):
        """Smallest phi for which k_s,suggested stays below the chattering
        limit (None: continuous)."""
        if self.sample_rate_hz is None:
            return None
        if self.sample_rate_hz <= self.lam:
            return math.inf
        return self.k_suggested / (self.sample_rate_hz - self.lam)

    @property
    def window_exists(self):
        k_max = self.k_max_chattering
        return k_max is None or self.k_suggested < k_max


def gain_hints(config):
    """``{"surge": GainHint, "yaw": GainHint}`` for ``config``."""
    load = load_for(config)
    sim = config.simulation
    bound_u, _bound_v, bound_r = config.disturbance.bounds()
    rate = sim.sample_rate_hz if sim.mode == cfg.DIGITAL else None
    return {
        "surge": GainHint(
            "surge", "u", PARAMS["z1"], "z_1", "m/s^2", "m/s", "N",
            config.surge.lambda_u, config.surge.k_s_u, config.surge.phi_u,
            bound_u / PARAMS["z1"], load.d_traj_u, rate),
        "yaw": GainHint(
            "yaw", "r", PARAMS["z3"], "z_3", "rad/s^2", "rad/s", "N m",
            config.yaw.lambda_r, config.yaw.k_s_r, config.yaw.phi_r,
            bound_r / PARAMS["z3"], load.d_traj_r, rate),
    }


def suggested_value(value):
    """Round a suggested gain up to 2 significant digits."""
    if value <= 0.0 or not math.isfinite(value):
        return 0.0
    exponent = math.floor(math.log10(value)) - 1
    step = 10.0 ** exponent
    return round(math.ceil(value / step - 1e-9) * step, 10)


# ---------------------------------------------------------------------------
# Actuator check: can the thrusters deliver trajectory + disturbance?
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ActuatorCheck:
    utilization_pct: float   # worst thruster, nominal motion + worst disturbance
    tu_peak: float           # peak surge force of the nominal motion [N]
    tr_peak: float           # peak yaw moment of the nominal motion [N m]

    @property
    def ok(self):
        return self.utilization_pct < 100.0


def actuator_check(config):
    """Worst-case thruster use for the reference plus the disturbances.

    Nominal forces of the motion on the reference (sway neglected):
    T_u = z4 u + z5 |u| u + z1 u_dot,  T_r = z8 r + z9 |r| r + z3 r_dot, with
    u = u_cmd(t), r = r_cmd(t).  The thrusters must also cancel the matched
    disturbances, taken with their worst sign.
    """
    load = load_for(config)
    dt = load.t[1] - load.t[0]
    u, r = load.u_cmd, load.r_cmd
    tu = (PARAMS["z4"] * u + PARAMS["z5"] * np.abs(u) * u
          + PARAMS["z1"] * _periodic_derivative(u, dt))
    tr = (PARAMS["z8"] * r + PARAMS["z9"] * np.abs(r) * r
          + PARAMS["z3"] * _periodic_derivative(r, dt))
    bound_u, _bound_v, bound_r = config.disturbance.bounds()
    extra = 0.25 * (bound_u + bound_r / THRUSTER_ARM)
    worst = 0.0
    for side in (-1.0, 1.0):                      # left, right thrusters
        force = 0.25 * (tu + side * tr / THRUSTER_ARM)
        worst = max(worst, float(np.max(force + extra)) / THRUSTER_FWD_MAX,
                    float(np.max(extra - force)) / THRUSTER_REV_MAX)
    return ActuatorCheck(100.0 * worst, float(np.max(np.abs(tu))),
                         float(np.max(np.abs(tr))))
