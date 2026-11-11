"""Form fragments shared by the tests (disturbance models, initial offset)."""


def constant(d_u=0.0, d_v=0.0, d_r=0.0):
    return dict(dist_kind="constant", dist_const_dTu=float(d_u),
                dist_const_dTv=float(d_v), dist_const_dTr=float(d_r))


def sinusoid(axis, amplitude, frequency, phase=0.0, bias=0.0):
    """Sinusoidal model with a single non-zero axis ('u', 'v' or 'r')."""
    form = dict(dist_kind="sinusoidal")
    for other in ("u", "v", "r"):
        form[f"dist_sin_amp_{other}"] = 0.0
        form[f"dist_sin_bias_{other}"] = 0.0
        form[f"dist_sin_phase_{other}"] = 0.0
    form[f"dist_sin_amp_{axis}"] = float(amplitude)
    form[f"dist_sin_freq_{axis}"] = float(frequency)
    form[f"dist_sin_phase_{axis}"] = float(phase)
    form[f"dist_sin_bias_{axis}"] = float(bias)
    return form


def offset(dx=0.0, dy=0.0, dpsi_deg=0.0, du=0.0):
    return dict(sim_initial_mode="offset", init_dx_m=float(dx),
                init_dy_m=float(dy), init_dpsi_deg=float(dpsi_deg),
                init_du_mps=float(du))
