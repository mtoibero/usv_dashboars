import numpy as np


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def position_error(df):
    if {"x_s", "y_s", "x_s_ref", "y_s_ref"}.issubset(df.columns):
        ex = df["x_s_ref"].to_numpy() - df["x_s"].to_numpy()
        ey = df["y_s_ref"].to_numpy() - df["y_s"].to_numpy()
    else:
        ex = df["x_ref"].to_numpy() - df["x"].to_numpy()
        ey = df["y_ref"].to_numpy() - df["y"].to_numpy()

    return np.sqrt(ex**2 + ey**2)


def is_saturated(force, fwd_max, rev_max, tol):
    return (
        np.isclose(force, fwd_max, atol=tol)
        | np.isclose(force, rev_max, atol=tol)
    )


def saturation_duty_cycle(df, fwd_max, rev_max, tol):
    result = {}
    n = len(df)

    for thruster in ["F1", "F2", "F3", "F4"]:
        sat = is_saturated(df[thruster].to_numpy(), fwd_max, rev_max, tol)
        result[thruster] = float(np.sum(sat) / n)

    return result
