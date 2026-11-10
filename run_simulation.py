from pathlib import Path

import numpy as np

from cbf_filters import (
    CBF_ALPHA_1,
    CBF_ALPHA_2,
    CBF_BARRIER_COLLISION_CONE,
    CBF_BARRIER_ELLIPSE,
    CBF_BARRIER_TYPE,
    CBF_ELLIPSE_ANGLE_DEG,
    CBF_ELLIPSE_MAJOR_SCALE,
    CBF_ELLIPSE_MINOR_SCALE,
    CBF_MARGIN,
    cbf_filtered_controller,
    cbf_history,
)
from controllers import (
    FIGURE_EIGHT_AMPLITUDE,
    FIGURE_EIGHT_LAPS,
    FIGURE_EIGHT_PERIOD,
    TRACKING_LOOKAHEAD,
    figure_eight_initial_state,
    figure_eight_lookahead_reference,
    figure_eight_reference,
    lookahead_point_from_state,
    smc_figure_eight,
)
from disturbances import (
    combined_disturbance,
    constant_disturbance,
    directional_force_disturbance,
    no_disturbance,
    pure_sinusoidal_disturbance,
    sinusoidal_disturbance,
    wind_like_disturbance,
)
from obstacle_config import configured_obstacles
from plot_results import plot_results, save_simulation_animation
from reference_governor import (
    GOVERNOR_GAMMA_MIN,
    GOVERNOR_FILTER_TAU_MAX,
    GOVERNOR_FILTER_TAU_MIN,
    GOVERNOR_H_SLOW,
    GOVERNOR_H_STOP,
    governor_history,
    virtual_time_governor,
)
from usv_model import TR_LIMITS, TU_LIMITS, simulate, thruster_forces_to_pwm_us


DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "figure_eight_simulation.csv"
FIGURES_DIR = DATA_DIR / "figures"
ANIMATION_PATH = FIGURES_DIR / "Trayectory_animation.gif"
DT = 0.02
SHOW_PLOTS = False
SAVE_ANIMATION = True
ANIMATION_FPS = 25
ANIMATION_DURATION = 20.0
ENABLE_OBSTACLES = True
LOOKAHEAD_L = TRACKING_LOOKAHEAD
ENABLE_CBF_AVOIDANCE = True
ENABLE_CBF_DIAGNOSTICS = True
CBF_SAFETY_MARGIN = CBF_MARGIN
CBF_RELATIVE_DEGREE_ALPHA_1 = CBF_ALPHA_1
CBF_RELATIVE_DEGREE_ALPHA_2 = CBF_ALPHA_2
# Strategy 3 del paper: usar CBF_BARRIER_COLLISION_CONE.
#CBF_SELECTED_BARRIER = CBF_BARRIER_TYPE
CBF_SELECTED_BARRIER = CBF_BARRIER_ELLIPSE
#CBF_SELECTED_BARRIER = CBF_BARRIER_COLLISION_CONE
#   las opciones son _type para el circ elipse y cono
CBF_SELECTED_ELLIPSE_MAJOR_SCALE = CBF_ELLIPSE_MAJOR_SCALE
CBF_SELECTED_ELLIPSE_MINOR_SCALE = CBF_ELLIPSE_MINOR_SCALE
CBF_SELECTED_ELLIPSE_ANGLE_DEG = CBF_ELLIPSE_ANGLE_DEG
ENABLE_REFERENCE_GOVERNOR = True
REFERENCE_GAMMA_MIN = GOVERNOR_GAMMA_MIN
REFERENCE_GOVERNOR_H_SLOW = GOVERNOR_H_SLOW
REFERENCE_GOVERNOR_H_STOP = GOVERNOR_H_STOP
REFERENCE_GAMMA_FILTER_TAU_MIN = GOVERNOR_FILTER_TAU_MIN
REFERENCE_GAMMA_FILTER_TAU_MAX = GOVERNOR_FILTER_TAU_MAX


DISTURBANCE_OPTIONS = {
    "none": no_disturbance,
    "wind": wind_like_disturbance,
    "directional": directional_force_disturbance(
        magnitude=25.0,
        direction_deg=45.0,
    ),
    "constant": constant_disturbance(
        dTu=0.0,
        dTv=0.0,
        dTr=15.0,
    ),
    "sinusoidal": pure_sinusoidal_disturbance,
    "combined": combined_disturbance(
        wind_like_disturbance,
        directional_force_disturbance(
            magnitude=7.0,
            direction_deg=64.0,
        ),
        sinusoidal_disturbance(
            amplitude=(6.0, 3.0, 1.5),
            frequency=(0.04, 0.06, 0.03),
            phase=(0.0, 0.5, 1.0),
        ),
    ),
}
DISTURBANCE_MODE = "none"


def state_with_reference_rate(state_i, gamma=None):
    """Agrega gamma a un estado aumentado [.., sigma] para evaluar control."""
    if gamma is None or len(state_i) < 7:
        return state_i

    return np.concatenate([state_i, [gamma]])


def controller_output_history(controller, t, state, gamma_history=None):
    """Calcula tau requerido, tau real y fuerzas de propulsores."""
    if hasattr(controller, "output"):
        output = [
            controller.output(
                ti,
                state_with_reference_rate(
                    state_i,
                    None if gamma_history is None else gamma_i,
                ),
            )
            for ti, state_i, gamma_i in zip(
                t,
                state,
                np.ones(len(t)) if gamma_history is None else gamma_history,
            )
        ]
        tau_required, tau_delivered, thruster_forces = (
            np.array(values)
            for values in zip(*output)
        )
    else:
        tau_required = np.array([
            controller.required_tau(ti, state_i)
            for ti, state_i in zip(t, state)
        ])
        tau_delivered = np.array([
            controller.delivered_tau(ti, state_i)
            for ti, state_i in zip(t, state)
        ])
        thruster_forces = np.array([
            controller.thruster_forces(ti, state_i)
            for ti, state_i in zip(t, state)
        ])

    return tau_required, tau_delivered, thruster_forces


def reference_history(
    t,
    state=None,
    lookahead_L=LOOKAHEAD_L,
    gamma_history=None,
    amplitude=FIGURE_EIGHT_AMPLITUDE,
    period=FIGURE_EIGHT_PERIOD,
):
    """Calcula la referencia del centro y del punto adelantado."""
    reference_time = t
    reference_rate = 1.0 if gamma_history is None else gamma_history
    if state is not None and state.shape[1] >= 7:
        reference_time = state[:, 6]

    x_ref, y_ref, x_dot_ref, y_dot_ref, x_ddot_ref, y_ddot_ref = (
        figure_eight_reference(reference_time, amplitude, period)
    )
    x_dot_ref = reference_rate * x_dot_ref
    y_dot_ref = reference_rate * y_dot_ref
    x_ddot_ref = reference_rate**2 * x_ddot_ref
    y_ddot_ref = reference_rate**2 * y_ddot_ref

    psi_ref = np.arctan2(y_dot_ref, x_dot_ref)
    u_ref = np.hypot(x_dot_ref, y_dot_ref)
    speed_squared = np.maximum(u_ref**2, 1e-12)
    r_ref = (
        x_dot_ref * y_ddot_ref
        - y_dot_ref * x_ddot_ref
    ) / speed_squared

    p_s_ref, p_s_dot_ref, _, _ = figure_eight_lookahead_reference(
        reference_time,
        lookahead_L,
        time_scale=reference_rate,
        amplitude=amplitude,
        period=period,
    )

    if state is None:
        return (
            x_ref,
            y_ref,
            psi_ref,
            u_ref,
            r_ref,
            p_s_ref[:, 0],
            p_s_ref[:, 1],
            p_s_dot_ref[:, 0],
            p_s_dot_ref[:, 1],
        )

    p_s, p_s_dot = lookahead_point_from_state(state, lookahead_L)

    return (
        x_ref,
        y_ref,
        psi_ref,
        u_ref,
        r_ref,
        p_s[:, 0],
        p_s[:, 1],
        p_s_dot[:, 0],
        p_s_dot[:, 1],
        p_s_ref[:, 0],
        p_s_ref[:, 1],
        p_s_dot_ref[:, 0],
        p_s_dot_ref[:, 1],
    )


def save_csv(
    path,
    t,
    state,
    controller=None,
    controller_history=None,
    cbf_history_data=None,
    governor_history_data=None,
    lookahead_L=LOOKAHEAD_L,
    amplitude=FIGURE_EIGHT_AMPLITUDE,
    period=FIGURE_EIGHT_PERIOD,
):
    """Guarda la simulacion en CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data_columns = [t, state]
    header_columns = ["t", "x", "y", "psi", "u", "v", "r"]
    gamma_history = None

    if state.shape[1] >= 7:
        header_columns.append("sigma")

    if governor_history_data is not None:
        gamma_history = governor_history_data["gamma"]
        data_columns.append(gamma_history)
        header_columns.append("gamma")
        data_columns.extend([
            governor_history_data["gamma_target"],
            governor_history_data["tau_gamma"],
        ])
        header_columns.extend([
            "gamma_target",
            "gamma_filter_tau",
        ])

    (
        x_ref,
        y_ref,
        psi_ref,
        u_ref,
        r_ref,
        x_s,
        y_s,
        x_s_dot,
        y_s_dot,
        x_s_ref,
        y_s_ref,
        x_s_dot_ref,
        y_s_dot_ref,
    ) = reference_history(
        t,
        state,
        lookahead_L,
        gamma_history,
        amplitude=amplitude,
        period=period,
    )
    data_columns.extend([
        x_ref,
        y_ref,
        psi_ref,
        u_ref,
        r_ref,
        x_s,
        y_s,
        x_s_dot,
        y_s_dot,
        x_s_ref,
        y_s_ref,
        x_s_dot_ref,
        y_s_dot_ref,
    ])
    header_columns.extend([
        "x_ref",
        "y_ref",
        "psi_ref",
        "u_ref",
        "r_ref",
        "x_s",
        "y_s",
        "x_s_dot",
        "y_s_dot",
        "x_s_ref",
        "y_s_ref",
        "x_s_dot_ref",
        "y_s_dot_ref",
    ])

    if controller is not None and hasattr(controller, "thruster_forces"):
        if controller_history is None:
            controller_history = controller_output_history(controller, t, state)
        tau_required, tau_delivered, thruster_forces = controller_history
        pwm_us = thruster_forces_to_pwm_us(thruster_forces)
        data_columns.extend([
            tau_required[:, 0],
            tau_required[:, 2],
            tau_delivered[:, 0],
            tau_delivered[:, 2],
            thruster_forces,
            pwm_us,
        ])
        header_columns.extend([
            "tau_u_cmd",
            "tau_r_cmd",
            "tau_u_real",
            "tau_r_real",
            "F1",
            "F2",
            "F3",
            "F4",
            "PWM1",
            "PWM2",
            "PWM3",
            "PWM4",
        ])

    if cbf_history_data is not None:
        obstacle_count = cbf_history_data["obstacle_count"]
        data_columns.extend([
            cbf_history_data["h_min"],
            cbf_history_data["active"].astype(float),
            cbf_history_data["z_cbf"][:, 0],
            cbf_history_data["z_cbf"][:, 1],
            cbf_history_data["delta_z"][:, 0],
            cbf_history_data["delta_z"][:, 1],
        ])
        header_columns.extend([
            "cbf_h_min",
            "cbf_active",
            "cbf_Tu",
            "cbf_Tr",
            "cbf_delta_Tu",
            "cbf_delta_Tr",
        ])

        for obstacle_index in range(obstacle_count):
            data_columns.append(cbf_history_data["h"][:, obstacle_index])
            header_columns.append(f"cbf_h_obs{obstacle_index + 1}")

    data = np.column_stack(data_columns)
    header = ",".join(header_columns)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def main():
    t_final = FIGURE_EIGHT_PERIOD * FIGURE_EIGHT_LAPS
    nominal_controller = smc_figure_eight(
        Tu_limits=TU_LIMITS,
        Tr_limits=TR_LIMITS,
        lookahead_L=LOOKAHEAD_L,
    )
    state0 = figure_eight_initial_state()
    disturbance_fun = DISTURBANCE_OPTIONS[DISTURBANCE_MODE]
    obstacles = configured_obstacles(enabled=ENABLE_OBSTACLES)
    reference_governor_fun = virtual_time_governor(
        obstacles,
        enabled=ENABLE_REFERENCE_GOVERNOR,
        lookahead_L=LOOKAHEAD_L,
        margin=CBF_SAFETY_MARGIN,
        gamma_min=REFERENCE_GAMMA_MIN,
        h_slow=REFERENCE_GOVERNOR_H_SLOW,
        h_stop=REFERENCE_GOVERNOR_H_STOP,
        tau_min=REFERENCE_GAMMA_FILTER_TAU_MIN,
        tau_max=REFERENCE_GAMMA_FILTER_TAU_MAX,
    )
    controller = cbf_filtered_controller(
        nominal_controller,
        obstacles,
        enabled=ENABLE_CBF_AVOIDANCE,
        lookahead_L=LOOKAHEAD_L,
        alpha_1=CBF_RELATIVE_DEGREE_ALPHA_1,
        alpha_2=CBF_RELATIVE_DEGREE_ALPHA_2,
        margin=CBF_SAFETY_MARGIN,
        barrier_type=CBF_SELECTED_BARRIER,
        ellipse_major_scale=CBF_SELECTED_ELLIPSE_MAJOR_SCALE,
        ellipse_minor_scale=CBF_SELECTED_ELLIPSE_MINOR_SCALE,
        ellipse_angle_deg=CBF_SELECTED_ELLIPSE_ANGLE_DEG,
    )

    if ENABLE_REFERENCE_GOVERNOR:
        state0 = np.concatenate([state0, [0.0, 1.0]])

    t, state = simulate(
        state0=state0,
        tau_fun=controller,
        disturbance_fun=disturbance_fun,
        t_final=t_final,
        dt=DT,
        reference_governor_fun=reference_governor_fun,
    )

    governor_history_data = governor_history(
        state,
        obstacles,
        enabled=ENABLE_REFERENCE_GOVERNOR,
        lookahead_L=LOOKAHEAD_L,
        margin=CBF_SAFETY_MARGIN,
        gamma_min=REFERENCE_GAMMA_MIN,
        h_slow=REFERENCE_GOVERNOR_H_SLOW,
        h_stop=REFERENCE_GOVERNOR_H_STOP,
        tau_min=REFERENCE_GAMMA_FILTER_TAU_MIN,
        tau_max=REFERENCE_GAMMA_FILTER_TAU_MAX,
    )
    controller_history = controller_output_history(
        controller,
        t,
        state,
        gamma_history=governor_history_data["gamma"],
    )
    cbf_history_data = None
    if ENABLE_CBF_DIAGNOSTICS:
        cbf_history_data = cbf_history(
            t,
            state,
            controller_history[0],
            obstacles,
            lookahead_L=LOOKAHEAD_L,
            alpha_1=CBF_RELATIVE_DEGREE_ALPHA_1,
            alpha_2=CBF_RELATIVE_DEGREE_ALPHA_2,
            margin=CBF_SAFETY_MARGIN,
            barrier_type=CBF_SELECTED_BARRIER,
            ellipse_major_scale=CBF_SELECTED_ELLIPSE_MAJOR_SCALE,
            ellipse_minor_scale=CBF_SELECTED_ELLIPSE_MINOR_SCALE,
            ellipse_angle_deg=CBF_SELECTED_ELLIPSE_ANGLE_DEG,
        )

    save_csv(
        CSV_PATH,
        t,
        state,
        controller=controller,
        controller_history=controller_history,
        cbf_history_data=cbf_history_data,
        governor_history_data=governor_history_data,
        lookahead_L=LOOKAHEAD_L,
    )
    print(f"Simulacion guardada en: {CSV_PATH}")
    plot_results(
        t,
        state,
        controller=controller,
        controller_history=controller_history,
        Tu_limits=TU_LIMITS,
        Tr_limits=TR_LIMITS,
        disturbance_fun=disturbance_fun,
        obstacles=obstacles,
        cbf_history_data=cbf_history_data,
        governor_history_data=governor_history_data,
        lookahead_L=LOOKAHEAD_L,
        save_dir=FIGURES_DIR,
        show=SHOW_PLOTS,
    )
    print(f"Graficos guardados en: {FIGURES_DIR}")
    for figure_path in sorted(FIGURES_DIR.glob("*.png")):
        print(f" - {figure_path}")

    if SAVE_ANIMATION:
        animation_path = save_simulation_animation(
            t,
            state,
            output_path=ANIMATION_PATH,
            fps=ANIMATION_FPS,
            duration=ANIMATION_DURATION,
            obstacles=obstacles,
            cbf_history_data=cbf_history_data,
            cbf_margin=CBF_SAFETY_MARGIN,
            governor_history_data=governor_history_data,
            lookahead_L=LOOKAHEAD_L,
        )
        print(f"Animacion guardada en: {animation_path}")


if __name__ == "__main__":
    main()
