import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle

from cbf_filters import CBF_MARGIN
from controllers import (
    TRACKING_LOOKAHEAD,
    figure_eight_lookahead_reference,
    figure_eight_reference,
    lookahead_point_from_state,
)
from usv_model import THRUSTER_FWD_MAX, THRUSTER_REV_MAX


def boat_outline(x, y, psi, length=1.2, width=0.55):
    """
    Devuelve los puntos de un bote simple visto desde arriba.

    El eje x del bote apunta hacia la proa.
    """
    points_body = np.array([
        [length / 2, 0.0],
        [0.10 * length, width / 2],
        [-length / 2, width / 3],
        [-length / 2, -width / 3],
        [0.10 * length, -width / 2],
        [length / 2, 0.0],
    ])

    c = np.cos(psi)
    s = np.sin(psi)
    rotation = np.array([
        [c, -s],
        [s,  c],
    ])

    points_world = points_body @ rotation.T
    points_world[:, 0] += x
    points_world[:, 1] += y

    return points_world


def draw_boat(ax, x, y, psi, length=1.2, width=0.55, **kwargs):
    """Dibuja un bote simple en el eje indicado."""
    outline = boat_outline(x, y, psi, length=length, width=width)
    ax.plot(outline[:, 0], outline[:, 1], **kwargs)


def draw_obstacles(ax, obstacles):
    """Dibuja boyas circulares y el contorno ocupado por cada obstaculo."""
    for index, obstacle in enumerate(obstacles):
        boundary_label = "obstacle boundary" if index == 0 else "_nolegend_"
        center_label = "buoy center" if index == 0 else "_nolegend_"
        boundary = Circle(
            (obstacle.x, obstacle.y),
            obstacle.radius,
            edgecolor="magenta",
            facecolor="magenta",
            alpha=0.25,
            linewidth=2.0,
            label=boundary_label,
            zorder=2,
        )
        ax.add_patch(boundary)
        ax.scatter(
            obstacle.x,
            obstacle.y,
            color="magenta",
            edgecolors="black",
            linewidths=0.5,
            s=28,
            label=center_label,
            zorder=4,
        )


def draw_disturbance_arrow(ax, magnitude, direction_deg):
    """Dibuja una flecha con la direccion de la perturbacion externa."""
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    arrow_length = 0.18 * min(x_span, y_span)

    direction_rad = np.deg2rad(direction_deg)
    dx = arrow_length * np.cos(direction_rad)
    dy = arrow_length * np.sin(direction_rad)
    x0 = x_limits[0] + 0.12 * x_span
    y0 = y_limits[1] - 0.16 * y_span

    ax.annotate(
        "",
        xy=(x0 + dx, y0 + dy),
        xytext=(x0, y0),
        arrowprops={
            "arrowstyle": "->",
            "color": "tab:purple",
            "linewidth": 2.0,
        },
    )
    ax.text(
        x0,
        y0 - 0.06 * y_span,
        f"disturbance: {magnitude:.1f} @ {direction_deg:.0f} deg",
        color="tab:purple",
        fontsize=9,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "tab:purple",
            "alpha": 0.85,
        },
    )


def plot_disturbance(t, state, disturbance_fun):
    """Grafica la perturbacion aditiva d(t) = [dTu, dTv, dTr]."""
    disturbance = np.array([
        disturbance_fun(ti, state_i)
        for ti, state_i in zip(t, state)
    ])

    fig, ax = plt.subplots()
    ax.plot(t, disturbance[:, 0], label="dTu")
    ax.plot(t, disturbance[:, 1], label="dTv")
    ax.plot(t, disturbance[:, 2], label="dTr")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("disturbance")
    ax.set_title("Additive Disturbances")
    ax.legend()
    ax.grid(True)

    return fig


def plot_tracking_error(
    t,
    state,
    lookahead_L=TRACKING_LOOKAHEAD,
    governor_history_data=None,
):
    """Grafica errores de posicion y velocidad del punto adelantado."""
    reference_time = state[:, 6] if state.shape[1] >= 7 else t
    reference_rate = (
        governor_history_data["gamma"]
        if governor_history_data is not None
        else 1.0
    )
    p_s, p_s_dot = lookahead_point_from_state(state, lookahead_L)
    p_s_ref, p_s_dot_ref, _, _ = figure_eight_lookahead_reference(
        reference_time,
        lookahead_L,
        time_scale=reference_rate,
    )

    error = p_s_ref - p_s
    error_x = error[:, 0]
    error_y = error[:, 1]
    position_error_norm = np.hypot(error_x, error_y)

    velocity_error = p_s_dot_ref - p_s_dot
    velocity_error_x = velocity_error[:, 0]
    velocity_error_y = velocity_error[:, 1]
    velocity_error_norm = np.hypot(velocity_error_x, velocity_error_y)

    fig, axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(t, position_error_norm, label="||e_p||")
    axes[0].plot(t, error_x, "--", alpha=0.75, label="e_x")
    axes[0].plot(t, error_y, "--", alpha=0.75, label="e_y")
    axes[0].set_ylabel("position error [m]")
    axes[0].set_title("Lookahead Point Position Tracking Error")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, velocity_error_norm, label="||e_v||")
    axes[1].plot(t, velocity_error_x, "--", alpha=0.75, label="e_vx")
    axes[1].plot(t, velocity_error_y, "--", alpha=0.75, label="e_vy")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("velocity error [m/s]")
    axes[1].set_title("Lookahead Point Velocity Tracking Error")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    return fig


def plot_torque_saturation(
    t,
    state,
    controller,
    Tu_limits=None,
    Tr_limits=None,
    tau_required=None,
    tau_delivered=None,
):
    """Grafica las entradas pedidas y entregadas por saturacion."""
    if tau_required is None and hasattr(controller, "required_tau"):
        tau_required = np.array([
            controller.required_tau(ti, state_i)
            for ti, state_i in zip(t, state)
        ])
    elif tau_required is None:
        tau_required = np.array([
            controller(ti, state_i)
            for ti, state_i in zip(t, state)
        ])

    if tau_delivered is None and hasattr(controller, "delivered_tau"):
        tau_delivered = np.array([
            controller.delivered_tau(ti, state_i)
            for ti, state_i in zip(t, state)
        ])
    elif tau_delivered is None:
        tau_delivered = np.array([
            controller(ti, state_i)
            for ti, state_i in zip(t, state)
        ])

    fig, axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(t, tau_required[:, 0], label="required surge force", alpha=0.75)
    axes[0].plot(t, tau_delivered[:, 0], label="delivered surge force", linewidth=2.0)

    if Tu_limits is not None:
        axes[0].axhline(
            Tu_limits[0],
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="force limits",
        )
        axes[0].axhline(
            Tu_limits[1],
            color="black",
            linestyle=":",
            linewidth=1.2,
        )

    axes[0].set_ylabel("surge force [N]")
    axes[0].set_title("Required vs Delivered Surge Force")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, tau_required[:, 2], label="required yaw torque", alpha=0.75)
    axes[1].plot(t, tau_delivered[:, 2], label="delivered yaw torque", linewidth=2.0)

    if Tr_limits is not None:
        axes[1].axhline(
            Tr_limits[0],
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="torque limits",
        )
        axes[1].axhline(
            Tr_limits[1],
            color="black",
            linestyle=":",
            linewidth=1.2,
        )

    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("yaw torque [N m]")
    axes[1].set_title("Required vs Delivered Yaw Torque")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    return fig


def plot_thruster_forces(t, state, controller, forces=None):
    """Grafica la fuerza entregada por cada propulsor."""
    if forces is None and not hasattr(controller, "thruster_forces"):
        raise ValueError("The controller does not expose thruster_forces(t, state).")

    if forces is None:
        forces = np.array([
            controller.thruster_forces(ti, state_i)
            for ti, state_i in zip(t, state)
        ])

    fig, ax = plt.subplots()
    ax.plot(t, forces[:, 0], label="F1 left")
    ax.plot(t, forces[:, 1], label="F2 left")
    ax.plot(t, forces[:, 2], label="F3 right")
    ax.plot(t, forces[:, 3], label="F4 right")
    ax.axhline(
        THRUSTER_FWD_MAX,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="thruster limits",
    )
    ax.axhline(
        -THRUSTER_REV_MAX,
        color="black",
        linestyle=":",
        linewidth=1.2,
    )
    ax.set_xlabel("t [s]")
    ax.set_ylabel("thruster force [N]")
    ax.set_title("Individual Thruster Forces")
    ax.legend()
    ax.grid(True)

    return fig


def plot_cbf_barrier(t, cbf_history_data):
    """Grafica la evolucion de la funcion de barrera por obstaculo."""
    fig, axes = plt.subplots(2, 1, sharex=True)
    h = cbf_history_data["h"]
    h_min = cbf_history_data["h_min"]
    active = cbf_history_data["active"].astype(float)

    if h.shape[1] == 0:
        axes[0].plot(t, np.zeros_like(t), label="no obstacles")
    else:
        for obstacle_index in range(h.shape[1]):
            axes[0].plot(
                t,
                h[:, obstacle_index],
                label=f"h obstacle {obstacle_index + 1}",
            )
        axes[0].plot(t, h_min, "k--", linewidth=1.5, label="min h")

    axes[0].axhline(0.0, color="red", linestyle=":", linewidth=1.2, label="h = 0")
    axes[0].set_ylabel("barrier h [m^2]")
    axes[0].set_title("CBF Barrier Function at Lookahead Point")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, active, color="tab:red", label="hypothetical CBF active")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("active")
    axes[1].set_yticks([0.0, 1.0])
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    return fig


def plot_reference_governor(t, governor_history_data):
    """Grafica el tiempo virtual sigma y la velocidad gamma."""
    fig, axes = plt.subplots(2, 1, sharex=True)
    sigma = governor_history_data["sigma"]
    gamma = governor_history_data["gamma"]
    gamma_target = governor_history_data.get("gamma_target")
    tau_gamma = governor_history_data.get("tau_gamma")

    if sigma is not None:
        axes[0].plot(t, sigma, label="sigma")
        axes[0].plot(t, t, "k--", alpha=0.35, label="real time")
    else:
        axes[0].plot(t, t, label="sigma = t")

    axes[0].set_ylabel("reference time [s]")
    axes[0].set_title("Virtual-Time Reference Governor")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, gamma, color="tab:orange", label="gamma")
    if gamma_target is not None:
        axes[1].plot(
            t,
            gamma_target,
            "k--",
            alpha=0.45,
            label="gamma target",
        )
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("sigma_dot")
    axes[1].legend()
    axes[1].grid(True)

    if tau_gamma is not None:
        ax_tau = axes[1].twinx()
        ax_tau.plot(
            t,
            tau_gamma,
            color="tab:blue",
            alpha=0.45,
            label="filter tau",
        )
        ax_tau.set_ylabel("filter tau [s]")
        ax_tau.legend(loc="lower right")

    fig.tight_layout()

    return fig


def save_simulation_animation(
    t,
    state,
    output_path,
    fps=25,
    duration=20.0,
    show_reference=True,
    obstacles=None,
    cbf_history_data=None,
    cbf_margin=CBF_MARGIN,
    governor_history_data=None,
    lookahead_L=TRACKING_LOOKAHEAD,
):
    """Guarda una animacion del movimiento del USV y su referencia."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = state[:, 0]
    y = state[:, 1]
    psi = state[:, 2]
    reference_time = state[:, 6] if state.shape[1] >= 7 else t
    reference_rate = (
        governor_history_data["gamma"]
        if governor_history_data is not None
        else 1.0
    )
    x_ref, y_ref, x_dot_ref, y_dot_ref, _, _ = figure_eight_reference(
        reference_time
    )
    x_dot_ref = reference_rate * x_dot_ref
    y_dot_ref = reference_rate * y_dot_ref
    psi_ref = np.arctan2(y_dot_ref, x_dot_ref)
    p_s, _ = lookahead_point_from_state(state, lookahead_L)
    p_s_ref, _, _, _ = figure_eight_lookahead_reference(
        reference_time,
        lookahead_L,
        time_scale=reference_rate,
    )
    x_path_ref, y_path_ref, _, _, _, _ = figure_eight_reference(t)
    p_s_path_ref, _, _, _ = figure_eight_lookahead_reference(t, lookahead_L)

    frame_count = max(2, int(fps * duration))
    frame_indices = np.linspace(0, len(t) - 1, frame_count, dtype=int)

    obstacles = () if obstacles is None else obstacles
    x_all = np.concatenate([x, x_path_ref, p_s[:, 0], p_s_path_ref[:, 0]])
    y_all = np.concatenate([y, y_path_ref, p_s[:, 1], p_s_path_ref[:, 1]])
    if obstacles:
        x_obstacle_bounds = np.array([
            coordinate
            for obstacle in obstacles
            for coordinate in (
                obstacle.x - obstacle.radius - cbf_margin,
                obstacle.x + obstacle.radius + cbf_margin,
            )
        ])
        y_obstacle_bounds = np.array([
            coordinate
            for obstacle in obstacles
            for coordinate in (
                obstacle.y - obstacle.radius - cbf_margin,
                obstacle.y + obstacle.radius + cbf_margin,
            )
        ])
        x_all = np.concatenate([x_all, x_obstacle_bounds])
        y_all = np.concatenate([y_all, y_obstacle_bounds])
    span = max(np.ptp(x_all), np.ptp(y_all), 1.0)
    margin = 0.12 * span
    boat_length = 0.06 * span
    boat_width = 0.45 * boat_length

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.set_xlim(np.min(x_all) - margin, np.max(x_all) + margin)
    ax.set_ylim(np.min(y_all) - margin, np.max(y_all) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("USV Motion Simulation")
    ax.grid(True)

    if show_reference:
        ax.plot(
            x_path_ref,
            y_path_ref,
            "--",
            color="0.45",
            alpha=0.45,
            label="figure-eight reference",
        )
        ax.plot(
            p_s_path_ref[:, 0],
            p_s_path_ref[:, 1],
            ":",
            color="0.10",
            alpha=0.55,
            label="lookahead reference",
        )

    cbf_safety_patches = []
    if obstacles:
        for index, obstacle in enumerate(obstacles):
            label = "CBF safety zone" if index == 0 else "_nolegend_"
            safety_zone = Circle(
                (obstacle.x, obstacle.y),
                obstacle.radius + cbf_margin,
                edgecolor=(1.0, 0.62, 0.0, 0.35),
                facecolor=(1.0, 0.85, 0.0, 0.10),
                linewidth=1.4,
                label=label,
                zorder=1,
            )
            ax.add_patch(safety_zone)
            cbf_safety_patches.append(safety_zone)

        draw_obstacles(ax, obstacles)

    actual_path, = ax.plot([], [], color="tab:blue", linewidth=2.0, label="USV path")
    lookahead_path, = ax.plot(
        [], [], color="black", linewidth=1.6, label="lookahead path"
    )
    reference_progress, = ax.plot(
        [], [], color="0.35", alpha=0.35, linewidth=2.0, label="reference progress"
    )
    actual_boat, = ax.plot([], [], color="tab:blue", linewidth=2.0, label="USV")
    reference_boat, = ax.plot(
        [], [], color="0.35", alpha=0.45, linewidth=2.0, label="reference USV"
    )
    lookahead_point, = ax.plot(
        [], [], "o", color="black", markersize=5.5, label="lookahead point"
    )
    cbf_text = ax.text(
        0.02,
        0.88,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="0.15",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.9,
        },
    )
    time_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.9,
        },
    )
    ax.legend(loc="upper right")

    def update(frame_index):
        index = frame_indices[frame_index]
        cbf_active = False
        critical_obstacle = None

        if (
            cbf_history_data is not None
            and cbf_history_data["h"].shape[1] > 0
        ):
            cbf_active = bool(cbf_history_data["active"][index])
            critical_obstacle = int(np.argmin(cbf_history_data["h"][index]))

        actual_path.set_data(x[:index + 1], y[:index + 1])
        lookahead_path.set_data(p_s[:index + 1, 0], p_s[:index + 1, 1])
        reference_progress.set_data(x_ref[:index + 1], y_ref[:index + 1])

        actual_outline = boat_outline(
            x[index],
            y[index],
            psi[index],
            length=boat_length,
            width=boat_width,
        )
        reference_outline = boat_outline(
            x_ref[index],
            y_ref[index],
            psi_ref[index],
            length=boat_length,
            width=boat_width,
        )
        actual_boat.set_data(actual_outline[:, 0], actual_outline[:, 1])
        reference_boat.set_data(reference_outline[:, 0], reference_outline[:, 1])
        lookahead_point.set_data([p_s[index, 0]], [p_s[index, 1]])
        lookahead_point.set_color("tab:orange" if cbf_active else "black")
        lookahead_point.set_markersize(7.0 if cbf_active else 5.5)

        for obstacle_index, safety_zone in enumerate(cbf_safety_patches):
            is_critical = cbf_active and obstacle_index == critical_obstacle
            if is_critical:
                safety_zone.set_facecolor((1.0, 0.82, 0.0, 0.35))
                safety_zone.set_edgecolor((1.0, 0.45, 0.0, 0.95))
                safety_zone.set_linewidth(2.6)
            else:
                safety_zone.set_facecolor((1.0, 0.85, 0.0, 0.10))
                safety_zone.set_edgecolor((1.0, 0.62, 0.0, 0.35))
                safety_zone.set_linewidth(1.4)

        if cbf_active and critical_obstacle is not None:
            obstacle = obstacles[critical_obstacle]
            obstacle_name = getattr(
                obstacle,
                "name",
                f"obstacle {critical_obstacle + 1}",
            )
            cbf_text.set_text(f"CBF active: {obstacle_name}")
            cbf_text.set_color("tab:orange")
        elif cbf_history_data is not None:
            cbf_text.set_text("CBF inactive")
            cbf_text.set_color("0.25")
        else:
            cbf_text.set_text("")

        if state.shape[1] >= 7:
            time_text.set_text(
                f"t = {t[index]:.1f} s, sigma = {reference_time[index]:.1f} s"
            )
        else:
            time_text.set_text(f"t = {t[index]:.1f} s")

        return (
            *cbf_safety_patches,
            actual_path,
            lookahead_path,
            reference_progress,
            actual_boat,
            reference_boat,
            lookahead_point,
            cbf_text,
            time_text,
        )

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / fps,
        blit=True,
    )

    available_writers = animation.writers.list()
    if output_path.suffix.lower() == ".mp4" and "ffmpeg" in available_writers:
        writer = animation.FFMpegWriter(fps=fps)
    elif output_path.suffix.lower() == ".gif" and "pillow" in available_writers:
        writer = animation.PillowWriter(fps=fps)
    elif "pillow" in available_writers:
        output_path = output_path.with_suffix(".gif")
        writer = animation.PillowWriter(fps=fps)
    else:
        raise RuntimeError("No animation writer available. Install ffmpeg or pillow.")

    anim.save(output_path, writer=writer)
    plt.close(fig)

    return output_path


def plot_results(
    t,
    state,
    controller=None,
    controller_history=None,
    Tu_limits=None,
    Tr_limits=None,
    show_reference=True,
    disturbance_info=None,
    disturbance_fun=None,
    obstacles=None,
    cbf_history_data=None,
    governor_history_data=None,
    lookahead_L=TRACKING_LOOKAHEAD,
    save_dir=None,
    show=False,
):
    x = state[:, 0]
    y = state[:, 1]
    psi = state[:, 2]
    u = state[:, 3]
    v = state[:, 4]
    r = state[:, 5]
    p_s, _ = lookahead_point_from_state(state, lookahead_L)
    reference_time = state[:, 6] if state.shape[1] >= 7 else t
    reference_rate = (
        governor_history_data["gamma"]
        if governor_history_data is not None
        else 1.0
    )

    fig, ax = plt.subplots()
    ax.plot(x, y, label="USV path")
    ax.plot(
        p_s[:, 0],
        p_s[:, 1],
        color="black",
        linewidth=1.8,
        label="lookahead point path",
    )

    if show_reference:
        x_ref, y_ref, _, _, _, _ = figure_eight_reference(t)
        x_ref_progress, y_ref_progress, _, _, _, _ = figure_eight_reference(
            reference_time
        )
        p_s_ref, _, _, _ = figure_eight_lookahead_reference(t, lookahead_L)
        p_s_ref_progress, _, _, _ = figure_eight_lookahead_reference(
            reference_time,
            lookahead_L,
            time_scale=reference_rate,
        )
        ax.plot(
            x_ref, y_ref, "--",
            color="black",
            alpha=0.45,
            label="figure-eight reference",
        )
        ax.plot(
            x_ref_progress,
            y_ref_progress,
            color="0.45",
            alpha=0.45,
            label="governed reference progress",
        )
        ax.plot(
            p_s_ref[:, 0],
            p_s_ref[:, 1],
            ":",
            color="0.10",
            alpha=0.65,
            label="lookahead reference",
        )
        ax.plot(
            p_s_ref_progress[:, 0],
            p_s_ref_progress[:, 1],
            color="tab:orange",
            alpha=0.55,
            label="governed lookahead reference",
        )

    if obstacles:
        draw_obstacles(ax, obstacles)

    ax.scatter(x[0], y[0], color="green", s=45, label="start", zorder=3)
    ax.scatter(x[-1], y[-1], color="red", s=45, label="finish", zorder=3)
    ax.scatter(
        p_s[-1, 0],
        p_s[-1, 1],
        color="black",
        s=55,
        label="lookahead point",
        zorder=5,
    )

    boat_length = max(np.ptp(x), np.ptp(y), 1.0) * 0.08
    boat_width = boat_length * 0.45
    sample_count = min(14, len(x))
    sample_indices = np.linspace(0, len(x) - 1, sample_count, dtype=int)

    for index in sample_indices[:-1]:
        draw_boat(
            ax, x[index], y[index], psi[index],
            length=boat_length,
            width=boat_width,
            color="tab:blue",
            alpha=0.35,
            linewidth=1.0,
        )

    draw_boat(
        ax, x[-1], y[-1], psi[-1],
        length=boat_length,
        width=boat_width,
        color="red",
        linewidth=2.0,
        label="USV",
    )

    ax.axis("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("USV Trajectory with Heading")
    ax.grid(True)

    if disturbance_info is not None:
        draw_disturbance_arrow(
            ax,
            magnitude=disturbance_info["magnitude"],
            direction_deg=disturbance_info["direction_deg"],
        )

    ax.legend()

    trajectory_fig = fig

    orientation_fig, ax_orientation = plt.subplots()
    ax_orientation.plot(t, np.rad2deg(psi))
    ax_orientation.set_xlabel("t [s]")
    ax_orientation.set_ylabel(r"$\psi$ [deg]")
    ax_orientation.set_title(r"Heading Angle $\psi$")
    ax_orientation.grid(True)

    velocities_fig, ax_velocities = plt.subplots()
    ax_velocities.plot(t, u, label="u")
    ax_velocities.plot(t, v, label="v")
    ax_velocities.plot(t, r, label="r")
    ax_velocities.set_xlabel("t [s]")
    ax_velocities.set_ylabel("body-frame velocities")
    ax_velocities.set_title("Body-Frame Velocities")
    ax_velocities.legend()
    ax_velocities.grid(True)

    figures = {
        "trajectory": trajectory_fig,
        "tracking_error": plot_tracking_error(
            t,
            state,
            lookahead_L,
            governor_history_data=governor_history_data,
        ),
        "orientation": orientation_fig,
        "velocities": velocities_fig,
    }

    if governor_history_data is not None:
        figures["reference_governor"] = plot_reference_governor(
            t,
            governor_history_data,
        )

    if disturbance_fun is not None:
        figures["disturbances"] = plot_disturbance(t, state, disturbance_fun)

    if controller is not None:
        tau_required = None
        tau_delivered = None
        thruster_forces = None
        if controller_history is not None:
            tau_required, tau_delivered, thruster_forces = controller_history

        figures["torque_saturation"] = plot_torque_saturation(
            t,
            state,
            controller,
            Tu_limits=Tu_limits,
            Tr_limits=Tr_limits,
            tau_required=tau_required,
            tau_delivered=tau_delivered,
        )

        if hasattr(controller, "thruster_forces"):
            figures["thruster_forces"] = plot_thruster_forces(
                t,
                state,
                controller,
                forces=thruster_forces,
            )

    if cbf_history_data is not None:
        figures["cbf_barrier"] = plot_cbf_barrier(t, cbf_history_data)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(save_dir / f"{name}.png", dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close("all")

    return figures
