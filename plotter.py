import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib import animation

def plot_trajectories_pos(x_history, x_des):
    """Plot position trajectories (actual vs reference)."""

    N = x_history.shape[1]

    sns.set_theme()
    plt.figure()
    plt.plot(x_history[0, :], label="x (actual)")
    plt.plot(x_history[1, :], label="y (actual)")
    plt.plot(x_history[2, :], label="z (actual)")
    plt.plot(x_des[0, :N], "b--", label="x (reference)")
    plt.plot(x_des[1, :N], "r--", label="y (reference)")
    plt.plot(x_des[2, :N], "g--", label="z (reference)")

    plt.xlabel("Time step")
    plt.legend()
    plt.title("Position Trajectory")
    plt.savefig("images/state_trajectories_pos.png")
    plt.show()


def plot_trajectories_vel(x_history, x_des):
    """Plot velocity trajectories (actual vs reference)."""

    N = x_history.shape[1]

    sns.set_theme()
    plt.figure()
    plt.plot(x_history[3, :], label="vx (actual)")
    plt.plot(x_history[4, :], label="vy (actual)")
    plt.plot(x_history[5, :], label="vz (actual)")
    plt.plot(x_des[3, :N], "b--", label="vx (reference)")
    plt.plot(x_des[4, :N], "r--", label="vy (reference)")
    plt.plot(x_des[5, :N], "g--", label="vz (reference)")

    plt.xlabel("Time step")
    plt.legend()
    plt.title("Velocity Trajectory")
    plt.savefig("images/state_trajectories_vel.png")
    plt.show()


def export_trajectory_gif(x_history, x_des, save_path="images/trajectory.gif", fps=20):
    """Export a 3D trajectory GIF (reference + actual)."""

    sns.set_theme()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Take the common length of both trajectories
    N = min(x_history.shape[1], x_des.shape[1])

    # Extract position components
    x_act, y_act, z_act = x_history[0, :N], x_history[1, :N], x_history[2, :N]
    x_ref, y_ref, z_ref = x_des[0, :N], x_des[1, :N], x_des[2, :N]

    # Create 3D plot
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Draw the reference trajectory (dashed)
    ax.plot(x_ref, y_ref, z_ref, "k--", linewidth=1.5, label="Reference trajectory")

    # Actual trajectory (animated frame-by-frame)
    act_line, = ax.plot([], [], [], "b-", linewidth=2.0, label="Actual trajectory")
    act_dot, = ax.plot([], [], [], "ro", markersize=5, label="Current UAV position")

    # Set coordinate ranges
    xs = np.concatenate([x_act, x_ref])
    ys = np.concatenate([y_act, y_ref])
    zs = np.concatenate([z_act, z_ref])
    pad = 0.5
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_zlim(zs.min() - pad, zs.max() + pad)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Quadrotor Trajectory Tracking")
    ax.legend(loc="upper right")

    def init():
        act_line.set_data([], [])
        act_line.set_3d_properties([])
        act_dot.set_data([], [])
        act_dot.set_3d_properties([])
        return act_line, act_dot

    def update(frame):
        # Draw the actual trajectory up to frame
        act_line.set_data(x_act[:frame + 1], y_act[:frame + 1])
        act_line.set_3d_properties(z_act[:frame + 1])

        # Current point
        act_dot.set_data([x_act[frame]], [y_act[frame]])
        act_dot.set_3d_properties([z_act[frame]])
        return act_line, act_dot

    ani = animation.FuncAnimation(
        fig, update, frames=N, init_func=init, interval=1000 / fps, blit=False
    )

    # Dependency on Pillow
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)


