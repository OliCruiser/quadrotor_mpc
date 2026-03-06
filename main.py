"""
Quadrotor trajectory tracking control with Model Predictive Control (MPC).

Author: Elena Oikonomou
Date: Spring 2023
"""
import utils
import mpc
import plotter
import config


def main():
    # 生成参考轨迹（给 MPC 跟踪）和线性化轨迹（给模型线性化用）
    X_ref, U_ref, X_lin_ref, U_lin_ref = utils.get_ref_trajectory()
    # 将非线性模型转换为线性模型(泰勒展开)，得到每一步的 A、B 矩阵
    A, B = utils.get_linearized_dynamics_matrices(X_ref, U_ref, config.dt)  # nxnxN, nxmxN
    # 算！
    x_history, u_history = mpc.simulation_MPC(A, B, X_ref, U_ref, X_lin_ref, U_lin_ref)

    # Plot trajectories (position & linear velocities)
    plotter.plot_trajectories_pos(x_history, X_ref)
    plotter.plot_trajectories_vel(x_history, X_ref)
    plotter.export_trajectory_gif(x_history, X_ref)

    # Save data for visualization
    utils.save_to_file(x_history, X_ref)

if __name__ == "__main__":
    main()
    print("over!")
