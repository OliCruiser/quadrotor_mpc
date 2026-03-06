import numpy as np
import cvxpy as cp

import utils
import config


def convex_mpc_quadrotor(A, B, Q, R, X_ref, U_ref, X_lin, U_lin, x0, u_min, u_max, N_mpc, dt):
    """将四旋翼 OCP 写成凸优化并求解（预测时域长度为 N_mpc）。

    参数:
      - A(np.ndarray): 每个时刻的离散状态矩阵，形状 [n, n, N_mpc]
      - B(np.ndarray): 每个时刻的离散输入矩阵，形状 [n, m, N_mpc]
      - Q(np.ndarray): 状态误差权重矩阵，形状 [n, n]
            - R(np.ndarray): 输入误差权重矩阵，形状 [m, m]
            - X_ref(np.ndarray): 参考状态轨迹，形状 [n, N_mpc]
      - U_ref(np.ndarray): 参考输入轨迹，形状 [m, N_mpc-1]
      - X_lin(np.ndarray): 线性化状态轨迹，形状 [n, N_mpc]
      - U_lin(np.ndarray): 线性化输入轨迹，形状 [m, N_mpc-1]
      - x0(np.ndarray): 当前时刻真实状态，形状 [n,]
      - u_min(np.ndarray): 输入下界，形状 [m,]
      - u_max(np.ndarray): 输入上界，形状 [m,]
      - N_mpc(int): 预测时域长度
      - dt(float): 离散步长
    返回:
      - np.ndarray: 当前时刻要施加的最优控制（只取第 1 步），形状 [m,]
    """
    Nx, Nu = B.shape[0], B.shape[1]  # 状态维度/控制维度

    # 决策变量：整段预测时域内的状态与控制
    X = cp.Variable((Nx, N_mpc))
    U = cp.Variable((Nu, N_mpc - 1))

    # 目标函数：状态跟踪误差 + 控制跟踪误差（二次型）
    objective = 0
    for i in range(N_mpc - 1):
        objective += 0.5 * cp.quad_form(X[:, i] - X_ref[:, i], Q)
        objective += 0.5 * cp.quad_form(U[:, i] - U_ref[:, i], R)

    # 终端状态代价
    objective += 0.5 * cp.quad_form(X[:, N_mpc - 1] - X_ref[:, N_mpc - 1], Q)

    # 约束：初值约束 + 输入上下界 + 线性化动力学约束
    constraints = [X[:, 0] == x0]
    for i in range(N_mpc - 1):
        constraints += [u_min <= U[:, i]]
        constraints += [U[:, i] <= u_max]
        constraints += [
            X[:, i + 1] == utils.quadrotor_rk4(X_lin[:, i], U_lin[:, i], dt)
            + A[:, :, i] @ (X[:, i] - X_lin[:, i])
            + B[:, :, i] @ (U[:, i] - U_lin[:, i])
        ]

    # 求解优化问题
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(verbose=True)

    # MPC 只执行第一步控制（滚动时域）
    return U.value[:, 0]


def simulation_MPC(A, B, X_ref, U_ref, X_lin_ref, U_lin_ref):
    """使用 MPC 控制器做闭环仿真。"""
    # 读取配置参数
    x0 = config.x0
    Q = config.Q
    R = config.R
    N_mpc = config.N_mpc
    N_sim = config.N_sim
    u_min = config.u_min
    u_max = config.u_max
    dt = config.dt
    Nx = config.Nx
    Nu = config.Nu

    # 结果缓存：
    # X_sim 保存状态轨迹 [Nx, N_sim]
    # U_sim 保存控制轨迹 [Nu, N_sim-1]
    X_sim = np.zeros((Nx, N_sim))
    X_sim[:, 0] = x0
    U_sim = np.zeros((Nu, N_sim - 1))

    # 滚动优化：每个仿真步都重新解一次 N_mpc 时域的优化问题
    for i in range(N_sim - 1):
        # 当前时刻对应的参考窗口
        X_ref_tilde = X_ref[:, i:(i + N_mpc)]
        U_ref_tilde = U_ref[:, i:(i + N_mpc - 1)]

        # 当前时刻对应的线性化基准窗口
        X_lin = X_lin_ref[:, i:(i + N_mpc)]
        U_lin = U_lin_ref[:, i:(i + N_mpc - 1)]

        # 当前窗口的离散线性模型
        Ad = A[:, :, i:(i + N_mpc)]
        Bd = B[:, :, i:(i + N_mpc)]

        # 求当前步最优控制
        U_sim[:, i] = convex_mpc_quadrotor(
            Ad, Bd, Q, R,
            X_ref_tilde, U_ref_tilde,
            X_lin, U_lin,
            X_sim[:, i],
            u_min, u_max,
            N_mpc, dt
        )

        # 用非线性模型推进一步，得到下一时刻真实状态
        X_sim[:, i + 1] = utils.quadrotor_rk4(X_sim[:, i], U_sim[:, i], dt)

    return X_sim, U_sim
