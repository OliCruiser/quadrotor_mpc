import jax
# 1. 导入 JAX 配置并重命名为 jax_config
from jax import config as jax_config 
# 2. 使用新名字进行更新
jax_config.update("jax_enable_x64", True)   

import numpy as np
import jax.numpy as jnp
from jax import jit, jacfwd
import pandas as pd

# 3. 导入项目本地的配置文件 config.py
import config

@jit
def skew(v):
    """Computes the skew-symmetric matrix that converts a vector cross product to matrix multiplication.

    axb = skew(a)*b
    Inputs:
      - v(np.ndarray): A 3D vector [3x1]
    Returns:
      - (jnp.ndarray): The skew-symmetric matrix [3x3]
    """
    v = v.reshape(3,)  # To ensure both [3,1] and (3,) vectors work
    return jnp.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])


@jit
def dcm_from_mrp(p):
    """Converts Modified Rodrigues Parameters (MRP) to Direction Cosine Matrix (DCM).

    Inputs:
      - p(np.ndarray):  The Modified Rodrigues Parameters (MRP) [3x1]
    Returns:
      - R(jnp.ndarray): The Direction Cosine Matrix (DCM) [3x3]
    """
    p1 = p[0][0]
    p2 = p[1][0]
    p3 = p[2][0]

    den = (p1**2 + p2**2 + p3**2 + 1)**2
    a = (4*p1**2 + 4*p2**2 + 4*p3**2 - 4)
    R = jnp.array([[-((8*p2**2+8*p3**2)/den-1)*den, 8*p1*p2 + p3*a, 8*p1*p3 - p2*a],
                   [8*p1*p2 - p3*a, -((8*p1**2 + 8*p3**2)/den - 1)*den, 8*p2*p3 + p1*a],
                   [8*p1*p3 + p2*a, 8*p2*p3 - p1*a, -((8*p1**2 + 8*p2**2)/den - 1)*den]])/den
    return R


def quadrotor_dynamics(x, u):
    """Computes the continuous-time dynamics for a quadrotor ẋ=f(x,u).

    State is x = [r, v, p, omega], where:
    - r ∈R^3 is the position in world frame N
    - v ∈R^3 is the linear velocity in world frame N
    - p ∈R^3 is the attitude from B->N (MRP)
    - omega ∈R^3 is the angular velocity in body frame B
    Inputs:
      - x(np.ndarray): The system state   [12x1]
      - u(np.ndarray): The control inputs [4x1]
    Returns:
      - x_d(np.ndarray): The time derivative of the state [12x1]
    """
    # Quadrotor parameters
    mass = 0.5
    L = 0.1750
    J = jnp.diag(jnp.array([0.0023, 0.0023, 0.004]))
    kf = 1.0
    km = 0.0245
    gravity = jnp.array([0,0,-9.81]).reshape(3, 1)

    # State variables
    r = x[0:3].reshape(3, 1)
    v = x[3:6].reshape(3, 1)
    p = x[6:9].reshape(3, 1)
    omega = x[9:12].reshape(3, 1)

    Q = dcm_from_mrp(p)

    w1 = u[0]
    w2 = u[1]
    w3 = u[2]
    w4 = u[3]

    F1 = max(0, kf*w1)
    F2 = max(0, kf*w2)
    F3 = max(0, kf*w3)
    F4 = max(0, kf*w4)
    F = jnp.array([0.0, 0.0, F1+F2+F3+F4]).reshape(3, 1)  # Total rotor force in body frame

    M1 = km*w1
    M2 = km*w2
    M3 = km*w3
    M4 = km*w4
    tau = jnp.array([L*(F2-F4), L*(F3-F1), (M1-M2+M3-M4)]).reshape(3, 1)  # Total rotor torque in body frame

    f = mass*gravity + Q@F  # Forces in world frame

    # Dynamics
    r_d = v
    v_d = f/mass
    p_d = ((1+jnp.linalg.norm(p)**2)/4)*(jnp.eye(3) + 2*(skew(p)@skew(p)+skew(p))/(1+jnp.linalg.norm(p)**2))@omega
    cross_pr = jnp.cross(omega.reshape(3,), (J@omega).reshape(3,)).reshape(3,1)
    omega_d, _, _, _ = jnp.linalg.lstsq(J, tau - cross_pr, rcond=None)

    return jnp.vstack((r_d, v_d, p_d, omega_d)).reshape(12,)  # x_dot


def quadrotor_rk4(x, u, Ts):
    """Discrete-time dynamics: Integration with RK4 method."""
    f = quadrotor_dynamics
    k1 = Ts*f(x, u)
    k2 = Ts*f(x + k1/2, u)
    k3 = Ts*f(x + k2/2, u)
    k4 = Ts*f(x + k3, u)
    x_next = x + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    return x_next


def get_linearized_dynamics_matrices(X_ref, U_ref, Ts):
    """沿参考轨迹对离散动力学进行线性化，得到每一步的 A、B 矩阵。

    参数:
      - X_ref(np.ndarray): 状态参考轨迹，形状 [12, N]
      - U_ref(np.ndarray): 控制参考轨迹，形状 [4, N]（或至少第二维可按 i 索引）
      - Ts(float): 离散时间步长
    返回:
      - Ad(np.ndarray): 每一步的状态雅可比矩阵 A，形状 [12, 12, N]
      - Bd(np.ndarray): 每一步的输入雅可比矩阵 B，形状 [12, 4, N]
    """
    # 读取维度信息
    Nx = X_ref.shape[0]      # 状态维度（通常 12）
    Nu = U_ref.shape[0]      # 控制维度（通常 4）
    N = X_ref.shape[1]   # 时域长度 N

    # 预分配线性化结果容器
    Ad = np.zeros((Nx, Nx, N))
    Bd = np.zeros((Nx, Nu, N))

    # 对每个时刻 i 的 (x_i, u_i) 做一次线性化
    # jacfwd(..., 0): 对第 1 个参数 x 求导 => A_i = ∂f/∂x
    # jacfwd(..., 1): 对第 2 个参数 u 求导 => B_i = ∂f/∂u
    for i in range(N - 1):
        Ad[:, :, i] = jacfwd(quadrotor_rk4, 0)(X_ref[:, i], U_ref[:, i], Ts)  # [12x12]
        Bd[:, :, i] = jacfwd(quadrotor_rk4, 1)(X_ref[:, i], U_ref[:, i], Ts)  # [12x4]

    return Ad, Bd


# 表征轨迹
def get_ref_trajectory():
    """生成参考轨迹（给 MPC 跟踪）和线性化轨迹（给模型线性化用）。"""
    # 从配置文件读取参数
    x0 = config.x0      # 初始状态，长度 Nx
    Nx = config.Nx      # 状态维度（这里是 12）
    Nu = config.Nu      # 控制维度（这里是 4）
    N = config.N        # 轨迹总步数
    dt = config.dt      # 采样时间

    # ===== 1) 线性化参考轨迹(固定点线性化) =====
    # 状态线性化参考：每一列都用初始状态 x0（大小 Nx x N），接下来的预测，请全部以‘静止悬停’这个点作为基准进行泰勒展开（线性化）。
    X_lin_ref = jnp.tile(x0.reshape(Nx, 1), (1, N))

    # 控制线性化参考：使用“悬停推力”作为每个电机的基准输入（大小 Nu x (N-1)）
    # 0.5 是质量，9.81 是重力加速度，除以 4 表示四个电机平均分担
    U_lin_ref = jnp.tile((9.81 * 0.5 / 4) * jnp.ones(Nu).reshape(Nu, 1), (1, N - 1))

    # ===== 2) 跟踪参考轨迹 =====
    # 这里直接用同一份控制参考
    U_ref = U_lin_ref

    # 状态参考（Nx x N），后面逐列填充
    X_ref = np.zeros((Nx, N))

    # 生成一条空间曲线：
    # x = 5*cos(t), y = 5*cos(t)*sin(t), z = 1.2（恒定高度）
    # 其余状态：速度先置 0，姿态给极小值，角速度置 0
    i = 0
    for t in np.linspace(-np.pi / 2, 3 * np.pi / 2 + 4 * np.pi, N):
        pos = np.array([5 * np.cos(t), 5 * np.cos(t) * np.sin(t), 1.2])  # 位置 [x,y,z]
        vel = np.zeros(3)                                                 # 速度 [vx,vy,vz]（先占位）
        mrp = 1e-9 * np.ones(3)                                           # 姿态 MRP（避免严格为 0）
        omega = np.zeros(3)                                               # 角速度 [wx,wy,wz]
        X_ref[:, i] = np.hstack((pos, vel, mrp, omega))
        i += 1

    # 用前向差分计算参考速度：v_k = (r_{k+1} - r_k) / dt
    # 只计算到 N-2，因此最后一个时刻速度保持前面的默认值
    for i in range(N - 1):
        X_ref[3:6, i] = (X_ref[0:3, i + 1] - X_ref[0:3, i]) / dt
    # 此时姿态和角速度仍然是默认值0，保持不变
    return X_ref, U_ref, X_lin_ref, U_lin_ref

def save_to_file(x_history, X_ref):
    """Save state trajectory to csv file to visualize with Julia."""

    # Save trajectory
    filename = "X_quadrotor.csv"
    df = pd.DataFrame(x_history)   # convert array into dataframe
    df.to_csv(filename, index=False, header=False, float_format='%f')

    # Save reference trajectory
    filename = "X_ref.csv"
    df = pd.DataFrame(X_ref)  # convert array into dataframe
    df.to_csv(filename, index=False, header=False, float_format='%f')
