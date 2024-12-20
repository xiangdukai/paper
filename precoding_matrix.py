import numpy as np
import math
from array_response import array_response_Sa


# 参数定义
f = 10e9  # 系统频率 10GHz
lambda_ = 3e8 / f  # 天线波长
d = lambda_ / 2  # 用于虚拟信道表示的天线间距

# 生成 phi, theta 的码本
def generate_codebook_phi_theta(num_phi=8, num_theta=8):
    # phi_values = np.linspace(-np.pi, np.pi, num_phi)
    # theta_values = np.linspace(-np.pi / 2, np.pi / 2, num_theta)
    phi_values = np.linspace(-1, 1, num_phi)
    theta_values = np.linspace(-1, 1, num_theta)
    codebook = [(phi, theta) for phi in phi_values for theta in theta_values]
    return codebook

# 生成 x, y 的码本
def generate_codebook_x_y(num_x=4, num_y=4):
    x_values = np.linspace(0, (num_x - 1) * lambda_, num_x)
    y_values = np.linspace(0, (num_y - 1) * lambda_, num_y)
    codebook = [(x, y) for x in x_values for y in y_values]
    return codebook

# 从码本中随机选择 phi, theta
def select_random_phi_theta(codebook):
    return codebook[np.random.randint(len(codebook))]

def generate_random_F(Nx_t, Ny_t, Mx_t, My_t, N_t, M_t, codebook):
    diagonal_blocks = []

    for _ in range(N_t):
        phi, theta = select_random_phi_theta(codebook)
        f = array_response_Sa(Mx_t, My_t, math.sin(phi) * math.sin(theta), math.cos(theta)) 
        diagonal_blocks.append(f)

    # 构建对角矩阵，将每个 f 放在对角线上
    F = np.zeros((N_t * M_t, N_t), dtype=complex)

    for i, f in enumerate(diagonal_blocks):
        # 将 f 扁平化并放在对角线上
        F[i * M_t:(i + 1) * M_t, i] = f.flatten()

    return F

def generate_random_WH(Nx_r, Ny_r, Mx_r, My_r, N_r, M_r, codebook):
    diagonal_blocks = []

    for _ in range(N_r):
        phi, theta = select_random_phi_theta(codebook)
        w = array_response_Sa(Mx_r, My_r, math.sin(phi) * math.sin(theta), math.cos(theta)) 
        diagonal_blocks.append(w)

    # 构建对角矩阵，将每个 f 放在对角线上
    W = np.zeros((N_r * M_r, N_r), dtype=complex)

    for i, w in enumerate(diagonal_blocks):
        # 将 f 扁平化并放在对角线上
        W[i * M_r:(i + 1) * M_r, i] = w.flatten()

    WH = W.conj().T
    return WH

