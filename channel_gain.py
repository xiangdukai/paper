import numpy as np
from array_response import array_response_Sa,array_response_Sp

# 参数定义
N_t = 16  # t子阵列数
Nx_t = 4
Ny_t = 4
M_t = 64   # t天线单元数
Mx_t = 8
My_t = 8
N_r = 16  # r子阵列数
Nx_r = 4
Ny_r = 4
M_r = 64   # r天线单元数
Mx_r = 8
My_r = 8
L = 5    # 路径数
f = 10e9  # 系统频率 10GHz
lambda_ = 3e8 / f  # 天线波长
d = lambda_ / 2  # 用于虚拟信道表示的天线间距

alpha = (1 / np.sqrt(2)) * (np.random.normal(0, 1, L + 1) + 1j * np.random.normal(0, 1, L + 1))


def generate_channel(phi_t, theta_t, phi_r, theta_r, phi, theta, x_t, y_t, x_r, y_r):

    H = np.zeros((N_t * M_t, N_r * M_r), dtype=complex)

    # 累加L条路径的贡献
    for l in range(L):
        H += 1/2 * np.sqrt(N_t * M_t * N_r * M_r / 1) * alpha[l] * np.kron(array_response_Sa(Mx_r, My_r, phi_r[l], theta_r[l]), array_response_Sp(Nx_r, Ny_r, x_r, y_r, phi_r[l], theta_r[l])) \
        * np.kron(array_response_Sa(Mx_t, My_t, phi_t[l], theta_t[l]), array_response_Sp(Nx_t, Ny_t, x_t, y_t , phi_t[l], theta_t[l])).conj().T
    
    H += np.sqrt(N_t * M_t * N_r * M_r / 1) * alpha[L] * np.kron(array_response_Sa(Mx_r, My_r, phi, theta), array_response_Sp(Nx_r, Ny_r, x_r, y_r, phi, theta)) \
        * np.kron(array_response_Sa(Mx_t, My_t, phi, theta), array_response_Sp(Nx_t, Ny_t, x_t, y_t , phi, theta)).conj().T
    return H
