import numpy as np

# 参数定义
f = 10e9  # 系统频率 10GHz
lambda_ = 3e8 / f  # 天线波长
d = lambda_ / 2  # 用于虚拟信道表示的天线间距
sp_cache = {}
sa_cache = {}

# 优化的阵列响应函数 Sp
def array_response_Sp(Nx, Ny, x, y, phi, theta):

    indices = np.arange(Nx * Ny).reshape(Nx, Ny)
    x_coords = x[indices].flatten()
    y_coords = y[indices].flatten()
    exponent = 1j * 2 * np.pi / lambda_ * (x_coords * theta + y_coords * phi)
    a = (1 / np.sqrt(Nx * Ny)) * np.exp(exponent)

    return a.reshape(-1, 1)

# 优化的阵列响应函数 Sa，包含缓存机制
def array_response_Sa(Nx, Ny, phi, theta):
    if (Nx, Ny, phi, theta) in sa_cache:
        return sa_cache[(Nx, Ny, phi, theta)]

    indices = np.arange(Nx * Ny).reshape(Nx, Ny)
    n_indices = indices // Ny
    m_indices = indices % Ny
    exponent = 1j * np.pi * (n_indices.flatten() * theta + m_indices.flatten() * phi)
    a = (1 / np.sqrt(Nx * Ny)) * np.exp(exponent).reshape(-1, 1)
    
    # 缓存计算结果
    sa_cache[(Nx, Ny, phi, theta)] = a
    return a