import numpy as np

# 参数定义
f = 10e9  # 系统频率 10GHz
lambda_ = 3e8 / f  # 天线波长
d = lambda_ / 2  # 用于虚拟信道表示的天线间距
sa_cache = {}

# 阵列响应函数Sp
def array_response_Sp(Nx, Ny, x, y, phi, theta):
    a = np.zeros((Nx * Ny, 1), dtype=complex)
    for n in range(Nx):
        for m in range(Ny):
            a[n * Ny + m] = (1 / np.sqrt(Nx * Ny)) * np.exp(1j * 2 * np.pi / lambda_ * (x[n * Ny + m] * theta + y[n * Ny + m] * phi))
    return a

# 阵列响应函数Sa, d = lambda_ / 2
def array_response_Sa(Nx, Ny, phi, theta):
    if (Nx, Ny, phi, theta) in sa_cache:
        return sa_cache[(Nx, Ny, phi, theta)]
    
    a = np.zeros((Nx * Ny, 1), dtype=complex)
    for n in range(Nx):
        for m in range(Ny):
            a[n * Ny + m] = (1 / np.sqrt(Nx * Ny)) * np.exp(1j * np.pi * (n * theta + m * phi))

    # 用字典记住已经运算过的Sa
    sa_cache[(Nx, Ny, phi, theta)] = a
    return a