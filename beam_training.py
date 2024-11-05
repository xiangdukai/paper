import numpy as np
from channel_gain import generate_channel
from array_response import array_response_Sp,array_response_Sa
from precoding_matrix import *

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
def initialize_H_parameter():
    # 发射路径方位角 (AoA azimuth) phi_r = sin(phi_true)sin(theta)
    phi_t = np.random.uniform(-1, 1, L)
    # 发射路径俯仰角 (AoA elevation) theta_r = cos(theta)
    theta_t = np.random.uniform(-1, 1, L)
    # 发射路径方位角 (AoA azimuth) phi_r = sin(phi_true)sin(theta)
    phi_r = np.random.uniform(-1, 1, L)
    # 发射路径俯仰角 (AoA elevation) theta_r = cos(theta)
    theta_r = np.random.uniform(-1, 1, L)
    # 直射路径方位角 (AoA azimuth) phi_r = sin(phi_true)sin(theta)
    phi = np.random.uniform(-1, 1)
    # 直射路径俯仰角 (AoA elevation) theta_r = cos(theta)
    theta = np.random.uniform(-1, 1)

    return phi_t, theta_t, phi_r, theta_r, phi, theta

def initialize_x_y(Nx, Ny):
    N = Nx * Ny
    x = np.zeros((N, 1))
    y = np.zeros((N, 1))

    for i in range(Nx):
        for j in range(Ny):
            x[i * Ny_r + j] = i * 12 * lambda_
            y[i * Ny_r + j] = j * 12 * lambda_
    x0 = x.copy()
    y0 = y.copy()

    codebook_x_y = generate_codebook_x_y()
    for i in range(Nx):
        for j in range(Ny):
            code = codebook_x_y[np.random.randint(len(codebook_x_y))]
            x[i * Ny + j] += code[0]
            y[i * Ny + j] += code[1]

    return x, y, x0, y0

def calculate_objective_function(WH, H, F):
    WHF = WH @ H @ F
    R = 1 * WH @ (WH.conj().T)
    
    n, m = WHF.shape
    I = np.eye(n, m)

    objective = np.linalg.det(I + 1 * np.linalg.inv(R) @ WHF @ (WHF.conj().T))

    return objective

def beam_training_new():
    # 初始化波束训练参数
    phi_t, theta_t, phi_r, theta_r, phi, theta = initialize_H_parameter()
    
    # 初始化x_t, y_t, x_r, y_r
    codebook_x_y = generate_codebook_x_y()
    x_t, y_t, x_t0, y_t0 = initialize_x_y(Nx_t, Ny_t)
    x_r, y_r, x_r0, y_r0 = initialize_x_y(Nx_r, Ny_r)
    
    # 生成随机信道增益
    H = generate_channel(phi_t, theta_t, phi_r, theta_r, phi, theta, x_t, y_t, x_r, y_r)

    # 生成随机F,WH
    codebook_phi_theta = generate_codebook_phi_theta(num_phi=8, num_theta=8)
    F = generate_random_F(codebook_phi_theta)
    WH = generate_random_WH(codebook_phi_theta)

    # 计算目标函数
    objective = calculate_objective_function(WH, H, F)
    print(f"Random Objective Value: {objective}")

    for _ in range(3):
        # 选择波束
        F_temp = F.copy()
        for i in range(N_t):
            for k in range(len(codebook_phi_theta)):
                f = array_response_Sa(Mx_t, My_t, math.sin(codebook_phi_theta[k][0]) * math.sin(codebook_phi_theta[k][1]), math.cos(codebook_phi_theta[k][1]))
                F_temp[i * M_t:(i + 1) * M_t, i] = f.flatten()
                if(calculate_objective_function(WH, H, F_temp) > objective):
                    objective = calculate_objective_function(WH, H, F_temp)
                    F = F_temp.copy()

        W = WH.conj().T
        W_temp = W.copy()
        for i in range(N_r):
            for k in range(len(codebook_phi_theta)):
                w = array_response_Sa(Mx_r, My_r, math.sin(codebook_phi_theta[k][0]) * math.sin(codebook_phi_theta[k][1]), math.cos(codebook_phi_theta[k][1]))
                W_temp[i * M_r:(i + 1) * M_r, i] = w.flatten()
                WH_temp = W_temp.conj().T
                if(calculate_objective_function(WH_temp, H, F) > objective):
                    objective = calculate_objective_function(WH_temp, H, F)
                    WH = WH_temp.copy()

        print(f"Objective Value: {objective}")

        # 选择位置
        for i in range(N_t):
            x_t_temp = x_t.copy()
            y_t_temp = y_t.copy()
            for k in range(len(codebook_x_y)):
                x, y = codebook_x_y[k]
                x_t[i] = x_t0[i] + x
                y_t[i] = y_t0[i] + y
                H_temp = generate_channel(phi_t, theta_t, phi_r, theta_r, phi, theta, x_t, y_t, x_r, y_r)
                if(calculate_objective_function(WH, H_temp, F) > objective):
                    objective = calculate_objective_function(WH, H_temp, F)
                    x_t_temp = x_t.copy()
                    y_t_temp = y_t.copy()
                    H = H_temp.copy() 
            x_t = x_t_temp.copy()
            y_t = y_t_temp.copy()

        for i in range(N_r):
            x_r_temp = x_r.copy()
            y_r_temp = y_r.copy()
            for k in range(len(codebook_x_y)):
                x, y = codebook_x_y[k]
                x_r[i] = x_r0[i] + x
                y_r[i] = y_r0[i] + y
                H_temp = generate_channel(phi_t, theta_t, phi_r, theta_r, phi, theta, x_t, y_t, x_r, y_r)
                if(calculate_objective_function(WH, H_temp, F) > objective):
                    objective = calculate_objective_function(WH, H_temp, F)
                    x_r_temp = x_r.copy()
                    y_r_temp = y_r.copy()
                    H = H_temp.copy()
            x_r = x_r_temp.copy()
            y_r = y_r_temp.copy()
    
    # 输出最终的波束训练结果
    print(f"Final Objective Value: {objective}")
    print(f"Final Selected Position: {x_t - x_t0}, {y_t - y_t0}")

def beam_training_old():
    # 初始化波束训练参数
    phi_t, theta_t, phi_r, theta_r, phi, theta = initialize_H_parameter()
    
    # 初始化x_t, y_t, x_r, y_r
    _, _, x_t0, y_t0 = initialize_x_y(Nx_t, Ny_t)
    _, _, x_r0, y_r0 = initialize_x_y(Nx_r, Ny_r)
    
    # 生成随机信道增益
    H = generate_channel(phi_t, theta_t, phi_r, theta_r, phi, theta, x_t0, y_t0, x_r0, y_r0)

    # 生成随机F,WH
    codebook_phi_theta = generate_codebook_phi_theta(num_phi=8, num_theta=8)
    F = generate_random_F(codebook_phi_theta)
    WH = generate_random_WH(codebook_phi_theta)

    # 计算目标函数
    objective = calculate_objective_function(WH, H, F)
    print(f"Random Objective Value: {objective}")

    for _ in range(6):
        # 选择波束
        F_temp = F.copy()
        for i in range(N_t):
            for k in range(len(codebook_phi_theta)):
                f = array_response_Sa(Mx_t, My_t, math.sin(codebook_phi_theta[k][0]) * math.sin(codebook_phi_theta[k][1]), math.cos(codebook_phi_theta[k][1]))
                F_temp[i * M_t:(i + 1) * M_t, i] = f.flatten()
                if(calculate_objective_function(WH, H, F_temp) > objective):
                    objective = calculate_objective_function(WH, H, F_temp)
                    F = F_temp.copy()

        W = WH.conj().T
        W_temp = W.copy()
        for i in range(N_r):
            for k in range(len(codebook_phi_theta)):
                w = array_response_Sa(Mx_r, My_r, math.sin(codebook_phi_theta[k][0]) * math.sin(codebook_phi_theta[k][1]), math.cos(codebook_phi_theta[k][1]))
                W_temp[i * M_r:(i + 1) * M_r, i] = w.flatten()
                WH_temp = W_temp.conj().T
                if(calculate_objective_function(WH_temp, H, F) > objective):
                    objective = calculate_objective_function(WH_temp, H, F)
                    WH = WH_temp.copy()

        print(f"Objective Value: {objective}")

    # 输出最终的波束训练结果
    print(f"Final Objective Value: {objective}")
