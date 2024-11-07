import numpy as np
from channel_gain import generate_channel
from array_response import array_response_Sa
from precoding_matrix import *

# 参数定义
N_t = 2  # t子阵列数
Nx_t = 2
Ny_t = 1
M_t = 64  # t天线单元数
Mx_t = 8
My_t = 8
N_r = 2  # r子阵列数
Nx_r = 2
Ny_r = 1
M_r = 64   # r天线单元数
Mx_r = 8
My_r = 8
L = 3
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

    alpha = (1 / np.sqrt(2)) * (np.random.normal(0, 1, L + 1) + 1j * np.random.normal(0, 1, L + 1))

    return phi_t, theta_t, phi_r, theta_r, phi, theta, alpha

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
    
    n, m = R.shape
    I = np.eye(n, m)

    objective = np.log2(np.linalg.det(I + 1 * np.linalg.inv(R) @ WHF @ (WHF.conj().T))).real

    return objective

def beam_training_new(codebook_phi_theta, codebook_x_y, phi_t, theta_t, phi_r, theta_r, phi, theta, \
                        x_t, y_t, x_t0, y_t0, x_r, y_r, x_r0, y_r0, F, WH, H, objective, alpha):
    objective_history = [objective]
    for _ in range(5):
        # 选择波束
        F_temp = F.copy()
        for i in range(N_t):
            for k in range(len(codebook_phi_theta)):
                f = array_response_Sa(Mx_t, My_t, math.sin(codebook_phi_theta[k][0]) * math.sin(codebook_phi_theta[k][1]), \
                                        math.cos(codebook_phi_theta[k][1]))
                F_temp[i * M_t:(i + 1) * M_t, i] = f.flatten()
                if(calculate_objective_function(WH, H, F_temp) > objective):
                    objective = calculate_objective_function(WH, H, F_temp)
                    F = F_temp.copy()

        W = WH.conj().T
        W_temp = W.copy()
        for i in range(N_r):
            for k in range(len(codebook_phi_theta)):
                w = array_response_Sa(Mx_r, My_r, math.sin(codebook_phi_theta[k][0]) * math.sin(codebook_phi_theta[k][1]), \
                                        math.cos(codebook_phi_theta[k][1]))
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
                H_temp = generate_channel(phi_t, theta_t, phi_r, theta_r, phi, theta, x_t, y_t, x_r, y_r, alpha)
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
                H_temp = generate_channel(phi_t, theta_t, phi_r, theta_r, phi, theta, x_t, y_t, x_r, y_r, alpha)
                if(calculate_objective_function(WH, H_temp, F) > objective):
                    objective = calculate_objective_function(WH, H_temp, F)
                    x_r_temp = x_r.copy()
                    y_r_temp = y_r.copy()
                    H = H_temp.copy()
            x_r = x_r_temp.copy()
            y_r = y_r_temp.copy()

        # 储存当前objective
        objective_history.append(objective)
        print(f"Objective Value: {objective}")
    
    # 输出最终的波束训练结果
    print(f"Final Objective Value: {objective}")
    print(f"Final Selected Position: {(x_t - x_t0).T}, {(y_t - y_t0).T}")

    return objective_history

def beam_training_old(codebook_phi_theta, F, WH, H, objective):
    objective_history = [objective]
    for _ in range(5):
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

        objective_history.append(objective)
        print(f"Objective Value: {objective}")

    # 输出最终的波束训练结果
    print(f"Final Objective Value: {objective}")

    return objective_history

def beam_training_exhaustive_old(codebook_phi_theta, F, WH, H, objective):
    """
    使用深度优先搜索进行波束训练的穷举算法（优化版）
    
    参数:
    codebook_phi_theta: 码本
    F: 发射端预编码矩阵
    WH: 接收端组合矩阵的共轭转置
    H: 信道矩阵
    objective: 初始目标函数值
    """
    best_objective = objective
    best_F = F.copy()
    best_WH = WH.copy()
    
    # 预计算和缓存波束响应
    transmitter_responses = [array_response_Sa(Mx_t, My_t,
                                            math.sin(phi_theta[0]) * math.sin(phi_theta[1]),
                                            math.cos(phi_theta[1])).flatten()
                                for phi_theta in codebook_phi_theta]
                            
    receiver_responses = [array_response_Sa(Mx_r, My_r,
                                            math.sin(phi_theta[0]) * math.sin(phi_theta[1]),
                                            math.cos(phi_theta[1])).flatten()
                            for phi_theta in codebook_phi_theta]

    def dfs_transmitter(depth, current_F):
        """
        发射端波束的深度优先搜索
        depth: 当前搜索的子阵列索引
        current_F: 当前的预编码矩阵
        """
        nonlocal best_objective, best_F, best_WH
        
        if depth == N_t:
            dfs_receiver(0, WH.conj().T.copy(), current_F)
            return
        
        for k, f in enumerate(transmitter_responses):
            current_F[depth * M_t:(depth + 1) * M_t, depth] = f
            dfs_transmitter(depth + 1, current_F)
    
    def dfs_receiver(depth, current_W, current_F):
        """
        接收端波束的深度优先搜索
        depth: 当前搜索的子阵列索引
        current_W: 当前的组合矩阵
        current_F: 当前使用的预编码矩阵
        """
        nonlocal best_objective, best_F, best_WH
        
        if depth == N_r:
            current_WH = current_W.conj().T
            current_objective = calculate_objective_function(current_WH, H, current_F)
            
            if current_objective > best_objective:
                best_objective = current_objective
                best_F = current_F.copy()
                best_WH = current_WH.copy()
                print(f"Found better objective: {best_objective}")
            return
        
        for k, w in enumerate(receiver_responses):
            current_W[depth * M_r:(depth + 1) * M_r, depth] = w
            dfs_receiver(depth + 1, current_W, current_F)
    
    dfs_transmitter(0, F.copy())
    
    print(f"Final Objective Value: {best_objective}")
    best_objective = [best_objective] * 6
    return best_objective

def beam_training_exhaustive_new(codebook_phi_theta, codebook_x_y, phi_t, theta_t, phi_r, theta_r, 
                                phi, theta, x_t, y_t, x_t0, y_t0, x_r, y_r, x_r0, y_r0, F, WH, H, objective, alpha):
    """
    使用深度优先搜索进行波束训练和位置优化的穷举算法（优化版）
    """
    best_objective = 0
    best_F = F.copy()
    best_WH = WH.copy()
    best_x_t = x_t.copy()
    best_y_t = y_t.copy()
    best_x_r = x_r.copy()
    best_y_r = y_r.copy()
    best_H = H.copy()
    
    # 缓存波束和位置响应，避免重复计算
    transmitter_beam_responses = [array_response_Sa(Mx_t, My_t, 
                                                    math.sin(pt[0]) * math.sin(pt[1]), 
                                                    math.cos(pt[1])).flatten()
                                    for pt in codebook_phi_theta]
                                
    receiver_beam_responses = [array_response_Sa(Mx_r, My_r, 
                                                math.sin(pt[0]) * math.sin(pt[1]), 
                                                math.cos(pt[1])).flatten()
                                for pt in codebook_phi_theta]
    
    # 缓存位置偏移值
    transmitter_position_offsets = [(x_t0[i] + xy[0], y_t0[i] + xy[1]) for i in range(N_t) for xy in codebook_x_y]
    receiver_position_offsets = [(x_r0[i] + xy[0], y_r0[i] + xy[1]) for i in range(N_r) for xy in codebook_x_y]

    def dfs_transmitter_position(pos_depth, current_x_t, current_y_t):
        
        if pos_depth == N_t:
            dfs_receiver_position(0, current_x_t, current_y_t, x_r.copy(), y_r.copy())
            return
            
        for (x_offset, y_offset) in transmitter_position_offsets:
            x_t_temp = current_x_t.copy()
            y_t_temp = current_y_t.copy()
            x_t_temp[pos_depth], y_t_temp[pos_depth] = x_offset, y_offset
            
            dfs_transmitter_position(pos_depth + 1, x_t_temp, y_t_temp)
    
    def dfs_receiver_position(pos_depth, current_x_t, current_y_t, current_x_r, current_y_r):
        
        if pos_depth == N_r:
            current_H = generate_channel(phi_t, theta_t, phi_r, theta_r, phi, theta, 
                            current_x_t, current_y_t, current_x_r, current_y_r, alpha)
            dfs_transmitter_beam(0, F.copy(), current_x_t, current_y_t, current_x_r, current_y_r, current_H)
            return
            
        for (x_offset, y_offset) in receiver_position_offsets:
            x_r_temp = current_x_r.copy()
            y_r_temp = current_y_r.copy()
            x_r_temp[pos_depth], y_r_temp[pos_depth] = x_offset, y_offset
            
            dfs_receiver_position(pos_depth + 1, current_x_t, current_y_t, x_r_temp, y_r_temp)
    
    def dfs_transmitter_beam(beam_depth, current_F, current_x_t, current_y_t, current_x_r, current_y_r, current_H):
        
        if beam_depth == N_t:
            dfs_receiver_beam(0, WH.conj().T.copy(), current_F, current_x_t, current_y_t, current_x_r, current_y_r, current_H)
            return
            
        for f in transmitter_beam_responses:
            F_temp = current_F.copy()
            F_temp[beam_depth * M_t:(beam_depth + 1) * M_t, beam_depth] = f
            dfs_transmitter_beam(beam_depth + 1, F_temp, current_x_t, current_y_t, current_x_r, current_y_r, current_H)
    
    def dfs_receiver_beam(beam_depth, current_W, current_F, current_x_t, current_y_t, current_x_r, current_y_r, current_H):
        nonlocal best_objective, best_F, best_WH, best_x_t, best_y_t, best_x_r, best_y_r, best_H
        
        if beam_depth == N_r:
            current_WH = current_W.conj().T
            current_objective = calculate_objective_function(current_WH, current_H, current_F)
            # print(f"Current Objective: {current_objective}")
            
            if current_objective > best_objective:
                best_objective = current_objective.copy()
                best_F = current_F.copy()
                best_WH = current_WH.copy()
                best_x_t = current_x_t.copy()
                best_y_t = current_y_t.copy()
                best_x_r = current_x_r.copy()
                best_y_r = current_y_r.copy()
                best_H = current_H.copy()
                print(f"Found better objective: {best_objective}")
            return
            
        for w in receiver_beam_responses:
            W_temp = current_W.copy()
            W_temp[beam_depth * M_r:(beam_depth + 1) * M_r, beam_depth] = w
            dfs_receiver_beam(beam_depth + 1, W_temp, current_F, current_x_t, current_y_t, current_x_r, current_y_r, current_H)
    
    dfs_transmitter_position(0, x_t.copy(), y_t.copy())
    
    print(f"Final Objective Value: {best_objective}")
    best_objective_history = [best_objective] * 6
    return best_objective_history