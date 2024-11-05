from beam_training import *
from visualization import *

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

def main_function(iterations = 1):
    objective_new = [0] * iterations
    objective_old = [0] * iterations

    for iteration in range(iterations):
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

        objective_new[iteration] = beam_training_new(codebook_phi_theta, codebook_x_y, phi_t, theta_t, phi_r, theta_r, phi, theta, \
                            x_t, y_t, x_t0, y_t0, x_r, y_r, x_r0, y_r0, F, WH, H, objective)
        objective_old[iteration] = beam_training_old(codebook_phi_theta, F, WH, H, objective)

    plot_objective_comparison(objective_new, objective_old)

main_function(2)