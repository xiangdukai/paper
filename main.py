from beam_training import *
from visualization import *

def main_function(iterations = 1):
    objective_new = [0] * iterations
    objective_old = [0] * iterations
    best_objective_old = [0] * iterations
    best_objective_new = [0] * iterations

    for iteration in range(iterations):
        # 初始化波束训练参数
        phi_t, theta_t, phi_r, theta_r, phi, theta, alpha = initialize_H_parameter()
        
        # 初始化x_t, y_t, x_r, y_r
        codebook_x_y = generate_codebook_x_y(2, 2)
        x_t, y_t, x_t0, y_t0 = initialize_x_y(Nx_t, Ny_t)
        x_r, y_r, x_r0, y_r0 = initialize_x_y(Nx_r, Ny_r)
        
        # 生成随机信道增益
        H = generate_channel(phi_t, theta_t, phi_r, theta_r, phi, theta, x_t, y_t, x_r, y_r, alpha)

        # 生成随机F,WH
        codebook_phi_theta = generate_codebook_phi_theta(3, 2)
        F = generate_random_F(codebook_phi_theta)
        WH = generate_random_WH(codebook_phi_theta)

        # 计算目标函数
        objective = calculate_objective_function(WH, H, F)
        print(f"Random Objective Value: {objective}")

        objective_old[iteration] = beam_training_old(codebook_phi_theta, F, WH, H, objective)

        objective_new[iteration] = beam_training_new(codebook_phi_theta, codebook_x_y, phi_t, theta_t, phi_r, theta_r, phi, theta, \
                            x_t, y_t, x_t0, y_t0, x_r, y_r, x_r0, y_r0, F, WH, H, objective, alpha)

        best_objective_old[iteration] = beam_training_exhaustive_old(codebook_phi_theta, F, WH, H, objective)

        best_objective_new[iteration] = beam_training_exhaustive_new(codebook_phi_theta, codebook_x_y, phi_t, theta_t, phi_r, theta_r, 
                                phi, theta, x_t, y_t, x_t0, y_t0, x_r, y_r, x_r0, y_r0, F, WH, H, objective, alpha)

    plot_objective_comparison(objective_new, objective_old, best_objective_new, best_objective_old)


main_function(1)