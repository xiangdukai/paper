from beam_training import *
from visualization import *

def Monte_Carlo_simulation(rounds, Nx_t=2, Ny_t=1, Mx_t=8, My_t=8, Nx_r=2, Ny_r=1, Mx_r=8, My_r=8):
    N_t = Nx_t * Ny_t
    M_t = Mx_t * My_t
    N_r = Nx_r * Ny_r
    M_r = Mx_r * My_r

    objective_new = [0] * rounds
    objective_old = [0] * rounds
    best_objective_old = [0] * rounds
    best_objective_new = [0] * rounds

    L = 5
    iterations = 7

    for round in range(rounds):
        # 初始化波束训练参数
        phi_t, theta_t, phi_r, theta_r, phi, theta, alpha = initialize_H_parameter(L)
        
        # 初始化x_t, y_t, x_r, y_r
        codebook_x_y = generate_codebook_x_y(2, 2)
        x_t, y_t, x_t0, y_t0 = initialize_x_y(Nx_t, Ny_t, Mx_t, My_t, num_x=2, num_y=2)
        x_r, y_r, x_r0, y_r0 = initialize_x_y(Nx_r, Ny_r, Mx_r, My_r, num_x=2, num_y=2)
        
        # 生成随机信道增益
        H = generate_channel(Nx_t, Ny_t, Mx_t, My_t, Nx_r, Ny_r, Mx_r, My_r, N_t, M_t, N_r, M_r, phi_t, theta_t, phi_r, theta_r, phi, theta, x_t, y_t, x_r, y_r, alpha, L)

        # 生成随机F,WH
        codebook_phi_theta = generate_codebook_phi_theta(2, 2)
        F = generate_random_F(Nx_t, Ny_t, Mx_t, My_t, N_t, M_t, codebook_phi_theta)
        WH = generate_random_WH(Nx_r, Ny_r, Mx_r, My_r, N_r, M_r, codebook_phi_theta)

        # 计算目标函数
        objective = calculate_objective_function(WH, H, F, SNR=10)
        print(f"Random Objective Value: {objective}")

        objective_old[round] = beam_training_old(Nx_t, Ny_t, Mx_t, My_t, Nx_r, Ny_r, Mx_r, My_r, N_t, M_t, N_r, M_r, iterations, codebook_phi_theta, F, WH, H, objective, SNR=10)

        objective_new[round] = beam_training_new(Nx_t, Ny_t, Mx_t, My_t, Nx_r, Ny_r, Mx_r, My_r, N_t, M_t, N_r, M_r, iterations, codebook_phi_theta, codebook_x_y, phi_t, theta_t, phi_r, theta_r, phi, theta, \
                            x_t, y_t, x_t0, y_t0, x_r, y_r, x_r0, y_r0, F, WH, H, objective, alpha, L, SNR=10)

        best_objective_old[round] = beam_training_exhaustive_old(Nx_t, Ny_t, Mx_t, My_t, Nx_r, Ny_r, Mx_r, My_r, N_t, M_t, N_r, M_r, iterations, codebook_phi_theta, F, WH, H, objective)

        best_objective_new[round] = beam_training_exhaustive_new(Nx_t, Ny_t, Mx_t, My_t, Nx_r, Ny_r, Mx_r, My_r, N_t, M_t, N_r, M_r, iterations, codebook_phi_theta, codebook_x_y, phi_t, theta_t, phi_r, theta_r, 
                                phi, theta, x_t, y_t, x_t0, y_t0, x_r, y_r, x_r0, y_r0, F, WH, H, objective, alpha, L)

    #print(objective_new)

    average_objective_new = [sum(values) / rounds for values in zip(*objective_new)]
    average_objective_old = [sum(values) / rounds for values in zip(*objective_old)]
    average_best_objective_new = [sum(values) / rounds for values in zip(*best_objective_new)]
    average_best_objective_old = [sum(values) / rounds for values in zip(*best_objective_old)]

    #print(average_objective_new)

    plot_Monte_Carlo_simulation(average_objective_new, average_objective_old, average_best_objective_new, average_best_objective_old)

def movable_area_SNR_simulation(rounds=1, Nx_t=4, Ny_t=4, Mx_t=8, My_t=8, Nx_r=4, Ny_r=4, Mx_r=8, My_r=8):
    N_t = Nx_t * Ny_t
    M_t = Mx_t * My_t
    N_r = Nx_r * Ny_r
    M_r = Mx_r * My_r

    objective_history = [0] * rounds
    SNR_objective = []

    L = 5
    iterations = 6

    # 初始化波束训练参数
    phi_t, theta_t, phi_r, theta_r, phi, theta, alpha = initialize_H_parameter(L)

    for SNR in range(-5, 25, 5):
        objective_value = []
        for num in range(1, 8):
            # 初始化x_t, y_t, x_r, y_r
            codebook_x_y = generate_codebook_x_y(num, num)
            x_t, y_t, x_t0, y_t0 = initialize_x_y(Nx_t, Ny_t, Mx_t, My_t, num, num)
            x_r, y_r, x_r0, y_r0 = initialize_x_y(Nx_r, Ny_r, Mx_r, My_r, num, num)
            
            # 生成信道增益
            H = generate_channel(Nx_t, Ny_t, Mx_t, My_t, Nx_r, Ny_r, Mx_r, My_r, N_t, M_t, N_r, M_r, phi_t, theta_t, phi_r, theta_r, phi, theta, x_t, y_t, x_r, y_r, alpha, L)

            # 生成随机F,WH
            codebook_phi_theta = generate_codebook_phi_theta(8, 8)
            F = generate_random_F(Nx_t, Ny_t, Mx_t, My_t, N_t, M_t, codebook_phi_theta)
            WH = generate_random_WH(Nx_r, Ny_r, Mx_r, My_r, N_r, M_r, codebook_phi_theta)

            # 计算目标函数
            objective = calculate_objective_function(WH, H, F, SNR)
            print(f"Random Objective Value: {objective}")

            objective_history = beam_training_new(Nx_t, Ny_t, Mx_t, My_t, Nx_r, Ny_r, Mx_r, My_r, N_t, M_t, N_r, M_r, iterations, codebook_phi_theta, codebook_x_y, phi_t, theta_t, phi_r, theta_r, phi, theta, \
                                x_t, y_t, x_t0, y_t0, x_r, y_r, x_r0, y_r0, F, WH, H, objective, alpha, L, SNR)
            objective_value.append(objective_history[-1])

            #average_objective.append([sum(values) / rounds for values in zip(*objective_history)][-1])
        SNR_objective.append(objective_value)

    plot_movable_area_SNR_simulation(SNR_objective)



#Monte_Carlo_simulation(rounds=4)
movable_area_SNR_simulation()