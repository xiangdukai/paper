import numpy as np
from array_response import array_response_Sa,array_response_Sp

def generate_channel(Nx_t, Ny_t, Mx_t, My_t, Nx_r, Ny_r, Mx_r, My_r, N_t, M_t, N_r, M_r, phi_t, theta_t, phi_r, theta_r, phi, theta, x_t, y_t, x_r, y_r, alpha, L):

    H = np.zeros((N_r * M_r, N_t * M_t), dtype=complex)
    scale_factor = np.sqrt(N_t * M_t * N_r * M_r / 1)

    # 累加L条路径的贡献
    for l in range(L):
        H += 1/2 * scale_factor * alpha[l] * np.kron(array_response_Sa(Mx_r, My_r, phi_r[l], theta_r[l]), array_response_Sp(Nx_r, Ny_r, x_r, y_r, phi_r[l], theta_r[l])) \
        @ (np.kron(array_response_Sa(Mx_t, My_t, phi_t[l], theta_t[l]), array_response_Sp(Nx_t, Ny_t, x_t, y_t , phi_t[l], theta_t[l])).conj().T)
    
    H += scale_factor * alpha[L] * np.kron(array_response_Sa(Mx_r, My_r, phi, theta), array_response_Sp(Nx_r, Ny_r, x_r, y_r, phi, theta)) \
        @ (np.kron(array_response_Sa(Mx_t, My_t, phi, theta), array_response_Sp(Nx_t, Ny_t, x_t, y_t , phi, theta)).conj().T)
    return H
