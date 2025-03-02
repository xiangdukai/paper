import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

def save_simulation_data(data_dict, filename=None):
    """
    Save simulation data to a JSON file.
    
    Args:
        data_dict: Dictionary containing simulation data
        filename: Optional filename, if None uses timestamp
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_data_{timestamp}.json"
    
    # Create data directory if it doesn't exist
    data_dir = "D:/learning/Jilin University/Sophomore/papers/1/data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    file_path = os.path.join(data_dir, filename)
    
    with open(file_path, 'w') as f:
        json.dump(data_dict, f, indent=4)
    
    print(f"Data saved to {file_path}")
    return file_path

def load_simulation_data(file_path):
    """
    Load simulation data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        Dictionary containing the loaded data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Data loaded from {file_path}")
    return data

def plot_Monte_Carlo_simulation(objective_new, objective_old, best_objective_new, best_objective_old, save_data=True, filename=None):
    # Save data if requested
    if save_data:
        data = {
            'objective_new': objective_new,
            'objective_old': objective_old,
            'best_objective_new': best_objective_new,
            'best_objective_old': best_objective_old
        }
        save_simulation_data(data, filename)
    
    plt.figure(figsize=(12, 8))
    # Plot with different markers and increased font size
    plt.plot(objective_new, label='Objective New', linestyle='-', marker='o', color='red', markersize=20)
    plt.plot(best_objective_new, label='Best Objective New', linestyle='--', marker='*', color='red', markersize=24)
    
    plt.plot(objective_old, label='Objective Old', linestyle='-', marker='^', color='blue', markersize=20)
    plt.plot(best_objective_old, label='Best Objective Old', linestyle='--', marker='s', color='blue', markersize=24)
    
    # Labels with larger font
    plt.xlabel('Iteration', fontsize=28)
    plt.ylabel('R', fontsize=28)
    
    # Larger legend
    plt.legend(loc='best', fontsize=32)
    
    # Grid and layout adjustments for better visualization
    plt.grid(True, which='both', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    
    # Increase tick font size
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    
    # Show plot
    plt.savefig('D:/learning/Jilin University/Sophomore/papers/1/figure/monte_carlo_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_movable_area_SNR_simulation(SNR_objective, save_data=True, filename=None):
    # Save data if requested
    if save_data:
        data = {
            'SNR_objective': SNR_objective
        }
        save_simulation_data(data, filename)
    
    markers = ['o', '*', '^', 's', 'D', 'v', 'p', 'h', 'x', '+']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    plt.figure(figsize=(12, 8))
    # Plot with different markers for each SNR
    for SNR, obj in enumerate(SNR_objective):
        marker_idx = SNR % len(markers)
        color_idx = SNR % len(colors)
        x_values = [i for i in range(1, len(obj) + 1)]
        plt.plot(x_values, obj, label=f'SNR = {(SNR + 1) * 5}', 
                    linestyle='-', marker=markers[marker_idx], color=colors[color_idx],
                    markersize=28)
    
    # Labels with larger font
    plt.xlabel('MA', fontsize=28)
    plt.ylabel('R', fontsize=28)
    
    # Larger legend
    plt.legend(loc='best', fontsize=32)
    
    # Grid and layout adjustments for better visualization
    plt.grid(True, which='both', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    
    # Increase tick font size
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    
    # Show plot
    plt.savefig('D:/learning/Jilin University/Sophomore/papers/1/figure/movable_area_snr_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_path_number_simulation(objective_old_value, objective_new_value, save_data=True, filename=None):
    # Save data if requested
    if save_data:
        data = {
            'objective_old_value': objective_old_value,
            'objective_new_value': objective_new_value
        }
        save_simulation_data(data, filename)
    
    plt.figure(figsize=(12, 8))
    # Plot with different markers and increased size
    plt.plot(range(1, len(objective_old_value) + 1), objective_old_value, 
                label='Old Method', linestyle='-', marker='o', color='red', markersize=28)
    plt.plot(range(1, len(objective_new_value) + 1), objective_new_value, 
                label='New Method', linestyle='-', marker='^', color='blue', markersize=28)
    
    # Labels with larger font
    plt.xlabel('Path\'s number', fontsize=28)
    plt.ylabel('Objective Value', fontsize=28)
    
    # Larger legend
    plt.legend(loc='best', fontsize=32)
    
    # Grid and layout adjustments for better visualization
    plt.grid(True, which='both', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    
    # Increase tick font size
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    
    # Show plot
    plt.savefig('D:/learning/Jilin University/Sophomore/papers/1/figure/path_number_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()

# 示例用法:
# 1. 运行模拟并保存数据
# plot_Monte_Carlo_simulation(objective_new, objective_old, best_objective_new, best_objective_old)

# 2. 加载之前的数据并重新绘图
# data = load_simulation_data("simulation_data/simulation_data_20230515_123045.json")
# plot_Monte_Carlo_simulation(
#     data['objective_new'], 
#     data['objective_old'], 
#     data['best_objective_new'], 
#     data['best_objective_old'],
#     save_data=False
# )