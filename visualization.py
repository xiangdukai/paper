import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

# Define a consistent color palette that works well together
COLORS = {
    'msa': '#2E86C1',      # Strong blue for MSA
    'fsa': '#E74C3C',      # Strong red for FSA
    'msa_variant': '#5DADE2',  # Lighter blue for MSA variant
    'fsa_variant': '#F1948A',  # Lighter red for FSA variant
}

# More distinct blue color palette with strong contrast - less colors, bolder differences
BLUE_PALETTE = [
    '#BBDEFB',  # Light blue that's still clearly visible
    '#2196F3',  # Medium vibrant blue
    '#0D47A1',  # Very dark rich blue
    '#64B5F6',  # Medium-light blue
    '#1976D2',  # Medium-dark blue
    '#E3F2FD',  # Very light blue 
    '#1565C0',  # Dark navy blue
    '#90CAF9',  # Pale but visible blue
    '#0277BD',  # Strong saturated blue
    '#01579B'   # Dark indigo-blue
]

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
    # Plot with different markers and improved colors - keeping original sizes
    plt.plot(objective_new, label='MSA-IOSS', linestyle='-', marker='o', 
             color=COLORS['msa'], markersize=20)
    plt.plot(best_objective_new, label='MSA-ES', linestyle='--', marker='*', 
             color=COLORS['msa_variant'], markersize=24)
    
    plt.plot(objective_old, label='FSA-IOSS', linestyle='-', marker='^', 
             color=COLORS['fsa'], markersize=20)
    plt.plot(best_objective_old, label='FSA-ES', linestyle='--', marker='s', 
             color=COLORS['fsa_variant'], markersize=24)
    
    # Labels with original font size
    plt.xlabel('Iteration', fontsize=28)
    plt.ylabel('Spectral Efficiency', fontsize=28)
    
    # Legend with original font size
    plt.legend(loc='best', fontsize=32)
    
    # Grid and layout adjustments
    plt.grid(True, which='both', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    
    # Original tick font size
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    
    # Save and show plot
    plt.savefig('D:/learning/Jilin University/Sophomore/papers/1/figure/monte_carlo_simulation.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_movable_area_SNR_simulation(SNR_objective, save_data=True, filename=None):
    # Save data if requested
    if save_data:
        data = {
            'SNR_objective': SNR_objective
        }
        save_simulation_data(data, filename)
    
    markers = ['o', '*', '^', 's', 'D', 'v', 'p', 'h', 'x', '+']
    
    plt.figure(figsize=(12, 8))
    
    # Plot with different markers for each SNR - using more distinctive blue shades
    for SNR, obj in enumerate(SNR_objective):
        marker_idx = SNR % len(markers)
        color_idx = SNR % len(BLUE_PALETTE)
        x_values = [i for i in range(1, len(obj) + 1)]
        
        plt.plot(x_values, obj, label=f'SNR = {(SNR + 1) * 5}', 
                 linestyle='-', marker=markers[marker_idx], 
                 color=BLUE_PALETTE[color_idx],
                 markersize=28)
    
    # Labels with original font size
    plt.xlabel('Movable Area', fontsize=28)
    plt.ylabel('Spectral Efficiency', fontsize=28)
    
    # Legend with original font size
    plt.legend(loc='best', fontsize=32)
    
    # Grid and layout adjustments
    plt.grid(True, which='both', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    
    # Original tick font size
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    
    # Save and show plot
    plt.savefig('D:/learning/Jilin University/Sophomore/papers/1/figure/movable_area_snr_simulation.png', 
                dpi=300, bbox_inches='tight')
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
    # CORRECTED: Swapped labels and colors for MSA and FSA
    plt.plot(range(1, len(objective_old_value) + 1), objective_old_value, 
             label='FSA', linestyle='-', marker='^', 
             color=COLORS['fsa'], markersize=28)
    plt.plot(range(1, len(objective_new_value) + 1), objective_new_value, 
             label='MSA', linestyle='-', marker='o', 
             color=COLORS['msa'], markersize=28)
    
    # Labels with original font size
    plt.xlabel('Path\'s number', fontsize=28)
    plt.ylabel('Spectral Efficiency', fontsize=28)
    
    # Legend with original font size
    plt.legend(loc='best', fontsize=32)
    
    # Grid and layout adjustments
    plt.grid(True, which='both', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    
    # Original tick font size
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    
    # Save and show plot
    plt.savefig('D:/learning/Jilin University/Sophomore/papers/1/figure/path_number_simulation.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


# # 加载之前的数据并重新绘图
# data = load_simulation_data("D:/learning/Jilin University/Sophomore/papers/1/data/simulation_data_20250313_093957.json")
# plot_Monte_Carlo_simulation(
#     data['objective_new'], 
#     data['objective_old'], 
#     data['best_objective_new'], 
#     data['best_objective_old'],
#     save_data=False
# )


# # 加载之前的数据并重新绘图 - Movable Area SNR simulation
# data_snr = load_simulation_data("D:/learning/Jilin University/Sophomore/papers/1/data/simulation_data_20250308_134039.json")
# plot_movable_area_SNR_simulation(
#     data_snr['SNR_objective'],
#     save_data=False
# )

# 加载之前的数据并重新绘图 - Path Number simulation
data_path = load_simulation_data("D:/learning/Jilin University/Sophomore/papers/1/data/simulation_data_20250304_174312.json")
plot_path_number_simulation(
    data_path['objective_old_value'],
    data_path['objective_new_value'],
    save_data=False
)
