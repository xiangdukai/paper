import matplotlib.pyplot as plt

def plot_Monte_Carlo_simulation(objective_new, objective_old, best_objective_new, best_objective_old):
    
    plt.figure(figsize=(10, 6))

    # Plot new objective values
    plt.plot(objective_new, label=f'Objective New', linestyle='-', marker='o', color='red')

    plt.plot(best_objective_new, label=f'Best Objective New', linestyle='--', marker='*', color='red')
    
    # Plot old objective values
    plt.plot(objective_old, label=f'Objective Old', linestyle='-', marker='o', color='blue')

    plt.plot(best_objective_old, label=f'Best Objective Old', linestyle='--', marker='*', color='blue')
    
    # Labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Monte Carlo simulation')
    
    # Legend
    plt.legend(loc='best')
    
    # Grid and layout adjustments for better visualization
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Show plot
    plt.show()

def plot_movable_area_SNR_simulation(SNR_objective):
    plt.figure(figsize=(10, 6))

    # Plot new objective values
    for SNR, obj in enumerate(SNR_objective):
        plt.plot(obj, label=f'SNR = {(SNR - 1) * 5}', linestyle='-', marker='o')
    
    # Labels and title
    plt.xlabel('Movable Area')
    plt.ylabel('Objective Value')
    plt.title('movable_area_SNR_simulation')
    
    # Legend
    plt.legend(loc='best')
    
    # Grid and layout adjustments for better visualization
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Show plot
    plt.show()