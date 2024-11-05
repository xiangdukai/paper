import matplotlib.pyplot as plt

def plot_objective_comparison(objective_new, objective_old):
    print(objective_new)
    print(objective_old)
    
    plt.figure(figsize=(10, 6))

    # Plot new objective values
    for i, obj_new in enumerate(objective_new):
        plt.plot(obj_new, label=f'Objective New - Iteration {i+1}', linestyle='-', marker='o')
    
    # Plot old objective values
    for i, obj_old in enumerate(objective_old):
        plt.plot(obj_old, label=f'Objective Old - Iteration {i+1}', linestyle='--', marker='x')
    
    # Labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Comparison of Objective Values between New and Old Methods')
    
    # Legend
    plt.legend(loc='best')
    
    # Grid and layout adjustments for better visualization
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Show plot
    plt.show()
