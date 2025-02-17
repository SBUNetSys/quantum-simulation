import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24


def plot_decay_function(alpha=63109):
    # Create time values in microseconds
    t = np.linspace(0, 20, 1000)  # 0 to 20 microseconds

    # Calculate F values (converting microseconds to seconds by dividing t by 1e6)
    F = 0.75 * np.exp(-alpha * t / 1e6) + 0.25

    # Create the plot
    plt.figure(figsize=(12, 9))
    plt.plot(t, F, '-', label='Fidelity', linewidth=3)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=1)

    # Add labels and title
    plt.xlabel('Time (μs)')  # Changed to microseconds
    plt.ylabel('Fidelity')
    plt.title('Fidelity Over Time')

    # Add legend
    plt.legend()

    # Set axis limits
    plt.xlim(0, 20)  # Display 0 to 20 microseconds
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('./figures/decay_function.png')
    # Show the plot
    plt.show()


if __name__ == '__main__':
    # Call the function
    plot_decay_function()