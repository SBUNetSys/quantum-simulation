import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


def plot_grouped_bars(df: pd.DataFrame,
                      group_by: str,
                      title: str,
                      x_label: str,
                      y_label: str,
                      save_name: str = None,
                      loc_str:str ="best",
                      ncol=3):
    """
    Create a grouped bar plot.

    Parameters:
        df: DataFrame with data to plot
        group_by: Column to group by (x-axis)
        title: Title of plot
        x_label: Label of x-axis
        y_label: Label of y-axis
        save_name: Optional filename to save the plot
        loc_str: Optional location of legend
        ncol: optional ncol of legend
    """
    # Create figure with specified DPI
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)  # Set DPI here

    # Create grouped bar plot
    df.plot(kind='bar',
            x=group_by,
            ax=ax)  # Use the created axis

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=14)

    # Customize plot
    # plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=0)
    # plt.legend()#title='Method'
    plt.legend(fontsize=16, loc=loc_str, ncol=ncol)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # Adjust layout
    plt.tight_layout()

    # Save if filename provided
    if save_name:
        plt.savefig(save_name)

    # Show plot
    plt.show()


if __name__ == '__main__':
    # Success Rate
    data = {
        "Node Count": [3, 4],
        "End to End": [0.4749, 0.4959],
        "End to End with Verification": [0.547, 0.608, ],
        "Multi-Hop": [0.788, 0.91],
    }
    data_frame = pd.DataFrame(data)
    plot_grouped_bars(data_frame, "Node Count",
                      "Qubit Transportation Success Rate Comparison",
                      "Network Node Count",
                      "Success Rate (Fidelity > 0.7)",
                      save_name="./figures/success_rate_comparison.png")
    # Throughput
    data = {
        "Node Count": [3, 4],
        "End to End with Verification": [6.917, 4.415],
        "Multi-Hop": [8.404, 6.9351],
    }
    data_frame = pd.DataFrame(data)
    plot_grouped_bars(data_frame, "Node Count",
                      "Transmission Throughput Comparison",
                      "Network Node Count",
                      "Transmitted Qubits Count",
                      save_name="./figures/throughput_comparison.png")

    # Transmission Time
    data = {
        "Node Count": [3, 4],
        "End to End": [16.82, 24.41],
        "Multi-Hop with Purification": [11.50, 14.39],
    }
    data_frame = pd.DataFrame(data)
    plot_grouped_bars(data_frame, "Node Count",
                      "Transmission Time Comparison",
                      "Network Node Count",
                      "Transmission Time (ms)",
                      save_name="./figures/transmission_time_comparison.png")

    # throughput
    data = {
        "Node Count": [3, 4],
        "End to End with Verification": [0.5544, 0.6129],
        "Multi-Hop": [0.7676, 0.9262],
    }
    data_frame = pd.DataFrame(data)
    plot_grouped_bars(data_frame, "Node Count",
                      "Qubit Transportation Success Rate Comparison (0.5Km Distance)",
                      "Network Node Count",
                      "Success Rate (Fidelity > 0.7)",
                      save_name="./figures/success_rate_comparison_0.5km.png",
                      ncol=1)

    data = {"Node Count": [3, 4],
            # "Multi-Hop with 6492Hz": [479.42, 539.14],
            "Multi-Hop with 24583Hz": [448.17, 624.63],
            "Multi-Hop with 63109Hz": [509.41, 664.88], }
    data_frame = pd.DataFrame(data)
    plot_grouped_bars(data_frame, "Node Count",
                      "Multi-Hop Transmission Time with Different Depolar Rate",
                      "Network Node Count",
                      "Transmission Time (ms)",
                      save_name="./figures/transmission_time_memory_comparison.png",
                      ncol=1)

    data = {"Node Count": [3, 4],
            # "Multi-Hop with 6492Hz": [479.42, 539.14],
            "Multi-Hop with 24583Hz": [0.79629, 0.92],
            "Multi-Hop with 63109Hz": [0.788, 0.91], }
    data_frame = pd.DataFrame(data)
    plot_grouped_bars(data_frame, "Node Count",
                      "Multi-Hop Success Rate with Different Depolar Rate",
                      "Network Node Count",
                      "Success Rate (Fidelity > 0.7)",
                      save_name="./figures/success_rate_memory_comparison.png",
                      ncol=1)
    # # E2E Delay vs HBH
    # data = {
    #     "Node Count": [3, 4],
    #     "End to End (Purification)": [16819.12, 24405.73],  # µs
    #     "Hop By Hop (Purification)": [11497.23, 14390.67],  # µs
    # }
    # data_frame = pd.DataFrame(data)
    # plot_grouped_bars(data_frame, "Node Count",
    #                   "Transmission Delay Comparison",
    #                   "Network Node Count",
    #                   "Transmission Time (µs)",
    #                   save_name="./figures/delay_comparison_e2e_p_vs_hbh_p.png")
    #
    # # E2E Th vs HBH
    # data = {
    #     "Node Count": [3, 4],
    #     "End to End (Purification) Total Count": [131.64, 120.33],
    #     "End to End (Purification) Fidelity > 0.7": [65.74, 60.21],
    #     "End to End (Purification) Fidelity > 0.9": [65.74, 60.21],
    #     "Hop By Hop (Purification) Total Count": [165.25, 162.16],
    #     "Hop By Hop (Purification) Fidelity > 0.7": [82.85,81.43],
    #     "Hop By Hop (Purification) Fidelity > 0.9": [82.85,81.43],
    # }
    # data_frame = pd.DataFrame(data)
    # plot_grouped_bars(data_frame, "Node Count",
    #                   "Throughput Comparison",
    #                   "Network Node Count",
    #                   "Transmitted Qubit Count",
    #                   save_name="./figures/throughput_comparison_e2e_p_vs_hbh_p.png")
    #
    #
    # # E2E V fid vs HBH V
    # data = {
    #     "Node Count": [3, 4],
    #     "End to End (Verification)": [0.6923, 0.6869],
    #     "Hop By Hop (Verification)": [0.7043, 0.7063],
    # }
    # data_frame = pd.DataFrame(data)
    # plot_grouped_bars(data_frame, "Node Count",
    #                   "Fidelity Comparison",
    #                   "Network Node Count",
    #                   "Fidelity",
    #                   save_name="./figures/fidelity_comparison_e2e_v_vs_hbh_v.png")
    # # E2E V  Delay vs HBH V
    # data = {
    #     "Node Count": [3, 4],
    #     "End to End (Verification)": [524998.81, 689674.57],  # µs
    #     "Hop By Hop (Verification)": [509411.18, 664875.80],  # µs
    # }
    # data_frame = pd.DataFrame(data)
    # plot_grouped_bars(data_frame, "Node Count",
    #                   "Transmission Delay Comparison",
    #                   "Network Node Count",
    #                   "Transmission Time (µs)",
    #                   save_name="./figures/delay_comparison_e2e_v_vs_hbh_v.png")
