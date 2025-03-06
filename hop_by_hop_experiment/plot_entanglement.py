import json
import os

from matplotlib import pyplot as plt

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14


def plot_lines(xs, ys, title, x_label, y_label, data_legends, xlim=None, save=True, save_dir="./", num_bins=20,
               y_point_labels=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    for x, y, legend in zip(xs, ys, data_legends):
        ax.plot(x, y, label=f'{legend}', lw=2)
        if y_point_labels:
            for i, txt in enumerate(y_point_labels):
                ax.text(list(x)[i], list(y)[i], txt, fontsize=12)

    ax.set_title(f'{title}')
    ax.set_xlabel(f'{x_label}')
    ax.set_ylabel(f'{y_label}')
    ax.tick_params(axis='x', labelrotation=90)
    ax.legend(loc="best")  # loc="upper left"bbox_to_anchor=(1.05, 1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # rotate x labels
    plt.xticks(rotation=0)
    # ax.set_xlim(left=0)
    # ax.set_ylim(top=1)
    if xlim:
        ax.set_xlim(xlim)
    # Group x labels
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=20))
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))  # Format the labels as integers
    plt.tight_layout()
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"{title}.png"))
    plt.show()


def main():
    # load the data
    with open("entanglement_results/entanglement_results_2nodes_5km.json") as fin:
        data = json.load(fin)
    """
    "500": {
        "actual_fidelity": 0.9819999999999999,
        "estimated_fidelity": 0.9839718721292161,
        "average_duration": 37011.934,
        "average_teleportation_success": 0.987
    }
    """
    x = data.keys()
    actual_fidelity = [data[key]["actual_fidelity"] for key in x]
    estimated_fidelity = [data[key]["estimated_fidelity"] for key in x]
    experiment_duration = [data[key]["average_duration"]/1e6 for key in x]
    teleportation_success = [data[key]["average_teleportation_success"] for key in x]
    x_dis = [int(i)/1000 for i in x ]
    plot_lines([x_dis, x_dis], [actual_fidelity, estimated_fidelity], "Entanglement Fidelity Comparison 2 Nodes",
               "Node Distance (km)",
               "Fidelity",
               ["Actual Fidelity", "Estimated Fidelity"],
               save_dir="./entanglement_results/figures",)
    plot_lines([x_dis, x_dis], [actual_fidelity, teleportation_success],
               "Fidelity vs Teleportation Success Rate Comparison 2 Nodes",
               "Node Distance (km)",
               "Fidelity / Success Count Count",
               ["Actual Fidelity", "Teleportation Success"],
               save_dir="./entanglement_results/figures")

    plot_lines([x_dis], [experiment_duration], "Experiment Duration vs Distance",
               "Node Distance (km)",
               "Experiment Duration (ms)",
               ["Experiment Duration"],
               save_dir="./entanglement_results/figures")


if __name__ == '__main__':
    main()
