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
    with open("purification_results/purification_results_2_nodes_128_paris.json") as fin:
        data = json.load(fin)
    """
    "5": {"actual_fidelity": 0.873, 
    "estimated_fidelity": 0.9966416586149264, 
    "purified_count": 3.42, 
    "purified_success_count": 3.182, 
    "experiment_duration": 0.001331604214}
    """
    x = data.keys()
    actual_fidelity = [data[key]["actual_fidelity"] for key in x]
    estimated_fidelity = [data[key]["estimated_fidelity"] for key in x]
    purified_count = [data[key]["purified_count"] for key in x]
    purified_success_count = [data[key]["purified_success_count"] for key in x]
    experiment_duration = [data[key]["experiment_duration"] for key in x]
    satisfied_pairs = [data[key]["satisfied_pairs_count"] for key in x]
    teleportation_success = [data[key]["teleport_success_count"] for key in x]

    plot_lines([x, x], [actual_fidelity, estimated_fidelity], "Fidelity vs Pair",
               "Numb of Entangled Pairs",
               "Fidelity",
               ["Actual Fidelity", "Estimated Fidelity"],
               save_dir="./purification_results/figures", y_point_labels=satisfied_pairs)
    plot_lines([x, x], [satisfied_pairs, teleportation_success], "Satisfied Pairs vs Success Teleportation",
               "Numb of Entangled Pairs",
               "Count",
               ["Satisfied Pairs", "Teleportation Success"],
               save_dir="./purification_results/figures")
    plot_lines([x, x], [purified_count, purified_success_count], "Purified Count vs Pair",
               "Numb of Entangled Pairs",
               "Purified Count",
               ["Purified Count", "Purified Success Count"],
               save_dir="./purification_results/figures")
    plot_lines([x], [experiment_duration], "Experiment Duration vs Pair",
               "Numb of Entangled Pairs",
               "Experiment Duration (s)",
               ["Experiment Duration"],
               save_dir="./purification_results/figures")


if __name__ == '__main__':
    main()
