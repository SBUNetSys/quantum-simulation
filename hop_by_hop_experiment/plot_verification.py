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
        if "Target" in legend or "Entanglement" in legend:
            ax.plot(x, y, label=f'{legend}', lw=2, linestyle="--")
        else:
            ax.plot(x, y, label=f'{legend}', lw=2)
        if y_point_labels:
            for i, txt in enumerate(y_point_labels):
                ax.text(list(x)[i], list(y)[i], txt, fontsize=12)

    ax.set_title(f'{title}')
    ax.set_xlabel(f'{x_label}')
    ax.set_ylabel(f'{y_label}')
    ax.tick_params(axis='x', labelrotation=90)
    ax.legend(loc="upper right",ncols=3)  # loc="upper left"bbox_to_anchor=(1.05, 1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # rotate x labels
    plt.xticks(rotation=0)
    # ax.set_xlim(left=0)
    if "Duration" not in title:
        ax.set_ylim(top=1.1)
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
    with open("verification_results/verification_result_2_nodes_1_paris_5.0km.json") as fin:
        data = json.load(fin)
    with open("verification_results/verification_result_2_nodes_1_paris_5.0km_no_verify.json") as fin:
        no_purify_data = json.load(fin)

    """
    "500": {
        "0.99": {
            "actual_fidelities": 0.48017536571918257,
            "total_verified_pairs": 4.0,
            "teleport_success_count": 0.051,
            "duration": 300662.415,
    }
    """
    x = data.keys()
    target_fids = []
    actual_fidelity = []
    experiment_duration = []
    teleportation_success = []
    for dis in x:
        target_fid = list(data[dis].keys())[0]
        target_fids.append(float(target_fid))
        actual_fidelity.append(data[dis][target_fid]["actual_fidelities"])
        experiment_duration.append(data[dis][target_fid]["duration"]/1e6)
        teleportation_success.append(data[dis][target_fid]["teleport_success_count"])

    with open("entanglement_results/entanglement_results_2nodes_5km.json") as fin:
        data = json.load(fin)
    entangle_fidelity = [data[key]["actual_fidelity"] for key in x]

    target_fids_no_purify = []
    actual_fidelity_no_purify = []
    experiment_duration_no_purify = []
    teleportation_success_no_purify = []
    for dis in x:
        target_fid = list(no_purify_data[dis].keys())[0]
        target_fids_no_purify.append(float(target_fid))
        actual_fidelity_no_purify.append(no_purify_data[dis][target_fid]["actual_fidelities"])
        experiment_duration_no_purify.append(float(no_purify_data[dis][target_fid]["duration"])/1e6)
        teleportation_success_no_purify.append(no_purify_data[dis][target_fid]["teleport_success_count"])

    x_dis = [int(i)/1000 for i in x ]

    plot_lines([x_dis, x_dis, x_dis ], [target_fids, actual_fidelity, teleportation_success],
               "Verification Fidelity vs Teleportation Success Rate Comparison",
               "Node Distance (km)",
               "Fidelity",
               ["Purify Target Fidelity", "Actual Fidelity", "Teleport Success Rate"],
               save_dir="./verification_results/figures")
    plot_lines([x_dis, x_dis, x_dis], [entangle_fidelity, actual_fidelity_no_purify,
                                       teleportation_success_no_purify],
               "Verification No Purify Fidelity vs Teleportation Success Rate Comparison",
               "Node Distance (km)",
               "Fidelity",
               ["Entanglement Fidelity", "Actual Fidelity", "Teleport Success Rate"],
               save_dir="./verification_results/figures")
    plot_lines([x_dis, x_dis], [experiment_duration, experiment_duration_no_purify],
               "Verification Experiment Duration Comparison",
               "Node Distance (km)",
               "Duration(ms)",
               ["Purify Duration","No Purify Duration"],
               save_dir="./verification_results/figures")
    # plot_lines([x_dis, x_dis,], [target_fids, actual_fidelity],
    #            "Verification Fidelity",
    #            "Node Distance (km)",
    #            "Fidelity",
    #            ["Purify Target Fidelity", "Actual Fidelity"],
    #            save_dir="./verification_results/figures")
    #
    # plot_lines([x_dis], [teleportation_success],
    #            "Verification Teleport Success Rate",
    #            "Node Distance (km)",
    #            "Success Rate",
    #            ["Teleport Success Rate"],
    #            save_dir="./verification_results/figures")

    # plot_lines([x_dis], [experiment_duration], "Experiment Duration vs Distance",
    #            "Node Distance (km)",
    #            "Experiment Duration (ms)",
    #            ["Experiment Duration"],
    #            save_dir="./entanglement_results/figures")


if __name__ == '__main__':
    main()
