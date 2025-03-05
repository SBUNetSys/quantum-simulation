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
        if "No Purify" in legend:
            ax.plot(x, y, label=f'{legend}', lw=1, linestyle="--")
        else:
            ax.plot(x, y, label=f'{legend}', lw=1)
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


def load_data(file_path):
    with open(file_path) as fin:
        data = json.load(fin)

    """
    "500": {
        "0.989": {
            "actual_fidelity": 0.79,
            "estimated_fidelity": 0.9891421077254133,
            "purified_count": 1.383,
            "purified_success_count": 1.0,
            "experiment_duration": 92817.377,
            "satisfied_pairs_count": 1.0,
            "teleport_success_count": 0.803,
    """
    x = data.keys()
    purify_one_actual_fed = []
    purify_one_estimated_fed = []
    purify_one_experiment_duration = []
    purify_one_teleport_success = []
    purify_one_target_fidelity = []

    purify_two_actual_fed = []
    purify_two_estimated_fed = []
    purify_two_experiment_duration = []
    purify_two_teleport_success = []
    purify_two_target_fidelity = []

    for key in x:
        p1 = list(data[key].values())[0]
        p2 = list(data[key].values())[1]
        purify_one_actual_fed.append(p1["actual_fidelity"])
        purify_one_estimated_fed.append(p1["estimated_fidelity"])
        purify_one_experiment_duration.append(p1["experiment_duration"] / 1e6)
        purify_one_teleport_success.append(p1["teleport_success_count"])
        purify_one_target_fidelity.append(list(data[key].keys())[0])

        purify_two_actual_fed.append(p2["actual_fidelity"])
        purify_two_estimated_fed.append(p2["estimated_fidelity"])
        purify_two_experiment_duration.append(p2["experiment_duration"] / 1e6)
        purify_two_teleport_success.append(p2["teleport_success_count"])
        purify_two_target_fidelity.append(list(data[key].keys())[1])
    return (x,
            purify_one_actual_fed,
            purify_one_estimated_fed,
            purify_one_experiment_duration,
            purify_one_teleport_success,
            purify_one_target_fidelity,
            purify_two_actual_fed,
            purify_two_estimated_fed,
            purify_two_experiment_duration,
            purify_two_teleport_success,
            purify_two_target_fidelity,)


def plot_purify_data():
    (x,
     purify_one_actual_fed,
     purify_one_estimated_fed,
     purify_one_experiment_duration,
     purify_one_teleport_success,
     purify_one_target_fidelity,
     purify_two_actual_fed,
     purify_two_estimated_fed,
     purify_two_experiment_duration,
     purify_two_teleport_success,
     purify_two_target_fidelity,) = load_data("purification_results/purification_results_2_nodes_1_paris_5km.json")

    with open("entanglement_results/entanglement_results_2nodes_5km.json") as fin:
        data = json.load(fin)
    actual_fidelity = [data[key]["actual_fidelity"] for key in x]
    teleportation_success = [data[key]["average_teleportation_success"] for key in x]
    experiment_duration = [data[key]["average_duration"] / 1e6 for key in x]

    x_dis = [int(i) / 1000 for i in x]

    plot_lines([x_dis, x_dis, x_dis, x_dis, x_dis],
               [actual_fidelity,
                purify_one_estimated_fed,
                purify_two_estimated_fed,
                purify_one_actual_fed,
                purify_two_actual_fed],
               "Purification Fidelity Comparison with Distance",
               "Node Distance (km)",
               "Fidelity",
               ["No Purify Actual Fidelity",
                "1 Level Purify Estimated Fidelity",
                "2 Level Purify Estimated Fidelity",
                "1 Level Actual Fidelity",
                "2 Level Actual Fidelity", ],
               save_dir="./purification_results/figures", )

    plot_lines([x_dis, x_dis, x_dis],
               [teleportation_success,
                purify_one_teleport_success,
                purify_two_teleport_success, ],
               "Purification Teleport Success Comparison with Distance",
               "Node Distance (km)",
               "Success Rate",
               ["No Purify",
                "1 Level Purify",
                "2 Level Purify", ],
               save_dir="./purification_results/figures", )

    plot_lines([x_dis, x_dis, x_dis],
               [experiment_duration,
                purify_one_experiment_duration,
                purify_two_experiment_duration, ],
               "Purification Duration Comparison with Distance",
               "Node Distance (km)",
               "Duration (ms)",
               ["No Purify",
                "1 Level Purify",
                "2 Level Purify", ],
               save_dir="./purification_results/figures", )


def plot_purify_delay_comparison():
    (x,
     purify_one_actual_fed,
     purify_one_estimated_fed,
     purify_one_experiment_duration,
     purify_one_teleport_success,
     purify_one_target_fidelity,
     purify_two_actual_fed,
     purify_two_estimated_fed,
     purify_two_experiment_duration,
     purify_two_teleport_success,
     purify_two_target_fidelity,) = load_data(
        "purification_results/purification_results_2_nodes_1_paris_5km.json")

    (x,
     purify_one_actual_fed_delay,
     purify_one_estimated_fed_delay,
     purify_one_experiment_duration_delay,
     purify_one_teleport_success_delay,
     purify_one_target_fidelity_delay,
     purify_two_actual_fed_delay,
     purify_two_estimated_fed_delay,
     purify_two_experiment_duration_delay,
     purify_two_teleport_success_delay,
     purify_two_target_fidelity_delay,) = load_data(
        "purification_results/purification_results_2_nodes_1_paris_5km_delayed.json")

    (x,
     concurrent_purify_one_actual_fed,
     concurrent_purify_one_estimated_fed,
     concurrent_purify_one_experiment_duration,
     concurrent_purify_one_teleport_success,
     concurrent_purify_one_target_fidelity,
     concurrent_purify_two_actual_fed,
     concurrent_purify_two_estimated_fed,
     concurrent_purify_two_experiment_duration,
     concurrent_purify_two_teleport_success,
     concurrent_purify_two_target_fidelity,) = load_data(
        "purification_results/concurrent_purification_results_2_nodes_1_paris_5.0km.json")


    with open("entanglement_results/entanglement_results_2nodes_5km.json") as fin:
        data = json.load(fin)
    actual_fidelity = [data[key]["actual_fidelity"] for key in x]

    x_dis = [int(i) / 1000 for i in x]

    # plot_lines([x_dis, x_dis, x_dis, x_dis, x_dis, x_dis, x_dis],
    #            [
    #                actual_fidelity,
    #                purify_one_estimated_fed,
    #                purify_two_estimated_fed,
    #                purify_one_actual_fed,
    #                purify_one_actual_fed_delay,
    #                purify_two_actual_fed,
    #                purify_two_actual_fed_delay, ],
    #            "Purification Fidelity Comparison Delayed vs Non-Delayed",
    #            "Node Distance (km)",
    #            "Fidelity",
    #            [
    #                "No Purify Actual Fidelity",
    #                "1 Level Purify Estimated Fidelity",
    #                "2 Level Purify Estimated Fidelity",
    #                "1 Level No Delay Actual Fidelity",
    #                "1 Level Delayed Actual Fidelity",
    #                "2 Level No Delay Actual Fidelity",
    #                "2 Level Delayed Actual Fidelity",
    #            ],
    #            save_dir="./purification_results/figures", )
    #
    # plot_lines([x_dis, x_dis, x_dis, x_dis],
    #            [
    #                purify_one_experiment_duration,
    #                purify_one_experiment_duration_delay,
    #                purify_two_experiment_duration,
    #                purify_two_experiment_duration_delay],
    #            "Purification Duration Comparison Delayed vs Non-Delayed",
    #            "Node Distance (km)",
    #            "Duration (ms)",
    #            [
    #                "1 Level No Delay Purify",
    #                "1 Level Delayed Purify",
    #                "2 Level No DelayPurify",
    #                "2 Level Delayed Purify",
    #            ],
    #            save_dir="./purification_results/figures", )

    plot_lines([x_dis, x_dis, x_dis, x_dis, x_dis, x_dis, x_dis],
               [
                   actual_fidelity,
                   purify_one_estimated_fed,
                   purify_two_estimated_fed,
                   purify_one_actual_fed_delay,
                   concurrent_purify_one_actual_fed,
                   purify_two_actual_fed_delay,
                   concurrent_purify_two_actual_fed, ],
               "Purification Fidelity Comparison Concurrent vs Non-Concurrent",
               "Node Distance (km)",
               "Fidelity",
               [
                   "No Purify Actual Fidelity",
                   "1 Level Purify Estimated Fidelity",
                   "2 Level Purify Estimated Fidelity",
                   "1 Level Concurrent Actual Fidelity",
                   "1 Level Delayed Actual Fidelity",
                   "2 Level Concurrent Actual Fidelity",
                   "2 Level Delayed Actual Fidelity",
               ],
               save_dir="./purification_results/figures", )
    plot_lines([x_dis, x_dis, x_dis, x_dis],
               [
                   purify_one_experiment_duration_delay,
                   concurrent_purify_one_experiment_duration,
                   purify_two_experiment_duration_delay,
                   concurrent_purify_two_experiment_duration],
               "Purification Duration Comparison Concurrent vs Non-Concurrent",
               "Node Distance (km)",
               "Duration (ms)",
               [
                   "1 Level Purify",
                   "1 Level Concurrent Purify",
                   "2 Level Purify",
                   "2 Level Concurrent Purify",
               ],
               save_dir="./purification_results/figures", )


def main():
    plot_purify_delay_comparison()


if __name__ == '__main__':
    main()
