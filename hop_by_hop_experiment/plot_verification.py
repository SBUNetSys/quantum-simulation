import json
import os

import numpy as np
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
        if "Target" in legend:
            ax.plot(x, y, label=f'{legend}', lw=2, linestyle="--")
        elif "Entanglement" in legend:
            ax.plot(x, y, label=f'{legend}', lw=2, linestyle=":")
        else:
            ax.plot(x, y, label=f'{legend}', lw=2)
        if y_point_labels:
            for i, txt in enumerate(y_point_labels):
                ax.text(list(x)[i], list(y)[i], txt, fontsize=12)

    ax.set_title(f'{title}')
    ax.set_xlabel(f'{x_label}')
    ax.set_ylabel(f'{y_label}')
    ax.tick_params(axis='x', labelrotation=90)
    ax.legend(loc="upper right", ncols=3, fontsize=12)  # loc="upper left"bbox_to_anchor=(1.05, 1)
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


def plot_non_concurrent():
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
        experiment_duration.append(data[dis][target_fid]["duration"] / 1e6)
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
        experiment_duration_no_purify.append(float(no_purify_data[dis][target_fid]["duration"]) / 1e6)
        teleportation_success_no_purify.append(no_purify_data[dis][target_fid]["teleport_success_count"])

    x_dis = [int(i) / 1000 for i in x]

    plot_lines([x_dis, x_dis, x_dis], [target_fids, actual_fidelity, teleportation_success],
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
               ["Purify Duration", "No Purify Duration"],
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


def plot_cdf(data, title, x_label, xlim=None, save=True):
    x = np.sort(data)
    y = np.arange(len(x)) / float(len(x))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y)
    ax.set_title(f'{title}')
    ax.set_xlabel(f'{x_label}')
    ax.set_ylabel('CDF')
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    # ax.tick_params(axis='x', labelrotation=70)
    # ax.legend(loc="upper left")
    plt.tight_layout()
    if save:
        plt.savefig(f'./figures/{title}.png')
    plt.show()


def plot_multi_cdf(datas,
                   title, x_label, data_legends,
                   xlim=None, save=True, save_dir="./"):
    xs = [np.sort(x) for x in datas]
    ys = [np.arange(len(x)) / float(len(x)) for x in xs]

    fig, ax = plt.subplots(figsize=(12, 6))
    for x, y, legend in zip(xs, ys, data_legends):
        ax.plot(x, y, label=f'{legend}', lw=1.5)
    ax.set_title(f'{title}')
    ax.set_xlabel(f'{x_label}')
    # ax.set_xscale('log')
    # ax.axvspan(0, 1, color='lightgreen', alpha=0.5, label="No Stall Zone")
    # ax.axvspan(1, max(max(x), max(x2)), color='lightcoral', alpha=0.5, label="Stall Zone")
    ax.set_ylabel('CDF')
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    # ax.tick_params(axis='x', labelrotation=70)
    ax.legend(loc="best", ncols=5, fontsize=12)  # loc="upper left"
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_dir, f'{title}.png'))  # bbox_inches='tight'
    plt.show()


def plot_concurrent_data():
    # load the data
    with open("verification_results/concurrent_verification_result_2_nodes_1_paris_5.0km.json") as fin:
        data = json.load(fin)
    with open("verification_results/concurrent_verification_result_2_nodes_1_paris_5.0km_no_verify.json") as fin:
        no_purify_data = json.load(fin)

    """
     "500": {
        "0.99": {
            "actual_fidelities": 0.5242126832710813,
            "total_verified_pairs": 4.0,
            "success_verification_probability":[],
            "total_verification_count": 7.993,
            "success_verification_count": 1.0,
            "teleport_success_count": 0.332,
            "teleport_fids": NaN,
            "duration": 131206.0,
            "raw_data": {
    """
    # distance
    all_dis = data.keys()
    purify_target = []
    purify_fid = []
    purify_teleport_fid = []
    purify_all_fids_data = {}
    purify_all_teleport_data = {}
    purify_durations = []
    for dis, dis_data in data.items():
        for key, value in dis_data.items():
            purify_target.append(float(key))
            all_fids = [x for x in value['raw_data']['actual_fidelities'] if not (isinstance(x, float) and np.isnan(x))]
            purify_all_fids_data[dis] = all_fids
            purify_fid.append(np.mean(all_fids))
            purify_all_teleport_data[dis] = [x for x in value['raw_data']['teleport_fids'] if
                                             not (isinstance(x, float) and np.isnan(x))]
            purify_durations.append(value['duration'] / 1e6)
            purify_teleport_fid.append(np.mean(purify_all_teleport_data[dis]))
    plot_multi_cdf(list(purify_all_fids_data.values()),
                   "Concurrent Verification Actual Fidelities CDF with Purification",
                   "Fidelity",
                   [f'{int(x) / 1000} Km' for x in purify_all_fids_data.keys()],
                   save=True,
                   save_dir="./verification_results/figures"
                   )
    plot_multi_cdf(list(purify_all_teleport_data.values()),
                   "Concurrent Verification Teleported Fidelities CDF with Purification",
                   "Fidelity",
                   [f'{int(x) / 1000} Km' for x in purify_all_teleport_data.keys()],
                   save=True,
                   save_dir="./verification_results/figures"
                   )

    no_purify_all_fids_data = {}
    no_purify_all_teleport_data = {}
    no_purify_durations = []
    no_purify_fid = []
    no_purify_teleport_fid = []
    for dis, dis_data in no_purify_data.items():
        for key, value in dis_data.items():
            all_fids = [x for x in value['raw_data']['actual_fidelities'] if not (isinstance(x, float) and np.isnan(x))]
            no_purify_all_fids_data[dis] = all_fids
            no_purify_fid.append(np.mean(all_fids))
            no_purify_all_teleport_data[dis] = [x for x in value['raw_data']['teleport_fids'] if
                                                not (isinstance(x, float) and np.isnan(x))]
            no_purify_durations.append(value['duration'] / 1e6)
            no_purify_teleport_fid.append(np.mean(no_purify_all_teleport_data[dis]))

    plot_multi_cdf(list(no_purify_all_fids_data.values()),
                   "Concurrent Verification Actual Fidelities CDF with No Purification",
                   "Fidelity",
                   [f'{int(x) / 1000} Km' for x in no_purify_all_fids_data.keys()],
                   save=True,
                   save_dir="./verification_results/figures"
                   )
    plot_multi_cdf(list(no_purify_all_teleport_data.values()),
                   "Concurrent Verification Teleported Fidelities CDF with No Purification",
                   "Fidelity",
                   [f'{int(x) / 1000} Km' for x in no_purify_all_teleport_data.keys()],
                   save=True,
                   save_dir="./verification_results/figures"
                   )
    dis_x = [int(x) / 1000 for x in all_dis]
    with open("entanglement_results/entanglement_results_2nodes_5km.json") as fin:
        data = json.load(fin)
    entangle_fidelity = [data[key]["actual_fidelity"] for key in all_dis]

    plot_lines([dis_x] * 6,
               [purify_fid, purify_teleport_fid,
                no_purify_fid,
                no_purify_teleport_fid,
                entangle_fidelity, purify_target],
               "Concurrent Verification Fidelities Comparison with and without Purification",
               "Node Distance (km)",
               "Fidelity",
               ["Fidelity with Purify",
                "Teleported Fidelity with Purify", "Fidelity without Purify",
                "Teleported Fidelity without Purify", "Entanglement Fidelity",
                "Purify Target Fidelity"],
               save=True,
               save_dir="./verification_results/figures")
    plot_lines([dis_x] * 2,
               [purify_durations, no_purify_durations],
               "Concurrent Verification Duration Comparison with and without Purification",
               "Node Distance (km)",
               "Duration (ms)",
               ["With Purify", "Without Purify"],
               save=True,
               save_dir="./verification_results/figures")


def main():
    # plot_non_concurrent()
    plot_concurrent_data()


if __name__ == '__main__':
    main()
