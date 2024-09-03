import math
import os
import json

import numpy as np
from fontTools.varLib.models import subList
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


def plot_lines_with_confidence_interval(xs, ys, title, x_label,
                                        y_label, data_legends,
                                        xlim=None, save=True, save_dir="./",
                                        num_bins=20,
                                        y_point_labels=None):
    # calculate means and standard deviations
    # means = [np.mean(y, axis=1) for y in ys]
    # stds = [np.std(y, axis=1) for y in ys]

    # fig, ax = plt.subplots(figsize=(12, 6))
    # for x, y, legend, std, in zip(xs, means, data_legends, stds):
    #     ax.plot(x, y, label=f'{legend}', lw=2)
    #     ax.fill_between(x, y - std, y + std, alpha=0.2)
    #     ax.errorbar(x, y, yerr=std, fmt='o', label=f'{legend}', lw=2)
    #     if y_point_labels:
    #         for i, txt in enumerate(y_point_labels):
    #             ax.text(list(x)[i], list(y)[i], txt, fontsize=12)
    fig, ax = plt.subplots(figsize=(12, 6))
    for x, y, legend in zip(xs, ys, data_legends):
        og_list = y
        y = np.array([np.median(sublist) for sublist in og_list])
        # std = np.array([np.std(sublist) for sublist in og_list])
        p5 = np.array([np.percentile(sublist, 5) for sublist in og_list])
        p95 = np.array([np.percentile(sublist, 95) for sublist in og_list])
        ax.plot(x, y, label=f'{legend}', lw=2)
        ax.fill_between(x, p5, p95, alpha=0.5)
        # ax.errorbar(x, y, yerr=std, fmt='o', label=f'{legend}', lw=2)
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
    """
    Plot the experiment data
    """
    # plot general fidelity rate compre between different algothrims
    # gen-entanglement result
    with open("./entanglement_results/entanglement_results_2_node_128_pairs_noise_True.json", "r") as fin:
        no_noise_entanglement = json.load(fin)
    with open("./entanglement_results/entanglement_results_2_node_128_pairs_noise_False.json", "r") as fin:
        noise_entanglement = json.load(fin)
    # plot noise vs non-noise
    entanglement_entangled_pairs = [int(key) for key in noise_entanglement.keys()]
    entanglement_noise_actual_fid = []
    entanglement_noise_success_teleport = []
    entanglement_no_noise_actual_fid = []
    entanglement_no_noise_success_teleport = []
    entanglement_est_fid = []
    """
    {
        "actual_fidelity": 0.9859999999999999,
        "estimated_fidelity": 0.9925373753118545,
        "average_duration": 100000.0,
        "average_teleportation_success": 0.993
    }
    """
    for no_noise_data, noise_data in zip(no_noise_entanglement.values(), noise_entanglement.values()):
        entanglement_no_noise_actual_fid.append(no_noise_data["actual_fidelity"])
        entanglement_no_noise_success_teleport.append(no_noise_data["average_teleportation_success"])
        entanglement_est_fid.append(no_noise_data["estimated_fidelity"])
        entanglement_noise_actual_fid.append(noise_data["actual_fidelity"])
        entanglement_noise_success_teleport.append(noise_data["average_teleportation_success"])
    plot_lines([entanglement_entangled_pairs, entanglement_entangled_pairs, entanglement_entangled_pairs],
               [entanglement_no_noise_actual_fid, entanglement_noise_actual_fid, entanglement_est_fid],
               "Gen-Entanglement Fidelity Comparison with Estimated Fidelity",
               "Entangled Pairs",
               "Fidelity",
               ["No Pop Noise", "With Pop Noise", "Estimated Fidelity"], save=True, save_dir="./figures")
    # plot gen-entanglement teleport rate
    plot_lines([entanglement_entangled_pairs, entanglement_entangled_pairs, entanglement_entangled_pairs],
               [entanglement_entangled_pairs, entanglement_no_noise_success_teleport,
                entanglement_noise_success_teleport],
               "Gen-Entanglement Teleportation Success Rate",
               "Entangle Pairs",
               "Teleportation Success Count",
               ["Theoretical", "No Pop Noise", "With Pop Noise"],
               save=True, save_dir="./figures")
    # load purification data
    # gen-entanglement result
    with open("./purification_results/purification_results_2_nodes_128_paris_noise_True.json", "r") as fin:
        no_noise_purification = json.load(fin)
    with open("./purification_results/purification_results_2_nodes_128_paris_noise_False.json", "r") as fin:
        noise_purification = json.load(fin)

    purify_entangled_pairs = []
    purify_noise_actual_fid = []
    purify_noise_success_teleport = []
    purify_no_noise_actual_fid = []
    purify_no_noise_success_teleport = []
    purify_est_fid = []
    """
    {
        "actual_fidelity": 0.819,
        "estimated_fidelity": 0.9966416586149264,
        "purified_count": 4.357,
        "purified_success_count": 4.136,
        "experiment_duration": 0.0014109,
        "satisfied_pairs_count": 1.0,
        "teleport_success_count": 0.845,
    }
    """
    for no_noise_data, noise_data in zip(no_noise_purification.values(), noise_purification.values()):
        purify_entangled_pairs.append(int(no_noise_data["satisfied_pairs_count"]))
        purify_no_noise_actual_fid.append(no_noise_data["actual_fidelity"])
        purify_no_noise_success_teleport.append(no_noise_data["teleport_success_count"])
        purify_est_fid.append(no_noise_data["estimated_fidelity"])
        purify_noise_actual_fid.append(noise_data["actual_fidelity"])
        purify_noise_success_teleport.append(noise_data["teleport_success_count"])
    plot_lines([purify_entangled_pairs, purify_entangled_pairs, purify_entangled_pairs],
               [purify_no_noise_actual_fid, purify_noise_actual_fid, purify_est_fid],
               "Purification Fidelity Comparison with Estimated Fidelity",
               "Success Purified Pairs",
               "Fidelity",
               ["No Pop Noise", "With Pop Noise", "Estimated Fidelity"], save=True, save_dir="./figures")
    # plot gen-purification teleport rate
    plot_lines([purify_entangled_pairs, purify_entangled_pairs, purify_entangled_pairs],
               [purify_entangled_pairs, purify_no_noise_success_teleport, purify_noise_success_teleport],
               "Purification Teleportation Success Rate",
               "Success Purified Pairs",
               "Teleportation Success Count",
               ["Theoretical", "No Pop Noise", "With Pop Noise"],
               save=True, save_dir="./figures")
    # load verification data
    with open("./verification_results/1000runs/batch_data_2_nodes_max_8_batch_size.json", "r") as fin:
        no_noise_verification_data = json.load(fin)
    with open("./verification_results/batch_data_2_nodes_max_8_batch_size.json", "r") as fin:
        no_noise_verification_data = json.load(fin)
    # TODO: add noise verification data
    verification_entangled_pairs = {}  # key: batch_size, value: list of entangled pairs
    verification_no_noise_actual_fid = {}  # key: batch_size, value: list of actual fidelity
    verification_no_noise_success_teleport = {}  # key: batch_size, value: list of teleport success
    verification_acceptance_rate = {}  # key: batch_size, value: list of acceptance rate
    verification_count = {}  # key: batch_size, value: list of verification count
    verification_success_count = {}  # key: batch_size, value: list of success verification count
    """
    {"2": {
        "8": {"actual_fidelities": 0.8602261453423939, 
              "total_verified_pairs": 2.0, 
              "success_verification_probability": [], 
              "total_verification_count": 2.972, 
              "success_verification_count": 1.0, 
              "teleport_success_count": 1.516,
            }
    """
    # TODO: we need experiment duration
    for batch_size, batch_data in no_noise_verification_data.items():
        verification_entangled_pairs[batch_size] = []
        verification_no_noise_actual_fid[batch_size] = []
        verification_no_noise_success_teleport[batch_size] = []
        verification_acceptance_rate[batch_size] = []
        verification_count[batch_size] = []
        verification_success_count[batch_size] = []
        for data in batch_data.values():
            if math.isnan(data["total_verified_pairs"]):
                continue
            verification_entangled_pairs[batch_size].append(int(data["total_verified_pairs"]))
            verification_no_noise_actual_fid[batch_size].append(data["actual_fidelities"])
            verification_no_noise_success_teleport[batch_size].append(data["teleport_success_count"])
            verification_acceptance_rate[batch_size].append(data["success_verification_probability"])
            verification_count[batch_size].append(data["total_verification_count"])
            verification_success_count[batch_size].append(data["success_verification_count"])
    # plot verification data, with all batches in one plot
    plot_lines([list(x) for x in verification_entangled_pairs.values()],
               [list(y) for y in verification_no_noise_actual_fid.values()],
               "Verification Fidelity Comparison with Different Batch Size",
               "Success Verified Pairs",
               "Fidelity",
               [f"BatchSize_{x}" for x in verification_entangled_pairs.keys()], save=True, save_dir="./figures")
    # plot verification teleport rate
    xs = [list(x) for x in verification_entangled_pairs.values()]
    ys = [list(y) for y in verification_no_noise_success_teleport.values()]
    theory_y = []
    for x in xs:
        theory_y.extend(x)
    theory_y = list(set(theory_y))
    ys.append(theory_y)
    xs.append(theory_y)
    legends = [f"BatchSize_{x}" for x in verification_entangled_pairs.keys()]
    legends.append(f"Theoretical")
    plot_lines(xs,
               ys,
               "Verification Teleportation Success Rate with Different Batch Size",
               "Success Verified Pairs",
               "Teleportation Success Count",
               legends, save=True, save_dir="./figures")
    # plot verification count and success count
    xs = [list(x) for x in verification_entangled_pairs.values()]
    xs += xs
    ys = [list(y) for y in verification_count.values()]
    ys += [list(y) for y in verification_success_count.values()]
    legends = [f"BatchSize_{x}_Total Count" for x in verification_entangled_pairs.keys()]
    legends += [f"BatchSize_{x}_Success Count" for x in verification_entangled_pairs.keys()]
    plot_lines(xs,
               ys,
               "Verification Count with Different Batch Size",
               "Success Verified Pairs",
               "Verification Count",
               legends, save=True, save_dir="./figures")
    # plot verification acceptance rate with confidence interval
    xs = [list(x) for x in verification_entangled_pairs.values()]
    ys = [y for y in verification_acceptance_rate.values()]
    plot_lines_with_confidence_interval(xs,
                                        ys,
                                        "Verification Acceptance Rate with Different Batch Size",
                                        "Success Verified Pairs",
                                        "Acceptance Rate",
                                        [f"BatchSize_{x}" for x in verification_entangled_pairs.keys()],
                                        save=True, save_dir="./figures")
    # plot average verification verification acceptance rate
    xs = [list(x) for x in verification_entangled_pairs.values()]
    ys = [y for y in verification_acceptance_rate.values()]
    ys = [[np.mean(sublist) for sublist in y] for y in ys]
    plot_lines(xs,
               ys,
               "Average Verification Acceptance Rate with Different Batch Size",
               "Success Verified Pairs",
               "Acceptance Rate",
               [f"BatchSize_{x}" for x in verification_entangled_pairs.keys()],
               save=True, save_dir="./figures")

if __name__ == '__main__':
    main()
