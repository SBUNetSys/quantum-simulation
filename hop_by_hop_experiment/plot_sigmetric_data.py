import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import os


# plt.rcParams['axes.labelsize'] = 18
# plt.rcParams['axes.titlesize'] = 18
# plt.rcParams['xtick.labelsize'] = 16
# plt.rcParams['ytick.labelsize'] = 16
# plt.rcParams['legend.fontsize'] = 16

width = 7
height = width * (np.sqrt(5) - 1.0) / 2.5
plt.rcParams['figure.figsize'] = (width, height)
plt.rcParams['font.size'] = 16
plt.rcParams['axes.axisbelow'] = True

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
        os.makedirs('./sigmetrics_goodput', exist_ok=True)
        plt.savefig(f'./sigmetrics_goodput/{title}.png')
    plt.show()

def plot_lines(xs, ys, title, x_label, y_label, data_legends, xlim=None, save=True, save_dir="./", num_bins=20,
               y_point_labels=None, ylim=None):
    fig, ax = plt.subplots(figsize=(10,5))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+', 'x']
    markers = markers * (len(xs) // len(markers) + 1)  # Repeat markers if needed
    for x, y, legend, mark in zip(xs, ys, data_legends, markers):
        ax.plot(x, y, label=f'{legend}', lw=2, marker=mark)
        if y_point_labels:
            for i, txt in enumerate(y_point_labels):
                ax.text(list(x)[i], list(y)[i], txt, fontsize=12)

    # ax.set_title(f'{title}')
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
    if ylim:
        ax.set_ylim(ylim)
    # Group x labels
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=20))
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))  # Format the labels as integers
    plt.tight_layout()
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"{title}.png"), dpi=300)
    plt.show()

def plot_bars(xs, ys, title, x_label, y_label, data_legends, xlim=None, save=True, save_dir="./", num_bins=20,
          y_point_labels=None):
    """
    Plot grouped bar chart.
    xs: list of x positions (categories) for each group
    ys: list of y values for each group
    data_legends: list of legend labels for each group
    """
    fig, ax = plt.subplots(figsize=(10,5))
    n_groups = len(xs[0])
    n_bars = len(xs)
    bar_width = 0.8 / n_bars
    indices = np.arange(n_groups)

    for i, (x, y, legend) in enumerate(zip(xs, ys, data_legends)):
        bars = ax.bar(indices + i * bar_width, y, bar_width, label=legend)
        # add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height * 1.01,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=12
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_title(title)
    ax.set_xticks(indices + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(xs[0])
    ax.legend(loc='upper left', ncol=3, bbox_to_anchor=(0, 1.15), fontsize=14)
    if xlim:
        ax.set_xlim(xlim)
    plt.tight_layout()
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"{title}.png"), dpi=300)
    plt.show()

def load_e2e_results(results_dir):
    """Load end-to-end experiment results"""
    results = {}
    pattern = r'e2e_(\d+)nodes_([\d.]+)m_1_qubit_([\d.]+)Hz_purification_(\w+)_verification_(\w+)'

    for file in Path(results_dir).glob('e2e_*.json'):
        if 'raw' in file.name:
            continue
        match = re.search(pattern, file.name)
        if match:
            nodes = int(match.group(1))
            distance = float(match.group(2))
            rate = float(match.group(3))
            purify = match.group(4) == 'True'
            verify = match.group(5) == 'True'
            goodput = 0
            raw_file = file.parent / file.name.replace(".json", "_raw.json")
            with open(raw_file) as f:
                data = json.load(f)
                for fid in data["teleport_fids"]:
                    if fid is not None and not np.isnan(fid) and fid >= 0.7:
                        goodput += 1
                goodput_ratio = goodput / len(data["teleport_fids"]) if len(data["teleport_fids"]) > 0 else 0
            with open(file) as f:
                data = json.load(f)
                key = (nodes, distance, rate, purify, verify)
                data["goodput"] = goodput
                data["goodput_ratio"] = goodput_ratio
                results[key] = data

    return results


def load_hopbyhop_results(results_dir):
    """Load hop-by-hop experiment results"""
    results = {}
    pattern = r'concurrent_transport_(\d+)_nodes_([\d.]+)km@([\d.]+)hz_purify_(\w+)_([\d.]+)_verify_(\w+)_(\d+)_qubit'

    for file in Path(results_dir).glob('concurrent_transport_*.json'):
        if 'raw' in file.name:
            continue
        match = re.search(pattern, file.name)
        if match:
            nodes = int(match.group(1))
            distance = float(match.group(2))
            rate = float(match.group(3))
            purify = match.group(4) == 'True'
            target_fid = float(match.group(5))
            verify = match.group(6) == 'True'
            with open(file) as f:
                data = json.load(f)
                fid_data = []
            nan_count = 0
            duration_data = []
            goodput = 0
            for run, val in data.items():
                fid_val = val["teleport_fids"][0]
                if fid_val is None or np.isnan(fid_val):
                    nan_count += 1
                else:
                    fid_data.append(fid_val)
                    if fid_val >= 0.7:
                        goodput += 1
                duration_data.append(val["duration"][0])
            goodput_ratio = goodput / len(fid_data) if len(fid_data) > 0 else 0
            key = (nodes, distance, rate, purify, verify, target_fid)
            if nodes == 5 and rate==8641 and not purify and verify:
                plot_cdf(fid_data,
                         f'Hop-by-Hop Fidelity CDF {nodes} Nodes {distance}km {rate}Hz Purify {purify} Verify {verify}',
                         'Fidelity', xlim=(0, 1), save=True,)
            results[key] = {"duration": np.mean(duration_data),
                            "teleport_fids": np.mean(fid_data),
                            "goodput_ratio": goodput_ratio,
                            "goodput": goodput}

    return results

def load_vbqt_results(results_dir):
    """Load hop-by-hop experiment results"""
    results = {}
    pattern = r'vbqt_transport_(\d+)_nodes_([\d.]+)km@([\d.]+)hz_purify_(\w+)_([\d.]+)_verify_(\w+)_(\d+)_qubit'
    pattern2 = r'vbqt_transport_(\d+)_nodes_([\d.]+)km@([\d.]+)hz_purify_(\w+)_([\d.]+)_verify_(\w+)_batch_(\d+)_qubit'

    for file in Path(results_dir).glob('vbqt_transport_*.json'):
        if 'raw' in file.name:
            continue
        if 'batch' in file.name:
            match = re.search(pattern2, file.name)
        else:
            match = re.search(pattern, file.name)
        if match:
            nodes = int(match.group(1))
            distance = float(match.group(2))
            rate = float(match.group(3))
            purify = match.group(4) == 'True'
            target_fid = float(match.group(5))
            verify = match.group(6) == 'True'
            with open(file) as f:
                data = json.load(f)
                fid_data = []
            nan_count = 0
            duration_data = []
            goodput = 0
            for run, val in data.items():
                for fid_val in val["teleport_fids_all"]:
                    if fid_val is None or np.isnan(fid_val):
                        nan_count += 1
                    else:
                        fid_data.append(fid_val)
                        if fid_val >= 0.7:
                            goodput += 1
                duration_data.append(val["duration"][0])
            goodput_ratio = goodput / len(fid_data) if len(fid_data) > 0 else 0
            key = (nodes, distance, rate, purify, verify, target_fid)
            # if nodes == 5 and rate==8641 and not purify and verify:
            #     plot_cdf(fid_data,
            #              f'Hop-by-Hop Fidelity CDF {nodes} Nodes {distance}km {rate}Hz Purify {purify} Verify {verify}',
            #              'Fidelity', xlim=(0, 1), save=True,)
            results[key] = {"duration": np.mean(duration_data),
                            "teleport_fids": np.mean(fid_data),
                            "goodput_ratio": goodput_ratio,
                            "goodput": goodput}

    return results


def load_vbqt_batch_results(results_dir):
    """Load hop-by-hop experiment results"""
    results = {}
    pattern = r'vbqt_transport_(\d+)_nodes_([\d.]+)km@([\d.]+)hz_purify_(\w+)_([\d.]+)_verify_(\w+)_batch_(\d+)_qubit'

    for file in Path(results_dir).glob('vbqt_transport_*.json'):
        if 'raw' in file.name:
            continue
        match = re.search(pattern, file.name)
        if match:
            nodes = int(match.group(1))
            distance = float(match.group(2))
            rate = float(match.group(3))
            purify = match.group(4) == 'True'
            target_fid = float(match.group(5))
            verify = match.group(6) == 'True'
            qubit_count = int(match.group(7))
            with open(file) as f:
                data = json.load(f)
                fid_data = []
            nan_count = 0
            duration_data = []
            goodput = 0
            for run, val in data.items():
                for fid_val in val["teleport_fids_all"]:
                    if fid_val is None or np.isnan(fid_val):
                        nan_count += 1
                    else:
                        fid_data.append(fid_val)
                        if fid_val >= 0.7:
                            goodput += 1
                duration_data.append(val["duration"][0])
            goodput_ratio = goodput / len(fid_data) if len(fid_data) > 0 else 0
            key = (nodes, distance, rate, purify, verify, target_fid, qubit_count)
            # if nodes == 5 and rate==8641 and not purify and verify:
            #     plot_cdf(fid_data,
            #              f'Hop-by-Hop Fidelity CDF {nodes} Nodes {distance}km {rate}Hz Purify {purify} Verify {verify}',
            #              'Fidelity', xlim=(0, 1), save=True,)
            results[key] = {"duration": np.mean(duration_data),
                            "teleport_fids": np.mean(fid_data),
                            "goodput_ratio": goodput_ratio,
                            "goodput": goodput,
                            "batch_size": qubit_count
                            }

    return results



def load_default_results(results_dir):
    """Load hop-by-hop experiment results"""
    results = {}
    pattern = r'hbh_transport_(\d+)_nodes_([\d.]+)km@([\d.]+)hz_purify_(\w+)_([\d.]+)_verify_(\w+)_(\d+)_qubit'

    for file in Path(results_dir).glob('hbh_transport_*.json'):
        if 'raw' in file.name:
            continue
        match = re.search(pattern, file.name)
        if match:
            nodes = int(match.group(1))
            distance = float(match.group(2))
            rate = float(match.group(3))
            purify = match.group(4) == 'True'
            target_fid = float(match.group(5))
            verify = match.group(6) == 'True'
            with open(file) as f:
                data = json.load(f)
                fid_data = []
            nan_count = 0
            duration_data = []
            goodput = 0
            for run, val in data.items():
                fid_val = val["teleport_fids"][0]
                if fid_val is None or np.isnan(fid_val):
                    nan_count += 1
                else:
                    fid_data.append(fid_val)
                    if fid_val >= 0.7:
                        goodput += 1
                duration_data.append(val["duration"][0])
            goodput_ratio = goodput / len(fid_data) if len(fid_data) > 0 else 0
            key = (nodes, distance, rate, purify, verify, target_fid)
            results[key] = {"duration": np.mean(duration_data),
                            "teleport_fids": np.mean(fid_data),
                            "goodput_ratio": goodput_ratio,
                            "goodput": goodput}

    return results

def plot_distance_comparison(e2e_results, hbh_results, default_result,  depolar_rate=8641, nodes=3,
                             save_dir="./sigmetrics_plots"):
    """Plot metrics vs distance for both schemes"""
    os.makedirs(save_dir, exist_ok=True)
    # Extract data for e2e (no purification, no verification)
    e2e_distances, e2e_fidelities, e2e_latencies, e2e_goodput = [], [], [], []
    for (n, d, r, p, v), data in e2e_results.items():
        if d < 1:
            continue
        if n == nodes and r == depolar_rate and not p and not v:
            e2e_distances.append(d)
            fid = data.get('teleport_fids', 0)
            e2e_fidelities.append(fid)
            e2e_goodput.append(data.get('goodput', 0))
            e2e_latencies.append(data.get('duration', 0))

    # Extract data for hop-by-hop (no purification, with verification)
    hbh_distances, hbh_fidelities, hbh_latencies, hbh_goodput = [], [], [], []
    for (n, d, r, p, v, tf), data in hbh_results.items():
        if n == nodes and r == depolar_rate and not p and v:
            if d not in hbh_distances:
                hbh_distances.append(d)
                hbh_fidelities.append(data.get('teleport_fids', 0))
                hbh_latencies.append(data.get('duration', 0))
                hbh_goodput.append(data.get('goodput', 0))
    default_distances, default_fidelities, default_latencies, default_goodput = [], [], [], []
    for (n, d, r, p, v, tf), data in default_result.items():
        if n == nodes and r == depolar_rate and not p and not v:
            if d not in default_distances:
                default_distances.append(d)
                default_fidelities.append(data.get('teleport_fids', 0))
                default_latencies.append(data.get('duration', 0))
                default_goodput.append(data.get('goodput', 0))


    # Sort by distance
    e2e_sorted = sorted(zip(e2e_distances, e2e_fidelities, e2e_latencies, e2e_goodput))
    hbh_sorted = sorted(zip(hbh_distances, hbh_fidelities, hbh_latencies, hbh_goodput))
    default_sorted = sorted(zip(default_distances, default_fidelities, default_latencies, default_goodput))

    if e2e_sorted:
        e2e_distances, e2e_fidelities, e2e_latencies, e2e_goodput = zip(*e2e_sorted)
    if hbh_sorted:
        hbh_distances, hbh_fidelities, hbh_latencies, hbh_goodput= zip(*hbh_sorted)
    if default_sorted:
        default_distances, default_fidelities, default_latencies, default_goodput = zip(*default_sorted)

    # Plot fidelity using plot_lines
    plot_lines(
        xs=[e2e_distances, hbh_distances, default_distances],
        ys=[e2e_fidelities, hbh_fidelities, default_fidelities],
        title=f'Fidelity vs Distance ({nodes} nodes, {depolar_rate} Hz)',
        x_label='Distance (km)',
        y_label='Average Fidelity',
        data_legends=[f'End-to-End {nodes} Nodes', f'VBQT {nodes} Nodes', f"Hop-by-Hop {nodes} Nodes"],
        save_dir=save_dir,
        ylim=(0, 1)
    )

    # Plot latency using plot_lines
    plot_lines(
        xs=[e2e_distances, hbh_distances, default_distances],
        ys=[e2e_latencies, hbh_latencies, default_latencies],
        title=f'Latency vs Distance ({nodes} nodes, {depolar_rate} Hz)',
        x_label='Distance (km)',
        y_label='Transmission Time (ns)',
        data_legends=[f'End-to-End {nodes} Nodes', f'VBQT {nodes} Nodes', f"Hop-by-Hop {nodes} Nodes"],
        save_dir=save_dir,
    )
    # plot goodput using plot_lines
    e2e_goodput = np.array(e2e_goodput) / (np.array(e2e_latencies) / 1e3)  # Convert ns to us
    hbh_goodput = np.array(hbh_goodput) / (np.array(hbh_latencies) / 1e3)  # Convert ns to us
    default_goodput = np.array(default_goodput) / (np.array(default_latencies) / 1e3)  # Convert ns to us
    plot_lines(
        xs=[e2e_distances, hbh_distances, default_distances],
        ys=[e2e_goodput, hbh_goodput, default_goodput],
        title=f'Goodput vs Distance ({nodes} nodes, {depolar_rate} Hz)',
        x_label='Distance (km)',
        y_label='Goodput',
        data_legends=[f'End-to-End {nodes} Nodes', f'VBQT {nodes} Nodes', f"Hop-by-Hop {nodes} Nodes"],
        save_dir=save_dir,
        # ylim=(0, 1)
    )
def plot_distance_comparison_more_hbh_node(e2e_results, hbh_results, default_result,
                                           depolar_rate=8641, e2e_nodes=5, hbh_nodes=5,
                                           save_dir="./sigmetrics_plots"):
    """Plot metrics vs distance for both schemes"""
    os.makedirs(save_dir, exist_ok=True)
    # Extract data for e2e (no purification, no verification)
    e2e_distances, e2e_fidelities, e2e_latencies, e2e_goodput, e2e_goodput_ratio = [], [], [], [], []
    for (n, d, r, p, v), data in e2e_results.items():
        if n == e2e_nodes and r == depolar_rate and not p and not v:
            e2e_distances.append(d)
            e2e_fidelities.append(data.get('teleport_fids', 0))
            e2e_latencies.append(data.get('duration', 0))
            e2e_goodput.append(data.get('goodput', 0))
            e2e_goodput_ratio.append(data.get('goodput_ratio', 0))

    # Extract data for hop-by-hop (no purification, with verification)
    # expected_dis = [(2 * x)/4 for x in e2e_distances]
    expected_dis = sorted(e2e_distances)

    hbh_distances, hbh_fidelities, hbh_latencies, hbh_goodput, hbh_goodput_ratio = [], [], [], [], []
    for (n, d, r, p, v, tf), data in hbh_results.items():
        if n == hbh_nodes and r == depolar_rate and not p and v and tf==0.0:
            if d in expected_dis and d not in hbh_distances:
                hbh_distances.append(d)
                hbh_fidelities.append(data.get('teleport_fids', 0))
                hbh_latencies.append(data.get('duration', 0))
                hbh_goodput.append(data.get('goodput', 0))
                hbh_goodput_ratio.append(data.get('goodput_ratio', 0))

    (hbh_distances_more, hbh_fidelities_more, hbh_latencies_more,
     hbh_goodput_more, hbh_goodput_ratio_more) = [], [], [], [], []
    for (n, d, r, p, v, tf), data in hbh_results.items():
        if n == 6 and r == depolar_rate and not p and v and tf == 0.0:
            if d not in hbh_distances_more:
                hbh_distances_more.append(d)
                hbh_fidelities_more.append(data.get('teleport_fids', 0))
                hbh_latencies_more.append(data.get('duration', 0))
                hbh_goodput_more.append(data.get('goodput', 0))
                hbh_goodput_ratio_more.append(data.get('goodput_ratio', 0))

    default_distances, default_fidelities, default_latencies, default_goodput, default_goodput_ratio = [], [], [], [], []
    for (n, d, r, p, v, tf), data in default_result.items():
        if n == hbh_nodes and r == depolar_rate and not p and not v:
            if d in expected_dis and d not in default_distances:
                default_distances.append(d)
                default_fidelities.append(data.get('teleport_fids', 0))
                default_latencies.append(data.get('duration', 0))
                default_goodput.append(data.get('goodput', 0))
                default_goodput_ratio.append(data.get('goodput_ratio', 0))
    total_dis = [4*x for x in hbh_distances]
    total_dis = sorted(total_dis)
    # Sort by distance
    e2e_sorted = sorted(zip(e2e_distances, e2e_fidelities, e2e_latencies, e2e_goodput, e2e_goodput_ratio))
    hbh_sorted = sorted(zip(hbh_distances, hbh_fidelities, hbh_latencies, hbh_goodput, hbh_goodput_ratio))
    hbh_sorted_more = sorted(zip(hbh_distances_more, hbh_fidelities_more, hbh_latencies_more, hbh_goodput_more, hbh_goodput_ratio_more))
    default_sorted = sorted(zip(default_distances, default_fidelities, default_latencies, default_goodput, default_goodput_ratio))

    if e2e_sorted:
        e2e_distances, e2e_fidelities, e2e_latencies, e2e_goodput, e2e_goodput_ratio= zip(*e2e_sorted)
    if hbh_sorted:
        hbh_distances, hbh_fidelities, hbh_latencies, hbh_goodput, hbh_goodput_ratio= zip(*hbh_sorted)
    if default_sorted:
        default_distances, default_fidelities, default_latencies, default_goodput, default_goodput_ratio= zip(*default_sorted)
    if hbh_sorted_more:
        hbh_distances_more, hbh_fidelities_more, hbh_latencies_more, hbh_goodput_more, hbh_goodput_ratio_more= zip(*hbh_sorted_more)

    # Plot fidelity using plot_lines
    plot_lines(
        xs=[total_dis, total_dis, total_dis],
        ys=[e2e_fidelities[:len(total_dis)], hbh_fidelities, default_fidelities],
        title=f'Fidelity vs Distance ({e2e_nodes} e2e nodes, 5 hbh nodes, {depolar_rate} Hz)',
        x_label='Total Path Distance (km)',
        y_label='Average Fidelity',
        data_legends=[f'End-to-End {e2e_nodes} Nodes', 'VBQT 5 Nodes', 'Hop-by-Hop 5 Nodes'],
        save_dir=save_dir,
        ylim=(0, 1)
    )

    # Plot latency using plot_lines
    plot_lines(
        xs=[total_dis, total_dis, total_dis],
        ys=[e2e_latencies[:len(total_dis)], hbh_latencies, default_latencies],
        title=f'Latency vs Distance ({e2e_nodes} e2e nodes, 5 hbh nodes, {depolar_rate} Hz)',
        x_label='Total Path Distance (km)',
        y_label='Transmission Time (ns)',
        data_legends=[f'End-to-End {e2e_nodes} Nodes', 'VBQT 5 Nodes', 'Hop-by-Hop 5 Nodes'],
        save_dir=save_dir,
    )
    # plot goodput using plot_lines
    e2e_latencies_ms = np.array(e2e_latencies) / 1e3  # Convert ns to us
    hbh_latencies_ms = np.array(hbh_latencies) / 1e3  # Convert ns to us
    default_latencies_ms = np.array(default_latencies) / 1e3 # Convert ns to us
    e2e_goodput_per_us = np.array(e2e_goodput[:len(total_dis)]) / e2e_latencies_ms[:len(total_dis)]
    hbh_goodput_per_us = np.array(hbh_goodput) / hbh_latencies_ms
    hbh_goodput_more_per_us = np.array(hbh_goodput_more) / (np.array(hbh_latencies_more) / 1e3)
    default_goodput_per_us = np.array(default_goodput) / default_latencies_ms
    plot_lines(
        xs=[total_dis, total_dis, total_dis, total_dis],
        ys=[e2e_goodput_per_us, hbh_goodput_per_us, default_goodput_per_us], # hbh_goodput_more_per_us
        title=f'Goodput vs Distance ({e2e_nodes} e2e nodes, 5 hbh nodes, {depolar_rate} Hz)',
        x_label='Total Path Distance (km)',
        y_label='Goodput',
        data_legends=[f'End-to-End {e2e_nodes} Nodes', 'VBQT 5 Nodes', 'Hop-by-Hop 5 Nodes', ], #'VBQT 6 Nodes'
        save_dir=save_dir,
        # ylim=(0, 1)
    )
    # plot goodput ratio
    plot_lines(
        xs=[total_dis, total_dis, total_dis],
        ys=[e2e_goodput_ratio[:len(total_dis)], hbh_goodput_ratio, default_goodput_ratio],
        title=f'Goodput Ratio vs Distance ({e2e_nodes} e2e nodes, 5 hbh nodes, {depolar_rate} Hz)',
        x_label='Total Path Distance (km)',
        y_label='Goodput Ratio',
        data_legends=[f'End-to-End {e2e_nodes} Nodes', 'VBQT 5 Nodes', 'Hop-by-Hop 5 Nodes'],
        save_dir=save_dir,
        ylim=(0, 1)
    )


def plot_nodecount_comparison(e2e_results, hbh_results,default_results, depolar_rate=8641, distance=1.0,
                              save_dir="./sigmetrics_plots"):
    """Plot metrics vs node count for both schemes"""

    os.makedirs(save_dir, exist_ok=True)
    # Extract data for e2e
    e2e_nodes, e2e_fidelities, e2e_latencies, e2e_goodput = [], [], [], []
    for (n, d, r, p, v), data in e2e_results.items():
        if d == distance and r == depolar_rate and not p and not v:
            e2e_nodes.append(n)
            e2e_fidelities.append(data.get('teleport_fids', 0))
            e2e_latencies.append(data.get('duration', 0))
            e2e_goodput.append(data.get('goodput', 0))

    # Extract data for hop-by-hop
    hbh_nodes, hbh_fidelities, hbh_latencies, hbh_goodput = [], [], [], []
    for (n, d, r, p, v, tf), data in hbh_results.items():
        if d == distance and r == depolar_rate and not p and v:
            hbh_nodes.append(n)
            hbh_fidelities.append(data.get('teleport_fids', 0))
            hbh_latencies.append(data.get('duration', 0))
            hbh_goodput.append(data.get('goodput', 0))
    default_nodes, default_fidelities, default_latencies, default_goodput = [], [], [], []
    for (n, d, r, p, v, tf), data in default_results.items():
        if d == distance and r == depolar_rate and not p and not v:
            if n in default_nodes:
                continue
            default_nodes.append(n)
            default_fidelities.append(data.get('teleport_fids', 0))
            default_latencies.append(data.get('duration', 0))
            default_goodput.append(data.get('goodput', 0))

    # Sort by node count
    e2e_sorted = sorted(zip(e2e_nodes, e2e_fidelities, e2e_latencies, e2e_goodput))
    hbh_sorted = sorted(zip(hbh_nodes, hbh_fidelities, hbh_latencies, hbh_goodput))
    default_sorted = sorted(zip(default_nodes, default_fidelities, default_latencies, default_goodput))

    if e2e_sorted:
        e2e_nodes, e2e_fidelities, e2e_latencies, e2e_goodput = zip(*e2e_sorted)

    if hbh_sorted:
        hbh_nodes, hbh_fidelities, hbh_latencies, hbh_goodput = zip(*hbh_sorted)
    if default_sorted:
        default_nodes, default_fidelities, default_latencies, default_goodput = zip(*default_sorted)
    # calculate the goodput over time
    e2e_goodput = np.array(e2e_goodput) / (np.array(e2e_latencies) / 1e3)  # Convert ns to us
    hbh_goodput = np.array(hbh_goodput) / (np.array(hbh_latencies) / 1e3)  # Convert ns to us
    default_goodput = np.array(default_goodput) / (np.array(default_latencies) / 1e3)  # Convert ns to us
    # Plot fidelity using plot_lines
    plot_lines(
        xs=[e2e_nodes, hbh_nodes, default_nodes],
        ys=[e2e_fidelities, hbh_fidelities, default_fidelities],
        title=f'Fidelity vs Node Count ({distance} km, {depolar_rate} Hz)',
        x_label='Number of Nodes',
        y_label='Average Fidelity',
        data_legends=['End-to-End', 'VBQT', 'Hop-by-Hop'],
        save_dir=save_dir
    )
    plot_bars(
        xs=[e2e_nodes, hbh_nodes, default_nodes],
        ys=[e2e_fidelities, hbh_fidelities, default_fidelities],
        title=f'Fidelity vs Node Count ({distance} km, {depolar_rate} Hz) bar',
        x_label='Number of Nodes',
        y_label='Average Fidelity',
        data_legends=['End-to-End', 'VBQT', 'Hop-by-Hop'],
        save_dir=save_dir
    )
    # Plot latency using plot_lines
    plot_lines(
        xs=[e2e_nodes, hbh_nodes, default_nodes],
        ys=[e2e_latencies, hbh_latencies, default_latencies],
        title=f'Latency vs Node Count ({distance} km, {depolar_rate} Hz)',
        x_label='Number of Nodes',
        y_label='Transmission Time (ns)',
        data_legends=['End-to-End', 'VBQT', 'Hop-by-Hop'],
        save_dir=save_dir
    )
    plot_bars(
        xs=[e2e_nodes, hbh_nodes, default_nodes],
    ys=[e2e_latencies, hbh_latencies, default_latencies],
        title=f'Latency Comparison ({distance} km, {depolar_rate} Hz) bar',
        x_label='Number of Nodes',
        y_label='Transmission Time (ns)',
        data_legends=['End-to-End', 'VBQT', 'Hop-by-Hop'],
        save_dir=save_dir
    )
    # plot goodput using plot_bar
    plot_bars(xs=[e2e_nodes, hbh_nodes, default_nodes],
        ys=[e2e_goodput, hbh_goodput, default_goodput],
        title=f'Goodput vs Node Count ({distance} km, {depolar_rate} Hz) bar',
        x_label='Number of Nodes',
        y_label='Goodput',
        data_legends=['End-to-End', 'VBQT', 'Hop-by-Hop'],
        save_dir=save_dir
    )

    plot_lines(xs=[e2e_nodes, hbh_nodes, default_nodes],
              ys=[e2e_goodput, hbh_goodput, default_goodput],
              title=f'Goodput vs Node Count ({distance} km, {depolar_rate} Hz)',
              x_label='Number of Nodes',
              y_label='Goodput',
              data_legends=['End-to-End', 'VBQT', 'Hop-by-Hop'],
              save_dir=save_dir
    )

def plot_vbqt_batch_size_comparison(vbqt_results, depolar_rate=8641, distance=1.0,
                              save_dir="./sigmetrics_plots"):
    """Plot metrics vs node count for both schemes"""

    os.makedirs(save_dir, exist_ok=True)
    # Extract data for e2e
    hbh_distances, hbh_fidelities, hbh_latencies, hbh_goodput, hbh_goodput_ratio, hbh_batch_size =\
        [], [], [], [], [], []
    for (n, d, r, p, v, tf, bs), data in vbqt_results.items():
        hbh_distances.append(d)
        hbh_fidelities.append(data.get('teleport_fids', 0))
        hbh_latencies.append(data.get('duration', 0))
        hbh_goodput.append(data.get('goodput', 0))
        hbh_goodput_ratio.append(data.get('goodput_ratio', 0))
        hbh_batch_size.append(data.get('batch_size', 0))
    # Sort by batch size
    hbh_sorted = sorted(zip(hbh_batch_size, hbh_fidelities, hbh_latencies, hbh_goodput, hbh_goodput_ratio))
    if hbh_sorted:
        hbh_batch_size, hbh_fidelities, hbh_latencies, hbh_goodput, hbh_goodput_ratio = zip(*hbh_sorted)
    # calculate the goodput over time
    hbh_latencies_ms = np.array(hbh_latencies) / 1e3
    hbh_goodput_per_us = np.array(hbh_goodput) / hbh_latencies_ms
    # Plot fidelity using plot_lines
    plot_bars(
        xs=[hbh_batch_size],
        ys=[hbh_fidelities],
        title=f'Fidelity vs Batch Size ({distance} km, {depolar_rate} Hz) bar',
        x_label='Batch Size',
        y_label='Average Fidelity',
        # data_legends=[f"VBQT Batch Size {size}" for size in hbh_batch_size],
        data_legends="VBQT Batch Size",
        save_dir=save_dir
    )
    plot_bars(xs=[hbh_batch_size],
              ys=[hbh_latencies],
              title=f'Latency vs Batch Size ({distance} km, {depolar_rate} Hz bar',
                x_label='Batch Size',
                y_label='Transmission Time (ns)',
                # data_legends=[f"VBQT Batch Size {size}" for size in hbh_batch_size],
              data_legends="VBQT Batch Size",
                save_dir=save_dir
              )
    # plot goodput using plot_bar
    plot_bars(xs=[hbh_batch_size],
              ys=[hbh_goodput_per_us],
              title=f'Goodput vs Batch Size ({distance} km, {depolar_rate} Hz) bar',
                x_label='Batch Size',
                y_label='Goodput',
                # data_legends=[f"VBQT Batch Size {size}" for size in hbh_batch_size],
              data_legends="VBQT Batch Size",
                save_dir=save_dir
              )


if __name__ == '__main__':
    # Load results
    # e2e_results = load_e2e_results('../end_to_end_experiment/transportation_results')
    e2e_results = load_e2e_results('../end_to_end_experiment/e2e_combined_result')
    # hbh_results = load_hopbyhop_results('./transportation_results')
    hbh_results = load_vbqt_results('./vbqt_results')
    batch_results = load_vbqt_batch_results("./vbqt_batchsize_experiments")
    default_results = load_default_results('./transportation_results')
    print(f"Loaded {len(e2e_results)} end-to-end results")
    print(f"Loaded {len(hbh_results)} hop-by-hop results")
    print(f"Loaded {len(default_results)} default hop-by-hop results")

    # Plot distance comparison (3 nodes, 8641 Hz)
    # plot_distance_comparison(e2e_results, hbh_results, default_results, depolar_rate=8641, nodes=3)
    # plot_distance_comparison(e2e_results, hbh_results, default_results, depolar_rate=24483, nodes=3)
    # plot_distance_comparison_more_hbh_node(e2e_results, hbh_results, default_results, depolar_rate=8641, e2e_nodes=5,
    #                                        hbh_nodes=5)

    # Plot node count comparison (1.0 km, 8641 Hz)
    # plot_nodecount_comparison(e2e_results, hbh_results, default_results, depolar_rate=8641, distance=1.0
    #                           , save_dir="./icdcs_plots/")
    # plot_distance_comparison(e2e_results, hbh_results, default_results, depolar_rate=8641, nodes=3,
    #                          save_dir="./icdcs_plots/")
    # plot_distance_comparison(e2e_results, hbh_results, default_results, depolar_rate=8641, nodes=5,
    #                          save_dir="./icdcs_plots/")
    # plot_distance_comparison_more_hbh_node(e2e_results, hbh_results, default_results, depolar_rate=8641, e2e_nodes=5,
    #                                        hbh_nodes=5,
    #                                        save_dir="./icdcs_plots/")
    plot_distance_comparison(e2e_results, hbh_results, default_results, depolar_rate=24483, nodes=3,
                             save_dir="./icdcs_plots/")
    plot_distance_comparison(e2e_results, hbh_results, default_results, depolar_rate=24483, nodes=5,
                             save_dir="./icdcs_plots/")
    # plot_vbqt_batch_size_comparison(vbqt_results=batch_results, depolar_rate=8641, distance=1.0,
    #                                 save_dir="./icdcs_plots/")