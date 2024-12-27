


import math
import os
import json

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
    # load hop by hop data
    save_dir = f"./transportation_results/figures"
    os.makedirs(save_dir, exist_ok=True)
    with open("./transportation_results/max_11_nodes_purification.json", "r") as f:
        multi_node_hbh = json.load(f)
    with open("../end_to_end_experiment/transportation_results/e2e_transport_11.json", "r") as f:
        multi_node_e2e = json.load(f)
    with open("./transportation_results/max_11_km_5_nodes_purification.json", "r") as f:
        multi_dis_hbh = json.load(f)
    with open("../end_to_end_experiment/transportation_results/e2e_transport_11_km_5_nodes.json", "r") as f:
        multi_dis_e2e = json.load(f)

    x = [int(key) for key in multi_node_hbh.keys()]
    keys = list(multi_node_hbh.keys())
    hbh_duration = [multi_node_hbh[k]["duration"] for k in keys]
    hbh_teleport_rate = [multi_node_hbh[k]["teleport_success_rate"] for k in keys]
    e2e_duration = [multi_node_e2e[k]["duration"] for k in keys]
    e2e_teleport_rate = [multi_node_e2e[k]["teleport_success_rate"] for k in keys]

    plot_lines([x, x],
               [hbh_duration, e2e_duration],
               "Multi Node Hope by Hope vs End to End Transport Duration",
               "Node Count",
               "Duration of Transport (ns)",
               ["Hop By Hop", "End to End"],
               save_dir=save_dir
               )
    plot_lines([x, x],
               [hbh_teleport_rate, e2e_teleport_rate],
               "Multi Node Hope by Hope vs End to End Transport Success Rate",
               "Node Count",
               "Success Rate",
               ["Hop By Hop", "End to End"],
               save_dir=save_dir)

    x = [int(key) for key in multi_dis_hbh.keys()]
    keys = list(multi_dis_hbh.keys())
    hbh_dis_duration = [multi_dis_hbh[k]["duration"] for k in keys]
    e2e_dis_duration = [multi_dis_e2e[k]["duration"] for k in keys]
    hbh_dis_teleport_rate = [multi_dis_hbh[k]["teleport_success_rate"] for k in keys]
    e2e_dis_teleport_rate = [multi_dis_e2e[k]["teleport_success_rate"] for k in keys]
    plot_lines([x, x],
               [hbh_dis_duration, e2e_dis_duration],
               "Multi Distance Hope by Hope vs End to End Transport Duration (5 Nodes)",
               "Node Distance (km)",
               "Duration of Transport (ns)",
               ["Hop By Hop", "End to End"],
               save_dir=save_dir
               )
    plot_lines([x, x],
               [hbh_dis_teleport_rate, e2e_dis_teleport_rate],
               "Multi Distance Hope by Hope vs End to End Transport Success Rate (5 Nodes)",
               "Node Distance (km)",
               "Success Rate",
               ["Hop By Hop", "End to End"],
               save_dir=save_dir)



if __name__ == '__main__':
    main()