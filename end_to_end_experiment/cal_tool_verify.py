import json

import numpy as np

if __name__ == '__main__':
    with open("./transportation_results/e2e_3nodes_1_qubit_verification_raw.json", "r") as file:
        data = json.load(file)
    fid_data = data["teleport_fids"]
    g_7_count = 0
    g_9_count = 0
    all_fid = []
    d_data = data["duration"]
    for fid in fid_data:
        if np.isnan(fid):
            continue
        if fid > 0.7:
            g_7_count += 1
        if fid > 0.9:
            g_9_count += 1
        all_fid.append(fid)

    print(f"3 Nodes > 0.7: {g_7_count/ len(all_fid)}")
    print(f"3 Nodes > 0.9: {g_9_count/ len(all_fid)}")
    print(f"average duration: {np.mean(d_data)}")
    print(f"average fid: {np.mean(all_fid)}")


    with open("./transportation_results/e2e_4nodes_1_qubit_verification_raw.json", "r") as file:
        data = json.load(file)
    fid_data = data["teleport_fids"]
    g_7_count = 0
    g_9_count = 0
    all_fid = []
    d_data = data["duration"]
    for fid in fid_data:
        if np.isnan(fid):
            continue
        if fid > 0.7:
            g_7_count += 1
        if fid > 0.9:
            g_9_count += 1
        all_fid.append(fid)

    print(f"4 Nodes > 0.7: {g_7_count/ len(all_fid)}")
    print(f"4 Nodes > 0.9: {g_9_count/ len(all_fid)}")
    print(f"average duration: {np.mean(d_data)}")
    print(f"average fid: {np.mean(all_fid)}\n")


    with open("./transportation_results/e2e_4nodes_throughput_1km_raw.json", "r") as file:
        data = json.load(file)
    g_7_counts = []
    g_9_counts = []
    g_7_global_count = 0
    g_9_global_count = 0
    total_count = 0
    all_count = []
    for value_data in data["4"].values():
        g_7_count = 0
        g_9_count = 0
        fid_data = value_data["all_fids"]
        for fid in fid_data:
            if fid > 0.7:
                g_7_count += 1
                g_7_global_count += 1
            if fid > 0.9:
                g_9_count += 1
                g_9_global_count += 1
            total_count += 1
        if len(fid_data) > 0:
            g_7_counts.append(g_7_count)
            g_9_counts.append(g_9_count)
        all_count.append(value_data["total_count"])

    print(f"e2e 4 Nodes 1km Throughput Count: {np.mean(all_count)}")
    print(f"e2e 4 Nodes 1km Throughput > 0.7: {np.mean(g_7_counts)}")
    print(f"e2e 4 Nodes 1km Throughput > 0.9: {np.mean(g_9_counts)}")
    print(f"e2e 4 Nodes 1km Global > 0.7: {g_7_global_count / total_count}")
    print(f"e2e 4 Nodes 1km Global > 0.9: {g_9_global_count / total_count}")



    with open("./transportation_results/e2e_4nodes_throughput_1km_raw.json", "r") as file:
        data = json.load(file)
    g_7_counts = []
    g_9_counts = []
    g_7_global_count = 0
    g_9_global_count = 0
    total_count = 0
    all_count = []
    for value_data in data["4"].values():
        g_7_count = 0
        g_9_count = 0
        fid_data = value_data["all_fids"]
        for fid in fid_data:
            if fid > 0.7:
                g_7_count += 1
                g_7_global_count += 1
            if fid > 0.9:
                g_9_count += 1
                g_9_global_count += 1
            total_count += 1
        if len(fid_data) > 0:
            g_7_counts.append(g_7_count)
            g_9_counts.append(g_9_count)
        all_count.append(value_data["total_count"])

    print(f"e2e 4 Nodes 1km Throughput Count: {np.mean(all_count)}")
    print(f"e2e 4 Nodes 1km Throughput > 0.7: {np.mean(g_7_counts)}")
    print(f"e2e 4 Nodes 1km Throughput > 0.9: {np.mean(g_9_counts)}")
    print(f"e2e 4 Nodes 1km Global > 0.7: {g_7_global_count / total_count}")
    print(f"e2e 4 Nodes 1km Global > 0.9: {g_9_global_count / total_count}\n")


    with open("./transportation_results/e2e_4nodes_1_qubit_verification_0.5km_raw.json", "r") as file:
        data = json.load(file)
    fid_data = data["teleport_fids"]
    g_7_count = 0
    g_9_count = 0
    all_fid = []
    d_data = data["duration"]
    for fid in fid_data:
        if np.isnan(fid):
            continue
        if fid > 0.7:
            g_7_count += 1
        if fid > 0.9:
            g_9_count += 1
        all_fid.append(fid)

    print(f"4 Nodes 0.5 Km Verify > 0.7: {g_7_count/ len(all_fid)}")
    print(f"4 Nodes 0.5 Km Verify > 0.9: {g_9_count/ len(all_fid)}")
    print(f"average duration: {np.mean(d_data)}")
    print(f"average fid: {np.mean(all_fid)}\n")
