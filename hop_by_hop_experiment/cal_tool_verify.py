import json

import numpy as np

if __name__ == '__main__':
    with open("./transportation_results/3nodes_1_qubit_verification_raw.json", "r") as file:
        data = json.load(file)
    fid_data = data["teleport_fids"]
    g_7_count = 0
    g_9_count = 0
    d_data = data["duration"]
    for fid in fid_data:
        if fid > 0.7:
            g_7_count += 1
        if fid > 0.9:
            g_9_count += 1

    print(f"3 Nodes > 0.7: {g_7_count/ len(fid_data)}")
    print(f"3 Nodes > 0.9: {g_9_count/ len(fid_data)}")
    print(f"3 Nodes Duration: {np.mean(d_data)}")

    with open("./transportation_results/4nodes_1_qubit_verification_raw.json", "r") as file:
        data = json.load(file)
    fid_data = data["teleport_fids"]
    d_data = data["duration"]
    g_7_count = 0
    g_9_count = 0
    for fid in fid_data:
        if fid > 0.7:
            g_7_count += 1
        if fid > 0.9:
            g_9_count += 1

    print(f"4 Nodes > 0.7: {g_7_count / len(fid_data)}")
    print(f"4 Nodes > 0.9: {g_9_count / len(fid_data)}")
    print(f"4 Nodes Duration: {np.mean(d_data)}")


    with open("./transportation_results/3nodes_verification_throughput_raw.json", "r") as file:
        data = json.load(file)
    g_7_counts = []
    g_9_counts = []
    g_7_global_count = 0
    g_9_global_count = 0
    total_count = 0
    for value_data in data.values():
        g_7_count = 0
        g_9_count = 0
        fid_data = value_data["all_fidelity"]
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

    print(f"3 Nodes Throughput > 0.7: {np.mean(g_7_counts)}")
    print(f"3 Nodes Throughput > 0.9: {np.mean(g_9_counts)}")
    print(f"3 Nodes Global > 0.7: {g_7_global_count/total_count}")
    print(f"3 Nodes Global > 0.9: {g_9_global_count/total_count}")

    with open("./transportation_results/4nodes_verification_throughput_raw.json", "r") as file:
        data = json.load(file)
    g_7_counts = []
    g_9_counts = []
    g_7_global_count = 0
    g_9_global_count = 0
    total_count = 0
    for value_data in data.values():
        g_7_count = 0
        g_9_count = 0
        fid_data = value_data["all_fidelity"]
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

    print(f"4 Nodes Throughput > 0.7: {np.mean(g_7_counts)}")
    print(f"4 Nodes Throughput > 0.9: {np.mean(g_9_counts)}")
    print(f"4 Nodes Global > 0.7: {g_7_global_count / total_count}")
    print(f"4 Nodes Global > 0.9: {g_9_global_count / total_count}")



    # 0.5 km
    with open("./transportation_results/4nodes_verification_throughput_raw_batch_4_0.5.json", "r") as file:
        data = json.load(file)
    g_7_counts = []
    g_9_counts = []
    g_7_global_count = 0
    g_9_global_count = 0
    total_count = 0
    for value_data in data.values():
        g_7_count = 0
        g_9_count = 0
        fid_data = value_data["all_fidelity"]
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

    print(f"4 Nodes 0.5km Throughput > 0.7: {np.mean(g_7_counts)}")
    print(f"4 Nodes 0.5km Throughput > 0.9: {np.mean(g_9_counts)}")
    print(f"4 Nodes 0.5km Global > 0.7: {g_7_global_count / total_count}")
    print(f"4 Nodes 0.5km Global > 0.9: {g_9_global_count / total_count}\n")


    with open("./transportation_results/4nodes_verification_throughput_raw_batch_4_1.0.json", "r") as file:
        data = json.load(file)
    g_7_counts = []
    g_9_counts = []
    g_7_global_count = 0
    g_9_global_count = 0
    total_count = 0
    all_count = []
    for value_data in data.values():
        g_7_count = 0
        g_9_count = 0
        fid_data = value_data["all_fidelity"]
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

    print(f"4 Nodes 1km Throughput Count: {np.mean(all_count)}")
    print(f"4 Nodes 1km Throughput > 0.7: {np.mean(g_7_counts)}")
    print(f"4 Nodes 1km Throughput > 0.9: {np.mean(g_9_counts)}")
    print(f"4 Nodes 1km Global > 0.7: {g_7_global_count / total_count}")
    print(f"4 Nodes 1km Global > 0.9: {g_9_global_count / total_count}\n")


    with open("./transportation_results/4nodes_1_qubit_verification_0.5km_raw.json", "r") as file:
        data = json.load(file)
    fid_data = data["teleport_fids"]
    d_data = data["duration"]
    all_fid = []
    g_7_count = 0
    g_9_count = 0
    for fid in fid_data:
        if fid > 0.7:
            g_7_count += 1
        if fid > 0.9:
            g_9_count += 1
        all_fid.append(fid)

    print(f"4 Nodes 0.5km > 0.7: {g_7_count / len(fid_data)}")
    print(f"4 Nodes 0.5km > 0.9: {g_9_count / len(fid_data)}")
    print(f"4 Nodes 0.5km Duration: {np.mean(d_data)}")
    print(f"4 Nodes 0.5km average fid: {np.mean(all_fid)}\n")


    with open("./transportation_results/3nodes_1_qubit_verification_0.5km_raw.json", "r") as file:
        data = json.load(file)
    fid_data = data["teleport_fids"]
    d_data = data["duration"]
    all_fid = []
    g_7_count = 0
    g_9_count = 0
    for fid in fid_data:
        if fid > 0.7:
            g_7_count += 1
        if fid > 0.9:
            g_9_count += 1
        all_fid.append(fid)

    print(f"3 Nodes 0.5km > 0.7: {g_7_count / len(fid_data)}")
    print(f"3 Nodes 0.5km > 0.9: {g_9_count / len(fid_data)}")
    print(f"3 Nodes 0.5km Duration: {np.mean(d_data)}")
    print(f"3 Nodes 0.5km average fid: {np.mean(all_fid)}")
