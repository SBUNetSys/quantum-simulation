import json

import numpy as np

if __name__ == '__main__':
    with open("./transportation_results/e2e_3nodes_1_qubit_purification_raw.json", 'r') as f:
        data = json.load(f)
    fid_data = data["teleport_fids"]
    g_7_count = 0
    g_9_count = 0
    d_data = data["duration"]
    for fid in fid_data:
        if fid > 0.7:
            g_7_count += 1
        if fid > 0.9:
            g_9_count += 1

    print(f"E2E 3 Node 1Km Purification Success Rate > 0.7: {g_7_count / len(fid_data)}")
    print(f"E2E 3 Node 1Km Purification Success Rate  > 0.9: {g_9_count / len(fid_data)}")
    print(f"E2E 3 Node 1Km Purification Success Rate Duration: {np.mean(d_data)}\n")

    with open("./transportation_results/e2e_3nodes_purification_throughput_raw.json", 'r') as f:
        data = json.load(f)
    g_7_counts = []
    g_9_counts = []
    g_7_global_count = 0
    g_9_global_count = 0
    total_count = 0
    all_count = []
    for value_data in data["3"].values():
        g_7_count = 0
        g_9_count = 0
        fid_data = value_data["all_fid"]
        all_count.append(len(fid_data))
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

    print(f"E2E 3 Node 1Km Throughput Count: {np.mean(all_count)}")
    print(f"E2E 3 Node 1Km Purification Throughput > 0.7: {np.mean(g_7_counts)}")
    print(f"E2E 3 Node 1Km Purification Throughput > 0.9: {np.mean(g_9_counts)}")
    print(f"E2E 3 Node 1Km Purification Global > 0.7: {g_7_global_count / total_count}")
    print(f"E2E 3 Node 1Km Purification Global > 0.9: {g_9_global_count / total_count}\n")

    with open("./transportation_results/e2e_3nodes_1_qubit_verification_1.0km_raw.json", 'r') as f:
        data = json.load(f)
    fid_data = data["teleport_fids"]
    g_7_count = 0
    g_9_count = 0
    d_data = data["duration"]
    for fid in fid_data:
        if fid > 0.7:
            g_7_count += 1
        if fid > 0.9:
            g_9_count += 1

    print(f"E2E 3 Node 1Km Verification Success Rate > 0.7: {g_7_count / len(fid_data)}")
    print(f"E2E 3 Node 1Km Verification Success Rate  > 0.9: {g_9_count / len(fid_data)}")
    print(f"E2E 3 Node 1Km Verification Success Rate Duration: {np.mean(d_data)}\n")


    with open("./transportation_results/e2e_3nodes_1_qubit_verification_0.5km_raw.json", 'r') as f:
        data = json.load(f)
    fid_data = data["teleport_fids"]
    g_7_count = 0
    g_9_count = 0
    d_data = data["duration"]
    for fid in fid_data:
        if fid > 0.7:
            g_7_count += 1
        if fid > 0.9:
            g_9_count += 1

    print(f"E2E 3 Node 0.5Km Verification Success Rate > 0.7: {g_7_count / len(fid_data)}")
    print(f"E2E 3 Node 0.5Km Verification Success Rate  > 0.9: {g_9_count / len(fid_data)}")
    print(f"E2E 3 Node 0.5Km Verification Success Rate Duration: {np.mean(d_data)}\n")

    with open("./transportation_results/e2e_3nodes_throughput_1km_verify_raw.json", 'r') as f:
        data = json.load(f)
    g_7_counts = []
    g_9_counts = []
    g_7_global_count = 0
    g_9_global_count = 0
    total_count = 0
    all_count = []
    for value_data in data["3"].values():
        g_7_count = 0
        g_9_count = 0
        fid_data = value_data["all_fids"]
        all_count.append(len(fid_data))
        for fid in fid_data:
            if fid > 0.7:
                g_7_count += 1
                g_7_global_count += 1
            if fid > 0.9:
                g_9_count += 1
                g_9_global_count += 1
            total_count += 1
        if len(fid_data) > 0:
            g_7_counts.append(g_7_count/len(fid_data))
            g_9_counts.append(g_9_count/len(fid_data))

    print(f"E2E 3 Node 1Km Verification Throughput Count: {np.mean(all_count)}")
    print(f"E2E 3 Node 1Km Verification Throughput > 0.7: {np.mean(g_7_counts)}")
    print(f"E2E 3 Node 1Km Verification Throughput > 0.9: {np.mean(g_9_counts)}")
    print(f"E2E 3 Node 1Km Verification Global > 0.7: {g_7_global_count / total_count}")
    print(f"E2E 3 Node 1Km Verification Global > 0.9: {g_9_global_count / total_count}\n")


    with open("./transportation_results/e2e_4nodes_1_qubit_purification_raw.json", 'r') as f:
        data = json.load(f)
    fid_data = data["teleport_fids"]
    g_7_count = 0
    g_9_count = 0
    d_data = data["duration"]
    for fid in fid_data:
        if fid > 0.7:
            g_7_count += 1
        if fid > 0.9:
            g_9_count += 1

    print(f"E2E 4 Node 1Km Purification Success Rate > 0.7: {g_7_count / len(fid_data)}")
    print(f"E2E 4 Node 1Km Purification Success Rate  > 0.9: {g_9_count / len(fid_data)}")
    print(f"E2E 4 Node 1Km Purification Success Rate Duration: {np.mean(d_data)}\n")


    with open("./transportation_results/e2e_4nodes_purification_throughput_raw.json", 'r') as f:
        data = json.load(f)
    g_7_counts = []
    g_9_counts = []
    g_7_global_count = 0
    g_9_global_count = 0
    total_count = 0
    all_count = []
    for value_data in data["4"].values():
        g_7_count = 0
        g_9_count = 0
        fid_data = value_data["all_fid"]
        all_count.append(len(fid_data))
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

    print(f"E2E 4 Node 1Km Throughput Count: {np.mean(all_count)}")
    print(f"E2E 4 Node 1Km Purification Throughput > 0.7: {np.mean(g_7_counts)}")
    print(f"E2E 4 Node 1Km Purification Throughput > 0.9: {np.mean(g_9_counts)}")
    print(f"E2E 4 Node 1Km Purification Global > 0.7: {g_7_global_count / total_count}")
    print(f"E2E 4 Node 1Km Purification Global > 0.9: {g_9_global_count / total_count}\n")

    with open("./transportation_results/e2e_4nodes_1_qubit_verification_1.0km_raw.json", 'r') as f:
        data = json.load(f)
    fid_data = data["teleport_fids"]
    g_7_count = 0
    g_9_count = 0
    d_data = data["duration"]
    for fid in fid_data:
        if fid > 0.7:
            g_7_count += 1
        if fid > 0.9:
            g_9_count += 1

    print(f"E2E 4 Node 1Km Verification Success Rate > 0.7: {g_7_count / len(fid_data)}")
    print(f"E2E 4 Node 1Km Verification Success Rate  > 0.9: {g_9_count / len(fid_data)}")
    print(f"E2E 4 Node 1Km Verification Success Rate Duration: {np.mean(d_data)}\n")


    with open("./transportation_results/e2e_4nodes_1_qubit_verification_0.5km_raw.json", 'r') as f:
        data = json.load(f)
    fid_data = data["teleport_fids"]
    g_7_count = 0
    g_9_count = 0
    d_data = data["duration"]
    for fid in fid_data:
        if fid > 0.7:
            g_7_count += 1
        if fid > 0.9:
            g_9_count += 1

    print(f"E2E 4 Node 0.5Km Verification Success Rate > 0.7: {g_7_count / len(fid_data)}")
    print(f"E2E 4 Node 0.5Km Verification Success Rate  > 0.9: {g_9_count / len(fid_data)}")
    print(f"E2E 4 Node 0.5Km Verification Success Rate Duration: {np.mean(d_data)}\n")


    with open("./transportation_results/e2e_4nodes_throughput_1km_verify_raw.json", 'r') as f:
        data = json.load(f)
    g_6_counts = []
    g_7_counts = []
    g_9_counts = []
    g_6_global_count = 0
    g_7_global_count = 0
    g_9_global_count = 0
    total_count = 0
    all_count = []
    for value_data in data["4"].values():
        g_7_count = 0
        g_9_count = 0
        g_6_count = 0
        fid_data = value_data["all_fids"]
        all_count.append(len(fid_data))
        for fid in fid_data:
            if fid > 0.7:
                g_7_count += 1
                g_7_global_count += 1
            if fid > 0.9:
                g_9_count += 1
                g_9_global_count += 1
            if fid > 0.65:
                g_6_count += 1
                g_6_global_count +=1
            total_count += 1
        if len(fid_data) > 0:
            g_7_counts.append(g_7_count/len(fid_data))
            g_9_counts.append(g_9_count/len(fid_data))
            g_6_counts.append(g_6_count/len(fid_data))

    print(f"E2E 4 Node 1Km Verification Throughput Count: {np.mean(all_count)}")
    print(f"E2E 4 Node 1Km Verification Throughput > 0.65: {np.mean(g_6_counts)}")
    print(f"E2E 4 Node 1Km Verification Throughput > 0.7: {np.mean(g_7_counts)}")
    print(f"E2E 4 Node 1Km Verification Throughput > 0.9: {np.mean(g_9_counts)}")
    print(f"E2E 4 Node 1Km Verification Global > 0.7: {g_6_global_count / total_count}")
    print(f"E2E 4 Node 1Km Verification Global > 0.7: {g_7_global_count / total_count}")
    print(f"E2E 4 Node 1Km Verification Global > 0.9: {g_9_global_count / total_count}\n")