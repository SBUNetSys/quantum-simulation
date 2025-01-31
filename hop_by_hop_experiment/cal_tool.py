import json

import numpy as np

if __name__ == '__main__':
    with open("./transportation_results/3nodes_1_qubit_verification_1.0km_24583hz_raw.json", 'r') as f:
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

    print(f"3 Node 1Km Verification Depolar 24583Hz Success Rate > 0.7: {g_7_count / len(fid_data)}")
    print(f"3 Node 1Km Verification Depolar 24583Hz Success Rate  > 0.9: {g_9_count / len(fid_data)}")
    print(f"3 Node 1Km Verification Depolar 24583Hz Success Rate Duration: {np.mean(d_data)}\n")


    with open("./transportation_results/3nodes_1_qubit_verification_1.0km_6492hz_raw.json", 'r') as f:
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

    print(f"3 Node 1Km Verification Depolar 6492Hz Success Rate > 0.7: {g_7_count / len(fid_data)}")
    print(f"3 Node 1Km Verification Depolar 6492Hz Success Rate  > 0.9: {g_9_count / len(fid_data)}")
    print(f"3 Node 1Km Verification Depolar 6492Hz Success Rate Duration: {np.mean(d_data)}\n")


    with open("./transportation_results/4nodes_1_qubit_verification_1.0km_24583hz_raw.json", 'r') as f:
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

    print(f"4 Node 1Km Verification Depolar 24583Hz Success Rate > 0.7: {g_7_count / len(fid_data)}")
    print(f"4 Node 1Km Verification Depolar 24583Hz Success Rate  > 0.9: {g_9_count / len(fid_data)}")
    print(f"4 Node 1Km Verification Depolar 24583Hz Success Rate Duration: {np.mean(d_data)}\n")


    with open("./transportation_results/4nodes_1_qubit_verification_1.0km_6492hz_raw.json", 'r') as f:
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

    print(f"4 Node 1Km Verification Depolar 6492Hz Success Rate > 0.7: {g_7_count / len(fid_data)}")
    print(f"4 Node 1Km Verification Depolar 6492Hz Success Rate  > 0.9: {g_9_count / len(fid_data)}")
    print(f"4 Node 1Km Verification Depolar 6492Hz Success Rate Duration: {np.mean(d_data)}\n")