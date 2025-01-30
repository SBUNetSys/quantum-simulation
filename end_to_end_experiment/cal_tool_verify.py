import json
from symbol import flow_stmt

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
    print(f"average fid: {np.mean(all_fid)}")


