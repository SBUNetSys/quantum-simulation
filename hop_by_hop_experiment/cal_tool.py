import json

import numpy as np

if __name__ == '__main__':
    with open("./transportation_results/5nodes_throughput_raw_4_node.json", "r") as file:
        data = json.load(file)
    run_data = data["4"]

    avg_count = []
    avg_fid = []
    avg_success = []
    for v in run_data.values():
        avg_count.append(v["total_count"])
        avg_fid.append(v["average_fidelity"])
        avg_success.append(v["teleport_success_count"])
    print(f"Avg count: {np.mean(avg_count)}")
    print(f"Avg fidelity: {np.mean(avg_fid)}")
    print(f"Avg success: {np.mean(avg_success)}")
