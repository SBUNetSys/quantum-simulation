import json
import sys

import numpy as np

if __name__ == '__main__':
    file_path = sys.argv[1]
    with open(file_path, "r") as f:
        results = json.load(f)
    fid_data = []
    nan_count = 0
    duration_data = []
    for run, val in results.items():
        fid_val = val["teleport_fids"][0]
        if fid_val is None or np.isnan(fid_val):
            nan_count += 1
        else:
            fid_data.append(fid_val)
        duration_data.append(val["duration"][0])
    print(f"Average Fidelity: {sum(fid_data)/len(fid_data)}")
    for _ in range(nan_count):
        fid_data.append(0)
    print(f"Average Fidelity with 0 for NaN: {sum(fid_data)/len(fid_data)}")
    print(f"NaN count: {nan_count}")
    print(f"Average Duration: {sum(duration_data)/len(duration_data)}")