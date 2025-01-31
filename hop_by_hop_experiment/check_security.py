import json
import os.path
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: python check_security.py opt')
    else:
        opt = int(sys.argv[1])
        if opt == 1:
            if os.path.exists("./security_results/3nodes_0.5km_security_raw.json"):
                with open("./security_results/3nodes_0.5km_security_raw.json") as f:
                    data = json.load(f)
                    if len(data["teleport_success_count"]) != 1000:
                        exit(-1)
                    else:
                        exit(0)
            else:
                exit(-1)
        elif opt == 2:
            if os.path.exists("./security_results/3nodes_1.0km_security_raw.json"):
                with open("./security_results/3nodes_1.0km_security_raw.json") as f:
                    data = json.load(f)
                    if len(data["teleport_success_count"]) != 1000:
                        exit(-1)
                    else:
                        exit(0)
            else:
                exit(-1)