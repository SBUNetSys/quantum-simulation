#!/bin/bash
while true; do
    python check_security.py $1
    if [ $? -eq 0 ]; then
        break
    fi
    python sim_security.py $1
done