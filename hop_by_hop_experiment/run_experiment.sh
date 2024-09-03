# python sim_purification.py true
# echo "Purification done"
echo "Starting verification simulation, skip noise = true"
python sim_verification.py true
echo "Verification, skip noise = true done, starting verification simulation, skip noise = false"
python sim_verification.py false 