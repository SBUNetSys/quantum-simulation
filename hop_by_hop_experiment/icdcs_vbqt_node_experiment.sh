# Configuration
NODE_COUNT=4
DEPOLAR_RATE=8641
DISTANCE=1.0
TARGET_FID=0.0

echo "Starting node experiments at distance ${DISTANCE}km"
echo "Node count: ${NODE_COUNT}"
echo "Depolar rate: ${DEPOLAR_RATE}"
echo "Target fidelity: ${TARGET_FID}"
echo "With verification, no purification"
echo "----------------------------------------"

python -u sim_vbqt_transportation_concurrent.py \
      --distance $distance \
      --target-fid $TARGET_FID \
      --depolar-rate $DEPOLAR_RATE \
      --with-verification \
      --node-count $NODE_COUNT \
      --batch-size 4 \
      --total-runs 1000 \
      --preload

  if [ $? -eq 0 ]; then
      echo "Successfully completed experiment for ${distance}km"
  else
      echo "Error occurred for ${distance}km"
  fi

  echo "----------------------------------------"


echo "All experiments completed"
