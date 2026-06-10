#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/UPSM_9stg.yaml}
CHECKPOINT=${2:-checkpoints/UPSM_9stg.pth}
DEVICE=${3:-cuda:0}

python tools/test_simu.py \
  --config-file "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --data-path data/TSA_simu_data/Truth \
  --mask-path data/TSA_simu_data/mask_3d_shift.mat \
  --device "$DEVICE" \
  --output-mat outputs/simu_results.mat
