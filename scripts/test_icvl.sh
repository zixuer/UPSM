#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/UPSM_9stg.yaml}
CHECKPOINT=${2:-checkpoints/UPSM_9stg.pth}
DEVICE=${3:-cuda:0}

python tools/test_icvl.py \
  --config-file "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --icvl-path data/ICVL \
  --mask-path data/TSA_simu_data/mask_3d_shift.mat \
  --device "$DEVICE" \
  --output-txt outputs/icvl_psnr.txt
