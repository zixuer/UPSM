#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/UPSM_6stg.yaml}
DEVICE=${2:-cuda:0}

python tools/train.py \
  --config-file "$CONFIG" \
  --device "$DEVICE"
