# UPSM

Training, evaluation and complexity profiling code for UPSM.

## Install

Install PyTorch for your CUDA environment first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data And Checkpoints

Use the data layout in [docs/DATA.md](docs/DATA.md). Put pretrained weights in:

```text
checkpoints/UPSM_6stg.pth
checkpoints/UPSM_9stg.pth
```

See [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) for checkpoint loading notes.

## ICVL Test

```bash
python tools/test_icvl.py \
  --config-file configs/UPSM_9stg.yaml \
  --checkpoint checkpoints/UPSM_9stg.pth \
  --icvl-path data/ICVL \
  --mask-path data/TSA_simu_data/mask_3d_shift.mat \
  --device cuda:0 \
  --output-txt outputs/icvl_psnr_9stg.txt
```

## TSA Simulation Test

```bash
python tools/test_simu.py \
  --config-file configs/UPSM_9stg.yaml \
  --checkpoint checkpoints/UPSM_9stg.pth \
  --data-path data/TSA_simu_data/Truth \
  --mask-path data/TSA_simu_data/mask_3d_shift.mat \
  --device cuda:0 \
  --output-mat outputs/simu_results_9stg.mat
```

## Train

```bash
python tools/train.py \
  --config-file configs/UPSM_6stg.yaml \
  --device cuda:0
```

## Parameters and FLOPs

```bash
python tools/profile_complexity.py --stages 9
```

No dataset or checkpoint is required. See [the complexity guide](docs/COMPLEXITY.md)
for counting conventions, expected results and MiJUN usage.
