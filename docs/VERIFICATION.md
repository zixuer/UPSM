# Verification

ICVL consistency was checked against the original repository on 201 `.mat` files from ICVL using `cuda:0`.

| Model | Original Avg PSNR | Open Avg PSNR | Avg Diff | Max Sample Diff |
| --- | ---: | ---: | ---: | ---: |
| UPSM_6stg | 38.5043 | 38.5043 | 0.0000 | 0.0000 |
| UPSM_9stg | 39.0446 | 39.0446 | 0.0000 | 0.0000 |

For both checkpoints, file order matched exactly and all 201 per-sample PSNR values matched to the saved precision.
