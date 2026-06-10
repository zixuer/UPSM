# Data

Place datasets under the repository root using this layout:

```text
data/
  CAVE_1024_28/
  TSA_simu_data/
    Truth/
    mask_3d_shift.mat
  TSA_real_data/
    Measurements/
    mask_3d_shift.mat
  ICVL/
```

`TSA_simu_data/Truth` is used for simulated validation/testing. `ICVL` should contain the original `.mat` files with key `rad`; `tools/test_icvl.py` center-crops, normalizes, and resamples channels before evaluation.
