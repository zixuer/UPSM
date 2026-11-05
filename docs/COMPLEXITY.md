# Reproducing parameter and operation counts

`tools/profile_complexity.py` profiles the actual model at batch size 1 and
256 x 256 x 28 reconstruction size. It needs a CUDA GPU but **no checkpoint,
dataset, or mask file**. Inputs are synthetic FP32 tensors: measurement
`[1,256,310]`, shifted mask `[1,28,256,310]`, and cube `[1,28,256,256]`.

## Verified UPSM results

Run from the repository root:

```bash
python tools/profile_complexity.py --stages 3 --output outputs/upsm3.json --check-paper
python tools/profile_complexity.py --stages 6 --output outputs/upsm6.json --check-paper
python tools/profile_complexity.py --stages 9 --output outputs/upsm9.json --check-paper
```

Use `--device cuda:1` to choose another GPU. The default config is
`configs/UPSM_9stg.yaml`; only its stage count is overridden in memory.

Measured on an RTX 4090:

| Stages | Parameters | Params (M) | Measured GFLOPs | Paper display |
|---|---:|---:|---:|---:|
| 3 | 1,907,066 | 1.907066 | 41.266486352 | 41.26 |
| 6 | 1,907,066 | 1.907066 | 82.769115296 | 82.76 |
| 9 | 1,907,066 | 1.907066 | 124.271744240 | 124.27 |

The 3/6-stage paper entries correspond to truncation, not round-to-nearest.
Rounded normally, the results are **41.27 / 82.77 / 124.27**. The JSON always
preserves the measured values. `--check-paper` permits at most 0.01 G for this
display difference; target values are used only for validation, never counting.

## Counting convention and scope

- Parameters are `sum(p.numel() for p in model.parameters())`, including SSM
  tensors and counting shared parameters once. THOP's parameter field is not used.
- The UPSM reproduction uses **fvcore**, matching the original UPSM profiling
  code. One multiply-accumulate is counted as one operation.
- Each selective-scan call contributes `6 * B * L * D_inner * N + B * L * D_inner`.
  The recurrence and D-skip contributions are reported separately. This is the
  convention of the original UPSM script; it includes a D-skip term in addition
  to the `6BLDN` formula printed in the supplement.
- Standard operations use fvcore's registered handlers. Unsupported operations
  are excluded and listed in the JSON, including elementwise arithmetic,
  reductions, sorting, and routing. These are profiler estimates, not a complete
  count of every executed GPU instruction or a measurement of runtime.
- The script records resolved configuration, source hashes, profiler hash,
  package versions, GPU, Git revision, operator breakdown and excluded operators.
  It fails if no scan is counted or an unrecognized scan/fused-Mamba op remains.

## Tested environment

Python 3.9.20, PyTorch 2.5.1 with CUDA 11.8, mamba-ssm 1.2.0,
causal-conv1d 1.2.0, fvcore 0.1.5.post20221221, timm 1.0.22, einops 0.8.1.
Install a matching CUDA-enabled PyTorch first, then the repository dependencies.
For the tested profiling versions:

```bash
pip install fvcore==0.1.5.post20221221 timm==1.0.22 einops==0.8.1
pip install --no-build-isolation causal-conv1d==1.2.0 mamba-ssm==1.2.0
```

CUDA extensions must match the installed PyTorch/CUDA ABI. Newer Mamba versions
are not claimed to reproduce these results.

## MiJUN diagnostic (not a verified reproduction of its paper-table FLOPs)

The same entry point also audits MiJUN at upstream revision
`be916bef9086c90504d4896bcba52f6a3f45e70e`:

```bash
git clone https://github.com/Mengjie-s/MiJUN.git ../MiJUN
git -C ../MiJUN checkout be916bef9086c90504d4896bcba52f6a3f45e70e
pip install python-box thop==0.1.1.post2209072238
python tools/profile_complexity.py --model mijun --repo-root ../MiJUN \
  --stages 9 --legacy-thop --output outputs/mijun9.json
```

The upstream `csi.data` import is aliased to its actual `sci.data` package in
memory, without changing upstream model files. The script disables standard
Mamba's fused path to expose internal operations, checks its output against the
original path (`rtol=1e-3`, `atol=1e-4`), and reports the counting variants
separately. This single synthetic-input check is not a dataset accuracy test.

MiJUN-9stg's complete parameter count is **1,892,204 (1.892204 M)**. The historical
THOP-visible cost is **137.277526568 G** and the scan-core contribution is
**30.916214784 G**, yielding **168.193741352 G**, not the paper's **169.03 G**.
The non-fused fvcore estimate is **264.592216112 G** and still excludes the custom
causal-convolution operator; it is not an interchangeable replacement for the
THOP-based estimate. The fused-path diagnostic counts only the scan core inside
`MambaInnerFn`, omitting internal non-scan work; it is not a complete cost either.

Thus, **this release does not claim to reproduce MiJUN's 169.03 G or establish
a fully unified complexity table for all baselines**. `--check-paper` deliberately
fails for that MiJUN result. No correction constant is added to force agreement.
SCM-DUN and SCI-SSM are not supported release targets in this script.

The supplement's THOP description and its MiJUN scan subtotal require
reconciliation with the original profiling provenance before this repository
can be cited as reproducing the entire comparison table.
