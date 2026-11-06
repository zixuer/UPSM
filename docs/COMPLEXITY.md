# Parameters and FLOPs

Run from the repository root. No dataset or checkpoint is required:

```bash
python tools/profile_complexity.py --stages 9 --device cuda:0
```

Change `--stages` to `3` or `6` for the other UPSM variants.

| Stages | Params (M) | GFLOPs |
|---|---:|---:|
| 3 | 1.907066 | 41.266486 |
| 6 | 1.907066 | 82.769115 |
| 9 | 1.907066 | 124.271744 |

## Counting convention

The input is FP32, batch size 1, with a reconstructed cube of 256 x 256 x 28.
The shifted measurement width is 310. Shared parameters are counted once using
`sum(p.numel() for p in model.parameters())`.

UPSM uses fvcore with one multiply-accumulate counted as one operation. Selective
scan contributes `6BLDN + BLD`, including the D-skip term. Other operations use
fvcore's default handlers; unsupported operations, such as sorting, elementwise
arithmetic and reductions, are excluded. FLOPs are a profiler estimate, not latency.
The paper's 3/6-stage display values use truncation to two decimal places.

Tested with Python 3.9, PyTorch 2.5.1 (CUDA 11.8), mamba-ssm 1.2.0,
causal-conv1d 1.2.0, fvcore 0.1.5.post20221221, timm 1.0.22 and einops 0.8.1.
Install CUDA extensions compatible with your PyTorch build.

## MiJUN

The script also supports the official MiJUN implementation:

```bash
git clone https://github.com/Mengjie-s/MiJUN.git ../MiJUN
git -C ../MiJUN checkout be916bef9086c90504d4896bcba52f6a3f45e70e
pip install python-box thop==0.1.1.post2209072238
python tools/profile_complexity.py --model mijun --repo-root ../MiJUN --stages 9
```

MiJUN uses THOP-visible operations with the fused Mamba path disabled, plus the
`6BLDN` scan-core term. This produces **1.892204 M / 168.193741 G** in the tested
environment, compared with **1.89 M / 169.03 G** in the paper. These counting
conventions are stated separately; the script does not force agreement with
published values or claim complete instruction-level accounting.
