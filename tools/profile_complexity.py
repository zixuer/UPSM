#!/usr/bin/env python3
"""Measure complete parameters and expose operation-count conventions explicitly.

No dataset or trained weights required. See docs/COMPLEXITY.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from fvcore.nn import FlopCountAnalysis

SCAN_OPS = (
    "SelectiveScanFn", "SelectiveScanStateFn", "SelectiveScanMamba",
    "SelectiveScanOflex", "SelectiveScanCore", "SelectiveScan", "SelectiveScanCuda",
)


def scan_handler(inputs, outputs):
    batch, dim, length = inputs[0].type().sizes()
    state = inputs[2].type().sizes()[1]
    return {"selective_scan_core": 6 * batch * length * dim * state,
            "selective_scan_D": batch * length * dim}


def historical_inner_handler(inputs, outputs):
    batch, twice_dim, length = inputs[0].type().sizes()
    state = inputs[6].type().sizes()[1]
    return 6 * batch * length * (twice_dim // 2) * state


def git(root, *args):
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["upsm", "mijun"], default="upsm")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", help="Relative to model repository")
    parser.add_argument("--stages", type=int, default=9)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-thop", action="store_true", help="Also audit the historical THOP + scan convention")
    parser.add_argument("--check-paper", action="store_true", help="Exit nonzero if measured costs disagree with the paper beyond 0.01 G")
    args = parser.parse_args()
    if args.stages < 1:
        parser.error("--stages must be positive")
    root = args.repo_root.resolve()
    output = args.output.resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    defaults = {
        "upsm": ("sci", "UPSM", "DERNN_LNLT", "configs/UPSM_9stg.yaml"),
        "mijun": ("sci", "MiJUN", "DERNN_LNLT", "configs/MiJUN_5stg.yaml"),
    }
    package, cls, node, default_config = defaults[args.model]
    # Official MiJUN contains a csi.data import although its package is sci.
    # Alias the existing package in memory; no third-party file is edited.
    if args.model == "mijun":
        sys.modules["csi"] = importlib.import_module("sci")
        sys.modules["csi.data"] = importlib.import_module("sci.data")
    cfg = importlib.import_module(package + ".config").get_cfg()
    config = root / (args.config or default_config)
    cfg.merge_from_file(str(config))
    cfg.defrost()
    getattr(cfg.MODEL.DENOISER, node).STAGE = args.stages
    cfg.freeze()
    torch.manual_seed(3407)
    model_class = getattr(importlib.import_module(package + ".architectures"), cls)
    model = model_class(cfg).to(args.device).eval()
    sample = {
        "Y": torch.ones(1, 256, 310, device=args.device),
        "mask": torch.ones(1, 28, 256, 310, device=args.device),
        "H": torch.ones(1, 28, 256, 256, device=args.device),
    }
    params = sum(p.numel() for p in model.parameters())
    fast_modules = [m for m in model.modules() if hasattr(m, "use_fast_path")]
    with torch.no_grad():
        before = model(sample)
        historical_fused = None
        if fast_modules:
            historical = FlopCountAnalysis(model, (sample,))
            historical.set_op_handle(**{"prim::PythonOp.MambaInnerFn": historical_inner_handler})
            historical.tracer_warnings("none").unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            historical_fused = {"gflops": historical.total() / 1e9,
                                "by_operator": dict(historical.by_operator()),
                                "unsupported": dict(historical.unsupported_ops())}
        for m in fast_modules:
            m.use_fast_path = False
        after = model(sample)
        # Outputs may be dicts in some forks; require a tensor rather than
        # silently claiming equivalence without checking it.
        torch.testing.assert_close(after, before, rtol=1e-3, atol=1e-4)
        max_error = float((after - before).abs().max())
        del before, after
        analyzer = FlopCountAnalysis(model, (sample,))
        analyzer.set_op_handle(**{"prim::PythonOp." + name: scan_handler for name in SCAN_OPS})
        analyzer.set_op_handle(**{op: None for op in ["aten::silu", "aten::neg", "aten::exp", "aten::flip"]})
        analyzer.tracer_warnings("none").unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        total = int(analyzer.total())
        ops = dict(analyzer.by_operator())
        unsupported = dict(analyzer.unsupported_ops())
        if not ops.get("selective_scan_core"):
            raise RuntimeError("No selective-scan calls counted; refusing an incomplete result")
        if any("MambaInner" in op or "SelectiveScan" in op for op in unsupported):
            raise RuntimeError("Unrecognized fused/scan operator: " + str(unsupported))
        legacy = None
        if args.legacy_thop:
            from thop import profile
            standard, reported_params = profile(model, inputs=(sample,), verbose=False)
            legacy = {"thop_visible_gflops": standard / 1e9,
                      "thop_reported_params": int(reported_params),
                      "plus_scan_core_gflops": (standard + ops["selective_scan_core"]) / 1e9}
    result = {
        "model": args.model, "stages": args.stages, "params": params, "params_m": params / 1e6,
        "fvcore_gflops": total / 1e9, "by_operator": ops,
        "unsupported_ops_excluded": unsupported, "legacy_thop": legacy,
        "historical_fused_scan_only_diagnostic": historical_fused,
        "fast_path_modules_disabled": len(fast_modules), "fast_reference_max_abs_error": max_error,
        "input_shapes": {k: list(v.shape) for k, v in sample.items()},
        "config": str(config.relative_to(root)), "resolved_config": cfg.dump(),
        "git_commit": git(root, "rev-parse", "HEAD"),
        "tracked_diff_sha256": hashlib.sha256((git(root, "diff", "HEAD") or "").encode()).hexdigest(),
        "environment": {name: version(name) for name in ["torch", "fvcore", "thop", "mamba-ssm", "causal-conv1d", "timm", "einops"]},
        "python": sys.version.split()[0], "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device),
    }
    result["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in [root / package / "architectures", root / package / "data", root / package / "config"]
        for path in sorted(directory.rglob("*.py"))
    }
    result["profiler_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    paper = {"upsm": {3: 41.26, 6: 82.76, 9: 124.27}, "mijun": {3: 56.43, 5: 93.96, 9: 169.03}}
    target = paper[args.model].get(args.stages)
    result["paper_gflops"] = target
    result["paper_agreement_within_0_01G"] = target is not None and abs(total / 1e9 - target) <= 0.01
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["model", "stages", "params_m", "fvcore_gflops", "legacy_thop", "unsupported_ops_excluded"]}, indent=2))
    if args.check_paper and not result["paper_agreement_within_0_01G"]:
        raise SystemExit("Paper FLOPs were NOT reproduced. See the saved diagnostic JSON and docs/COMPLEXITY.md.")


if __name__ == "__main__":
    main()
