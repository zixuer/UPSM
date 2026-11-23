"""Count parameters and FLOPs for a 256 x 256 x 28 input."""
import argparse
import importlib
from pathlib import Path
import sys

import torch
from thop import profile
from fvcore.nn import FlopCountAnalysis


def selective_scan_flops(inputs, outputs):
    batch, channels, length = inputs[0].type().sizes()
    state = inputs[2].type().sizes()[1]
    return {"scan_core": 6 * batch * length * channels * state}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["upsm", "mijun"], help="Optional model override (inferred from the repository by default)")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", help="Config path relative to the model repository")
    parser.add_argument("--stages", type=int, default=9)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stages < 1:
        parser.error("--stages must be positive")
    sys.path.insert(0, str(args.repo_root.resolve()))

    model_name = args.model
    if model_name is None:
        models = [name for name in ("UPSM", "MiJUN")
                  if (args.repo_root / "sci" / "architectures" / f"{name}.py").is_file()]
        if len(models) != 1:
            parser.error("Provide a UPSM or MiJUN repository, or specify --model.")
        model_name = models[0].lower()

    if model_name == "mijun":
        # Upstream MiJUN imports csi.data, but names its package sci.
        sys.modules["csi"] = importlib.import_module("sci")
        sys.modules["csi.data"] = importlib.import_module("sci.data")
    from sci.config import get_cfg
    from sci import architectures

    config = args.config or ("configs/UPSM_9stg.yaml" if model_name == "upsm"
                             else "configs/MiJUN_5stg.yaml")
    cfg = get_cfg()
    cfg.merge_from_file(str(args.repo_root / config))
    cfg.MODEL.DENOISER.DERNN_LNLT.STAGE = args.stages
    cfg.freeze()
    model_class = architectures.UPSM if model_name == "upsm" else architectures.MiJUN
    model = model_class(cfg).to(args.device).eval()
    for module in model.modules():
        if hasattr(module, "use_fast_path"):
            module.use_fast_path = False
    sample = {
        "Y": torch.ones(1, 256, 310, device=args.device),
        "mask": torch.ones(1, 28, 256, 310, device=args.device),
        "H": torch.ones(1, 28, 256, 256, device=args.device),
    }
    with torch.no_grad():
        analyzer = FlopCountAnalysis(model, (sample,))
        analyzer.set_op_handle("prim::PythonOp.SelectiveScanFn", selective_scan_flops)
        analyzer.tracer_warnings("none").unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        scan_core = analyzer.by_operator()["scan_core"]
        if not scan_core:
            raise RuntimeError("Selective scan was not counted; check your Mamba version.")
        standard_ops, _ = profile(model, inputs=(sample,), verbose=False)
        flops = standard_ops + scan_core
    params = sum(p.numel() for p in model.parameters())
    print(f"{model_name.upper()}-{args.stages}stg")
    print(f"Params: {params / 1e6:.6f} M")
    print(f"FLOPs:  {flops / 1e9:.6f} G")


if __name__ == "__main__":
    main()
