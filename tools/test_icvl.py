import os
import sys
from glob import glob

import h5py
import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, *([".."] * 2)))
sys.path.insert(0, PROJECT_ROOT)

from sci.architectures import UPSM
from sci.config import get_cfg
from sci.data import generate_mask_3d_shift, gen_meas_torch
from sci.engine import default_argument_parser
from sci.metrics import torch_psnr
from sci.utils.checkpoint import load_checkpoint_file


def compute_weights(source_lambdas, target_lambdas):
    dist_matrix = torch.abs(target_lambdas.unsqueeze(1) - source_lambdas.unsqueeze(0))
    weights = torch.exp(-dist_matrix) ** 4
    weights = weights / weights.sum(dim=1, keepdim=True)
    return weights


def resample_channels(data_tensor, source_range=(400, 700, 31), target_range=(450, 650, 28)):
    source_lambdas = torch.linspace(
        source_range[0], source_range[1], source_range[2], device=data_tensor.device
    )
    target_lambdas = torch.linspace(
        target_range[0], target_range[1], target_range[2], device=data_tensor.device
    )
    weights = compute_weights(source_lambdas, target_lambdas)
    return torch.einsum("ij,jhw->ihw", weights, data_tensor)


def center_crop_chw(cube, crop_size=256):
    _, h, w = cube.shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"Input spatial size ({h}, {w}) is smaller than crop size {crop_size}.")
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return cube[:, top : top + crop_size, left : left + crop_size]


def get_icvl_mat_paths(data_dir):
    mat_paths = sorted(glob(os.path.join(data_dir, "*.mat")))
    if not mat_paths:
        raise FileNotFoundError(f"No .mat files found in: {data_dir}")
    return mat_paths


def load_selected_names(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return {name if name.endswith(".mat") else f"{name}.mat" for name in names}


def load_single_icvl_sample(mat_path):
    with h5py.File(mat_path, "r") as f:
        data = f["rad"][:]

    hsi = torch.tensor(data, dtype=torch.float32)
    hsi = hsi.flip(dims=[1])
    hsi = center_crop_chw(hsi, crop_size=256)

    upper_quantile = np.percentile(hsi.cpu().numpy(), 99.5)
    hsi = hsi / (upper_quantile * 1.4)

    hsi = resample_channels(hsi)
    return hsi.clamp(0.0, 1.0)


def unwrap_model_output(output):
    if isinstance(output, (list, tuple)):
        return output[0]
    if isinstance(output, dict):
        for key in ("pred", "out", "hsi"):
            if key in output:
                return output[key]
        raise KeyError("Cannot find prediction tensor in model output dict.")
    return output


def build_parser():
    parser = default_argument_parser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path.")
    parser.add_argument("--icvl-path", type=str, default=None, help="Path to ICVL .mat files.")
    parser.add_argument("--mask-path", type=str, default=None, help="Path to mask_3d_shift.mat.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--output-txt", type=str, default="outputs/icvl_psnr.txt")
    parser.add_argument("--save-mat-dir", type=str, default=None)
    parser.add_argument("--save-gt-dir", type=str, default=None)
    parser.add_argument("--selected-names", type=str, default=None)
    return parser


def main():
    args = build_parser().parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or cfg.PRETRAINED_CKPT_PATH
    icvl_path = args.icvl_path or cfg.DATASETS.VAL.PATH
    mask_path = args.mask_path or cfg.DATASETS.VAL.MASK_PATH
    if not checkpoint_path:
        raise ValueError("Please provide --checkpoint or PRETRAINED_CKPT_PATH in the config.")
    if not icvl_path:
        raise ValueError("Please provide --icvl-path or DATASETS.VAL.PATH in the config.")
    if not mask_path:
        raise ValueError("Please provide --mask-path or DATASETS.VAL.MASK_PATH in the config.")

    mask = generate_mask_3d_shift(mask_path=mask_path).to(device)
    mask_batch = mask.unsqueeze(0)

    model = UPSM(cfg).to(device)
    print(f"===> Loading Checkpoint from {checkpoint_path}")
    load_info = load_checkpoint_file(model, checkpoint_path, device)
    if load_info["missing_keys"]:
        print("[WARN] Checkpoint has missing keys for the current model.")
    model.eval()

    output_dir = os.path.dirname(args.output_txt)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if args.save_mat_dir:
        os.makedirs(args.save_mat_dir, exist_ok=True)
    if args.save_gt_dir:
        os.makedirs(args.save_gt_dir, exist_ok=True)

    mat_paths = get_icvl_mat_paths(icvl_path)
    selected_names = load_selected_names(args.selected_names)
    if selected_names is not None:
        mat_paths = [p for p in mat_paths if os.path.basename(p) in selected_names]
        missing = selected_names - {os.path.basename(p) for p in mat_paths}
        if missing:
            raise FileNotFoundError(f"Missing selected ICVL files: {sorted(missing)}")

    psnr_results = []
    with open(args.output_txt, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for mat_path in tqdm(mat_paths, desc="Testing ICVL"):
                gt = load_single_icvl_sample(mat_path).to(device)
                yh = gen_meas_torch(
                    gt,
                    mask,
                    step=cfg.DATASETS.STEP,
                    wave_len=cfg.DATASETS.WAVE_LENS,
                    mask_type=cfg.DATASETS.MASK_TYPE,
                )
                yh = {k: v.to(device) if torch.is_tensor(v) else v for k, v in yh.items()}
                data = {
                    "hsi": gt.unsqueeze(0),
                    "H": yh["H"].unsqueeze(0),
                    "Y": yh["Y"].unsqueeze(0),
                    "mask": mask_batch,
                }

                pred = unwrap_model_output(model(data)).clamp(0.0, 1.0)
                psnr_val = torch_psnr(pred[0], gt).item()
                file_name = os.path.basename(mat_path)
                line = f"{file_name}: {psnr_val:.4f}"
                print(line)
                f.write(line + "\n")
                psnr_results.append(psnr_val)

                stem = os.path.splitext(file_name)[0]
                if args.save_mat_dir:
                    sio.savemat(
                        os.path.join(args.save_mat_dir, f"{stem}.mat"),
                        {"hsi": pred[0].detach().cpu().permute(1, 2, 0).numpy()},
                    )
                if args.save_gt_dir:
                    sio.savemat(
                        os.path.join(args.save_gt_dir, f"{stem}.mat"),
                        {"gt": gt.detach().cpu().permute(1, 2, 0).numpy()},
                    )

        avg_psnr = float(np.mean(psnr_results))
        avg_line = f"Average PSNR: {avg_psnr:.4f}"
        print(avg_line)
        f.write(avg_line + "\n")

    print(f"Evaluated files: {len(psnr_results)}")
    print(f"Saved PSNR results to: {args.output_txt}")


if __name__ == "__main__":
    main()
