import os
import sys

import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, *([".."] * 2)))
sys.path.insert(0, PROJECT_ROOT)

from sci.architectures import UPSM
from sci.config import get_cfg
from sci.data import LoadVal, generate_mask_3d_shift, gen_meas_torch
from sci.engine import default_argument_parser
from sci.metrics import sam, torch_psnr, torch_ssim
from sci.utils.checkpoint import load_checkpoint_file


def build_parser():
    parser = default_argument_parser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None, help="Path to TSA_simu_data/Truth.")
    parser.add_argument("--mask-path", type=str, default=None, help="Path to mask_3d_shift.mat.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-mat", type=str, default="outputs/simu_results.mat")
    return parser


def main():
    args = build_parser().parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or cfg.PRETRAINED_CKPT_PATH
    data_path = args.data_path or cfg.DATASETS.VAL.PATH
    mask_path = args.mask_path or cfg.DATASETS.VAL.MASK_PATH
    if not checkpoint_path:
        raise ValueError("Please provide --checkpoint or PRETRAINED_CKPT_PATH in the config.")

    mask = generate_mask_3d_shift(mask_path=mask_path).to(device)
    val_datas = LoadVal(data_path)

    model = UPSM(cfg).to(device)
    print(f"===> Loading Checkpoint from {checkpoint_path}")
    load_checkpoint_file(model, checkpoint_path, device)
    model.eval()

    psnr_list, ssim_list, sam_list = [], [], []
    pred_list, truth_list = [], []

    with torch.no_grad():
        for val_label in tqdm(val_datas["hsi"], desc="Testing TSA simulation"):
            gt = torch.from_numpy(val_label).permute(2, 0, 1).to(device).float()
            yh = gen_meas_torch(
                gt,
                mask,
                step=cfg.DATASETS.STEP,
                wave_len=cfg.DATASETS.WAVE_LENS,
                mask_type=cfg.DATASETS.MASK_TYPE,
            )
            data = {
                "hsi": gt.unsqueeze(0),
                "H": yh["H"].unsqueeze(0).to(device),
                "Y": yh["Y"].unsqueeze(0).to(device),
                "mask": mask.unsqueeze(0),
            }
            pred = model(data)[0].clamp(0.0, 1.0)

            psnr_list.append(torch_psnr(pred, gt).detach().cpu().numpy())
            ssim_list.append(torch_ssim(pred, gt).detach().cpu().numpy())
            sam_list.append(sam(pred.permute(1, 2, 0).cpu().numpy(), gt.permute(1, 2, 0).cpu().numpy()))
            pred_list.append(np.transpose(pred.detach().cpu().numpy(), (1, 2, 0)).astype(np.float32))
            truth_list.append(np.transpose(gt.detach().cpu().numpy(), (1, 2, 0)).astype(np.float32))

    psnr_mean = float(np.mean(np.asarray(psnr_list)))
    ssim_mean = float(np.mean(np.asarray(ssim_list)))
    sam_mean = float(np.mean(np.asarray(sam_list)))
    print(f"===> testing psnr = {psnr_mean:.2f}, ssim = {ssim_mean:.3f}, sam = {sam_mean:.3f}")

    output_dir = os.path.dirname(args.output_mat)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sio.savemat(
        args.output_mat,
        {
            "pred": np.stack(pred_list, axis=0),
            "truth": np.stack(truth_list, axis=0),
            "psnr": np.asarray(psnr_list),
            "ssim": np.asarray(ssim_list),
            "sam": np.asarray(sam_list),
        },
    )
    print(f"Saved results to: {args.output_mat}")


if __name__ == "__main__":
    main()
