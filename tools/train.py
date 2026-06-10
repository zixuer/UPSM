import os
import sys
import time

import numpy as np
import scipy.io as sio
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, *([".."] * 2)))
sys.path.insert(0, PROJECT_ROOT)

from sci.architectures import UPSM
from sci.config import get_cfg
from sci.data import CSITrainDataset, LoadVal, generate_mask_3d_shift, gen_meas_torch
from sci.engine import default_argument_parser, default_setup
from sci.losses import CharbonnierLoss, TVLoss
from sci.metrics import sam, torch_psnr, torch_ssim
from sci.utils.checkpoint import load_compatible_state_dict
from sci.utils.schedulers import get_cosine_schedule_with_warmup
from sci.utils.utils import checkpoint


def build_parser():
    parser = default_argument_parser()
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device, e.g. cuda:0 or cpu.")
    return parser


def evaluate(cfg, model, ema, val_datas, mask, device):
    psnr_list, ssim_list, sam_list = [], [], []
    pred_list, truth_list = [], []

    model.eval()
    with torch.no_grad():
        with ema.average_parameters():
            for val_label in val_datas["hsi"]:
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
                sam_list.append(
                    sam(pred.permute(1, 2, 0).cpu().numpy(), gt.permute(1, 2, 0).cpu().numpy())
                )
                pred_list.append(np.transpose(pred.detach().cpu().numpy(), (1, 2, 0)).astype(np.float32))
                truth_list.append(np.transpose(gt.detach().cpu().numpy(), (1, 2, 0)).astype(np.float32))

    model.train()
    return (
        np.stack(pred_list, axis=0),
        np.stack(truth_list, axis=0),
        psnr_list,
        ssim_list,
        sam_list,
        float(np.mean(np.asarray(psnr_list))),
        float(np.mean(np.asarray(ssim_list))),
        float(np.mean(np.asarray(sam_list))),
    )


def main():
    args = build_parser().parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    logger, _, output_dir = default_setup(cfg, args)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    generate_mask_3d_shift(mask_path=cfg.DATASETS.TRAIN.MASK_PATH)
    val_mask = generate_mask_3d_shift(mask_path=cfg.DATASETS.VAL.MASK_PATH).to(device)
    val_datas = LoadVal(cfg.DATASETS.VAL.PATH)

    model = UPSM(cfg).to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.MODEL.EMA.DECAY)
    optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.LR, betas=(0.9, 0.999))

    steps_per_epoch = max(1, int(np.floor(cfg.DATASETS.TRAIN.ITERATION / cfg.DATALOADER.BATCH_SIZE)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=steps_per_epoch,
        num_training_steps=steps_per_epoch * cfg.OPTIMIZER.MAX_EPOCH,
        eta_min=1e-6,
    )

    start_epoch = 0
    if cfg.RESUME_CKPT_PATH:
        logger.info(f"===> Loading Resume Checkpoint from {cfg.RESUME_CKPT_PATH}")
        save_state = torch.load(cfg.RESUME_CKPT_PATH, map_location=device)
        load_compatible_state_dict(model, save_state)
        if "ema" in save_state:
            ema.load_state_dict(save_state["ema"])
        if "optimizer" in save_state:
            optimizer.load_state_dict(save_state["optimizer"])
        if "scheduler" in save_state:
            scheduler.load_state_dict(save_state["scheduler"])
        start_epoch = int(save_state.get("epoch", 0))
    elif cfg.PRETRAINED_CKPT_PATH:
        logger.info(f"===> Loading Pretrained Checkpoint from {cfg.PRETRAINED_CKPT_PATH}")
        save_state = torch.load(cfg.PRETRAINED_CKPT_PATH, map_location=device)
        load_compatible_state_dict(model, save_state)
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.MODEL.EMA.DECAY)
        ema.update()
    else:
        ema.update()

    if cfg.LOSSES.L1_LOSS:
        l1_loss = CharbonnierLoss().to(device)
    if cfg.LOSSES.TV_LOSS:
        tv_loss = TVLoss().to(device)

    dataset = CSITrainDataset(cfg, crop_size=cfg.DATASETS.TRAIN.CROP_SIZE)
    psnr_best = 0.0

    for epoch in range(start_epoch + 1, cfg.OPTIMIZER.MAX_EPOCH + 1):
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=cfg.DATALOADER.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=cfg.DATALOADER.PIN_MEMORY,
            drop_last=True,
        )
        data_iter = iter(train_loader)
        batch_num = min(5, steps_per_epoch) if cfg.DEBUG else steps_per_epoch
        epoch_loss = 0.0
        begin = time.time()

        model.train()
        train_tqdm = tqdm(range(batch_num), desc=f"Epoch {epoch}")
        for _ in train_tqdm:
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                data = next(data_iter)

            data = {k: v.to(device) for k, v in data.items()}

            pred = model(data)
            loss = 0.0
            if cfg.LOSSES.L1_LOSS:
                loss = loss + l1_loss(pred, data["hsi"])
            if cfg.LOSSES.TV_LOSS:
                loss = loss + tv_loss(pred)

            loss.backward()
            if cfg.OPTIMIZER.GRAD_CLIP:
                clip_grad_norm_(model.parameters(), max_norm=0.2)
            optimizer.step()
            optimizer.zero_grad()
            ema.update()
            scheduler.step()

            epoch_loss += float(loss.detach().cpu())
            train_tqdm.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        logger.info(
            "===> Epoch %d Complete: Avg. Loss: %.6f time: %.2f"
            % (epoch, epoch_loss / batch_num, time.time() - begin)
        )

        pred, truth, _, _, _, psnr_mean, ssim_mean, sam_mean = evaluate(
            cfg, model, ema, val_datas, val_mask, device
        )
        logger.info(
            "===> Epoch %d: testing psnr = %.2f, ssim = %.3f, sam = %.3f"
            % (epoch, psnr_mean, ssim_mean, sam_mean)
        )

        if psnr_mean > psnr_best:
            psnr_best = psnr_mean
            checkpoint(model, ema, optimizer, scheduler, epoch, output_dir, logger)
            sio.savemat(
                os.path.join(output_dir, "val", f"epoch{epoch}_SAM{sam_mean:.3f}.mat"),
                {"pred": pred, "truth": truth},
            )


if __name__ == "__main__":
    main()
