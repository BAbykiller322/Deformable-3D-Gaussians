#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
# from lpipsPyTorch import lpips
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser


def readImages(renders_dir, gt_dir, masks_dir=None):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        mfile = (masks_dir / fname) if masks_dir is not None else None
        if mfile is not None and mfile.exists():
            m = tf.to_tensor(Image.open(mfile).convert('L')).unsqueeze(0).cuda()  # (1,1,H,W) in [0,1]
            masks.append((m > 0.5).float())
        else:
            masks.append(None)
        image_names.append(fname)
    return renders, gts, masks, image_names


def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                if not method.startswith("ours"):
                    continue
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                masks_dir = method_dir / "masks"
                renders, gts, masks, image_names = readImages(renders_dir, gt_dir, masks_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    r, g, m = renders[idx], gts[idx], masks[idx]
                    if m is not None:
                        # DyCheck covisibility-masked metrics (mPSNR/mSSIM/mLPIPS):
                        # PSNR over covisible pixels; SSIM map averaged over the
                        # mask; LPIPS on the mask-composited images.
                        denom = m.sum() * r.shape[1] + 1e-8
                        mse = (m * (r - g) ** 2).sum() / denom
                        psnrs.append((-10.0 * torch.log10(mse)).item())
                        ssims.append(ssim(r, g, mask=m).item())
                        lpipss.append(lpips_fn(r * m, g * m).detach().item())
                    else:
                        psnrs.append(psnr(r, g).item())
                        ssims.append(ssim(r, g).item())
                        lpipss.append(lpips_fn(r, g).detach().item())

                pfx = "m" if any(mm is not None for mm in masks) else " "
                print("  {}SSIM : {:>12.7f}".format(pfx, torch.tensor(ssims).mean()))
                print("  {}PSNR : {:>12.7f}".format(pfx, torch.tensor(psnrs).mean()))
                print("  {}LPIPS: {:>12.7f}".format(pfx, torch.tensor(lpipss).mean()))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                     "PSNR": torch.tensor(psnrs).mean().item(),
                                                     "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update(
                    {"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                     "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                     "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
