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
from utils.dycheck_metrics import compute_mpsnr, compute_mssim, compute_mlpips
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

                ssims = []          # full-image (backbone-native)
                psnrs = []
                lpipss = []
                mssims = []         # DyCheck covisibility-masked (test frames)
                mpsnrs = []
                mlpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    r, g, m = renders[idx], gts[idx], masks[idx]
                    # full-image metrics (always); backbone-native, vgg LPIPS is
                    # the 3DGS reporting convention (normalize=True maps [0,1]->[-1,1]).
                    psnrs.append(psnr(r, g).item())
                    ssims.append(ssim(r, g).item())
                    lpipss.append(lpips_vgg(r, g, normalize=True).detach().item())
                    # DyCheck covisibility-masked metrics (official protocol), via the
                    # vendored Shape-of-Motion port so they are directly comparable to
                    # the published MoSca / Dynamic Gaussian Marbles numbers.
                    if m is not None:
                        pr = r.permute(0, 2, 3, 1).contiguous()   # (1,H,W,3)
                        gt = g.permute(0, 2, 3, 1).contiguous()
                        mk = m[:, 0]                              # (1,H,W)
                        mpsnrs.append(compute_mpsnr(pr, gt, mk))
                        mssims.append(compute_mssim(pr, gt, mk))
                        mlpipss.append(compute_mlpips(pr, gt, lpips_alex, mk))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean()))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean()))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean()))
                if len(mpsnrs) > 0:
                    print("  mSSIM : {:>11.7f}".format(torch.tensor(mssims).mean()))
                    print("  mPSNR : {:>11.7f}".format(torch.tensor(mpsnrs).mean()))
                    print("  mLPIPS: {:>11.7f}".format(torch.tensor(mlpipss).mean()))
                print("")

                results = {"SSIM": torch.tensor(ssims).mean().item(),
                           "PSNR": torch.tensor(psnrs).mean().item(),
                           "LPIPS": torch.tensor(lpipss).mean().item()}
                if len(mpsnrs) > 0:
                    results.update({"mSSIM": torch.tensor(mssims).mean().item(),
                                    "mPSNR": torch.tensor(mpsnrs).mean().item(),
                                    "mLPIPS": torch.tensor(mlpipss).mean().item()})
                full_dict[scene_dir][method].update(results)
                per_view_dict[scene_dir][method].update(
                    {"SSIM": {name: s for s, name in zip(torch.tensor(ssims).tolist(), image_names)},
                     "PSNR": {name: p for p, name in zip(torch.tensor(psnrs).tolist(), image_names)},
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
    # Two LPIPS models, on purpose:
    #   lpips_vgg  : full-image LPIPS (3DGS / Deformable-3DGS reporting convention).
    #   lpips_alex : DyCheck official mLPIPS backbone (AlexNet, spatial map) for the
    #                masked metrics. Kept separate so each column matches its protocol.
    lpips_vgg = lpips.LPIPS(net='vgg').to(device).eval()
    lpips_alex = lpips.LPIPS(net='alex', spatial=True).to(device).eval()

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
