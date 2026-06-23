import os

import numpy as np
import torch

from gaussian_renderer import render


def warmup_depth_iteration(opt):
    requested_iteration = getattr(opt, "warmup_depth_iteration", 0)
    if requested_iteration > 0:
        return requested_iteration
    return max(1, int(getattr(opt, "warm_up", 1)))


def default_warmup_depth_path(model_path, iteration):
    filename = f"train_depth_{iteration:05d}.npz"
    return os.path.join(model_path, "warmup_depth", filename)


def should_export_warmup_depth(opt, iteration):
    return bool(getattr(opt, "warmup_depth_export", False)) and iteration == warmup_depth_iteration(opt)


def _camera_sort_key(camera):
    fid = float(camera.fid.detach().flatten()[0].cpu().item())
    return fid, str(camera.image_name)


def _depth_to_numpy(depth):
    depth_np = depth.detach().float().cpu().numpy()
    if depth_np.ndim == 3 and depth_np.shape[0] == 1:
        depth_np = depth_np[0]
    if depth_np.ndim != 2:
        raise ValueError(f"Expected a single-channel depth map, got shape {depth_np.shape}")
    return depth_np.astype(np.float32, copy=False)


def export_warmup_depth(scene, gaussians, pipe, background, dataset, opt, iteration):
    output_path = getattr(opt, "warmup_depth_output", "")
    if not output_path:
        output_path = default_warmup_depth_path(dataset.model_path, iteration)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    depths = []
    valid_masks = []
    frame_names = []
    fids = []
    image_sizes = []
    cameras = sorted(scene.getTrainCameras().copy(), key=_camera_sort_key)

    with torch.no_grad():
        for camera in cameras:
            if dataset.load2gpu_on_the_fly:
                camera.load2device()

            render_pkg = render(
                camera,
                gaussians,
                pipe,
                background,
                0.0,
                0.0,
                0.0,
                dataset.is_6dof,
            )
            depth_np = _depth_to_numpy(render_pkg["depth"])
            valid_mask = np.isfinite(depth_np) & (depth_np > 0.0)

            depths.append(depth_np)
            valid_masks.append(valid_mask)
            frame_names.append(str(camera.image_name))
            fids.append(float(camera.fid.detach().flatten()[0].cpu().item()))
            image_sizes.append((int(camera.image_width), int(camera.image_height)))

            if dataset.load2gpu_on_the_fly:
                camera.load2device("cpu")

            del render_pkg

    depths = np.stack(depths, axis=0)
    valid_masks = np.stack(valid_masks, axis=0)

    np.savez_compressed(
        output_path,
        depth=depths,
        valid_mask=valid_masks,
        frame_names=np.asarray(frame_names),
        fids=np.asarray(fids, dtype=np.float32),
        image_sizes=np.asarray(image_sizes, dtype=np.int32),
        iteration=np.asarray(iteration, dtype=np.int32),
        warm_up=np.asarray(getattr(opt, "warm_up", 0), dtype=np.int32),
        depth_space=np.asarray("camera_z_from_static_warmup_render"),
        split=np.asarray("train"),
        valid_fraction=np.asarray(valid_masks.mean(), dtype=np.float32),
    )
    torch.cuda.empty_cache()
    return output_path, len(frame_names), float(valid_masks.mean())
