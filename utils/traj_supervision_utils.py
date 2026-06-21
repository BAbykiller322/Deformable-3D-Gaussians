"""
Trajectory supervision utilities for DyCheck + Deformable 3DGS.

This module is intentionally a design skeleton at this stage.  It defines the
expected data flow for L_traj without implementing the core projection,
deformation, or optimization logic yet.

Planned supervision signal:
    MDE anchor at reference frame t0
        -> deform anchor from t0 to target frame t
        -> project deformed 3D anchor into camera_t
        -> compare against CoTracker track observation tracks[t, p]

Expected input files:
    tracks_npz:
        train_cotracker_online/{scene}_train_tracks_2x_g20.npz

    anchors_npz:
        MDE_initialization/{scene}_train_anchors_2x_g20.npz

Expected output during training:
    loss_traj:
        scalar tensor, differentiable w.r.t. deformation network and optionally
        learnable anchor positions.

    diagnostics:
        small dict for logging, e.g. valid pair count, mean reprojection error,
        sampled frame id, and selected strategy.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class TrajectorySupervisorConfig:
    """Configuration for trajectory supervision.

    Args:
        tracks_path:
            Path to CoTracker output npz.
        anchors_path:
            Path to MDE-initialized anchor npz.
        strategy:
            Anchor strategy name. Expected values: "first", "min_motion".
        lambda_traj:
            Weight applied in train.py: total_loss += lambda_traj * loss_traj.
        start_iter:
            Iteration from which L_traj is enabled.
        num_tracks_per_iter:
            Number of anchor tracks sampled per training iteration.
        robust_delta:
            Scale parameter for robust reprojection loss.
        learnable_anchors:
            If True, anchor_xyz becomes nn.Parameter and must be added to an
            optimizer in train.py.
    """

    tracks_path: str
    anchors_path: str
    strategy: str = "first"
    lambda_traj: float = 0.0
    start_iter: int = 0
    num_tracks_per_iter: int = 128
    robust_delta: float = 1.0
    learnable_anchors: bool = False


def load_tracks_npz(tracks_path: str) -> Dict[str, np.ndarray]:
    """Load CoTracker trajectories."""

    required_keys = ("tracks", "visibility", "frame_names", "time_ids", "max_warp_id")

    with np.load(tracks_path, allow_pickle=False) as npz:
        missing = [key for key in required_keys if key not in npz]
        if missing:
            raise KeyError(f"Missing CoTracker field in {tracks_path}: {missing}")

        tracks = np.asarray(npz["tracks"], dtype=np.float32)
        visibility = np.asarray(npz["visibility"], dtype=np.bool_)
        frame_names = np.asarray(npz["frame_names"]).astype(str)
        time_ids = np.asarray(npz["time_ids"], dtype=np.int64)
        max_warp_id = np.asarray(npz["max_warp_id"], dtype=np.int64)

        result: Dict[str, np.ndarray] = {
            "tracks": np.ascontiguousarray(tracks),
            "visibility": np.ascontiguousarray(visibility),
            "frame_names": frame_names,
            "time_ids": time_ids,
            "max_warp_id": max_warp_id,
        }

        for key in ("resolution", "query_frame", "grid_size", "camera"):
            if key in npz:
                result[key] = np.asarray(npz[key], dtype=np.int64)

    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(
            f"tracks must have shape [T, P, 2], got {tracks.shape} in {tracks_path}"
        )

    num_frames, num_tracks, _ = tracks.shape

    if visibility.shape != (num_frames, num_tracks):
        raise ValueError(
            "visibility must have shape [T, P] matching tracks, "
            f"got {visibility.shape} vs tracks {tracks.shape} in {tracks_path}"
        )
    if frame_names.shape != (num_frames,):
        raise ValueError(
            f"frame_names must have shape [T], got {frame_names.shape} for T={num_frames}"
        )
    if time_ids.shape != (num_frames,):
        raise ValueError(
            f"time_ids must have shape [T], got {time_ids.shape} for T={num_frames}"
        )
    if max_warp_id.shape != ():
        raise ValueError(
            f"max_warp_id must be a scalar, got shape {max_warp_id.shape} in {tracks_path}"
        )
    return result


def load_anchor_npz(anchors_path: str, strategy: str) -> Dict[str, np.ndarray]:
    """Load MDE-initialized anchors for one strategy."""

    suffix = strategy.strip()
    if not suffix:
        raise ValueError("strategy must be a non-empty string")
    required_by_output_key = {
        "valid": f"valid_{suffix}",
        "anchor_xyz": f"anchor_xyz_{suffix}",
        "anchor_xy": f"anchor_xy_{suffix}",
        "anchor_t0_idx": f"anchor_t0_idx_{suffix}",
        "anchor_warp": f"anchor_warp_{suffix}",
    }

    with np.load(anchors_path, allow_pickle=False) as npz:
        missing = [key for key in required_by_output_key.values() if key not in npz]
        if missing:
            available_strategies = sorted(
                key[len("valid_") :] for key in npz.files if key.startswith("valid_")
            )
            raise KeyError(
                f"Missing anchor field(s) for strategy '{suffix}' in {anchors_path}: "
                f"{missing}. Available strategies: {available_strategies or 'none'}."
            )

        valid = np.asarray(npz[required_by_output_key["valid"]], dtype=np.bool_)
        anchor_xyz = np.asarray(npz[required_by_output_key["anchor_xyz"]], dtype=np.float32)
        anchor_xy = np.asarray(npz[required_by_output_key["anchor_xy"]], dtype=np.float32)
        anchor_t0_idx = np.asarray(
            npz[required_by_output_key["anchor_t0_idx"]], dtype=np.int64
        )
        anchor_warp = np.asarray(npz[required_by_output_key["anchor_warp"]], dtype=np.int64)

        result: Dict[str, np.ndarray] = {
            "valid": np.ascontiguousarray(valid),
            "anchor_xyz": np.ascontiguousarray(anchor_xyz),
            "anchor_xy": np.ascontiguousarray(anchor_xy),
            "anchor_t0_idx": anchor_t0_idx,
            "anchor_warp": anchor_warp,
        }

        depth_key = f"anchor_depth_{suffix}"
        if depth_key in npz:
            result["anchor_depth"] = np.asarray(npz[depth_key], dtype=np.float32)

        for key in ("scene", "resolution", "model_id", "depth_convention", "coordinate_space"):
            if key in npz:
                result[key] = np.asarray(npz[key])

    if valid.ndim != 1:
        raise ValueError(f"valid must have shape [P], got {valid.shape} in {anchors_path}")

    num_anchors = valid.shape[0]

    if anchor_xyz.shape != (num_anchors, 3):
        raise ValueError(
            "anchor_xyz must have shape [P, 3] matching valid, "
            f"got {anchor_xyz.shape} for P={num_anchors} in {anchors_path}"
        )
    if anchor_xy.shape != (num_anchors, 2):
        raise ValueError(
            "anchor_xy must have shape [P, 2] matching valid, "
            f"got {anchor_xy.shape} for P={num_anchors} in {anchors_path}"
        )
    if anchor_t0_idx.shape != (num_anchors,):
        raise ValueError(
            "anchor_t0_idx must have shape [P] matching valid, "
            f"got {anchor_t0_idx.shape} for P={num_anchors} in {anchors_path}"
        )
    if anchor_warp.shape != (num_anchors,):
        raise ValueError(
            "anchor_warp must have shape [P] matching valid, "
            f"got {anchor_warp.shape} for P={num_anchors} in {anchors_path}"
        )
    return result


def normalize_time_ids(time_ids: torch.Tensor, max_warp_id: torch.Tensor) -> torch.Tensor:
    """Normalize CoTracker time_ids to [0, 1] range."""

    time_ids = time_ids.to(dtype=torch.float32)
    max_warp_id = max_warp_id.to(device=time_ids.device, dtype=torch.float32)

    if time_ids.ndim == 1:
        time_ids = time_ids[:, None]

    denom = torch.clamp(max_warp_id, min=1.0)

    return time_ids / denom
    


def sample_track_indices(
    valid_anchor_mask: torch.Tensor,
    visibility_t: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Sample track ids that are valid for the current training frame."""
    candidate_mask = valid_anchor_mask & visibility_t
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)

    if num_samples <= 0 or candidate_indices.numel() <= num_samples:
        return candidate_indices

    perm = torch.randperm(candidate_indices.numel(), device=candidate_indices.device)
    return candidate_indices[perm[:num_samples]]


def deform_anchors_between_times(
    deform_model,
    anchor_xyz: torch.Tensor,
    fid_t: torch.Tensor,
    fid0: torch.Tensor,
) -> torch.Tensor:
    """Move anchors from their reference time fid0 to target time fid_t."""

    num_anchors = anchor_xyz.shape[0]
    expected_shape = (num_anchors, 1)

    if fid_t.shape != expected_shape:
        raise ValueError(f"fid_t must have shape [K, 1], got {tuple(fid_t.shape)}")
    if fid0.shape != expected_shape:
        raise ValueError(f"fid0 must have shape [K, 1], got {tuple(fid0.shape)}")
    if fid_t.device != anchor_xyz.device or fid0.device != anchor_xyz.device:
        raise ValueError("fid_t, fid0, and anchor_xyz must be on the same device")
    if fid_t.dtype != anchor_xyz.dtype or fid0.dtype != anchor_xyz.dtype:
        raise ValueError("fid_t, fid0, and anchor_xyz must have the same dtype")

    d_xyz_t, _, _ = deform_model.step(anchor_xyz, fid_t)
    d_xyz_0, _, _ = deform_model.step(anchor_xyz, fid0)

    return anchor_xyz + d_xyz_t - d_xyz_0


def project_world_to_pixel(xyz_world: torch.Tensor, viewpoint_cam) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project world-space points to pixel coordinates for one Camera."""
    K = xyz_world.shape[0]

    ones = torch.ones(K, 1, dtype=xyz_world.dtype, device=xyz_world.device)
    xyz_h = torch.cat([xyz_world, ones], dim=-1)

    full_proj_transform = viewpoint_cam.full_proj_transform.to(
        device=xyz_world.device, dtype=xyz_world.dtype
    )
    world_view_transform = viewpoint_cam.world_view_transform.to(
        device=xyz_world.device, dtype=xyz_world.dtype
    )

    clip = xyz_h @ full_proj_transform
    ndc = clip[:, :3] / (clip[:, 3:] + 1e-7)

    width, height = viewpoint_cam.image_width, viewpoint_cam.image_height

    x = ((ndc[:, 0] + 1.0) * width - 1.0) * 0.5
    y = ((ndc[:, 1] + 1.0) * height - 1.0) * 0.5

    xy = torch.stack([x, y], dim=-1)

    view = xyz_h @ world_view_transform
    infront = view[:, 2] > 0.0

    return xy, infront


def masked_reprojection_loss(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    valid_mask: torch.Tensor,
    robust_delta: float,
) -> torch.Tensor:
    """Compute robust 2D reprojection loss.Uses a Huber-style penalty on the per-point 2D pixel distance."""

    if pred_xy.ndim != 2 or pred_xy.shape[-1] != 2:
        raise ValueError(f"pred_xy must have shape [K, 2], got {tuple(pred_xy.shape)}")
    if target_xy.shape != pred_xy.shape:
        raise ValueError(
            f"target_xy must match pred_xy shape, got {tuple(target_xy.shape)} "
            f"vs {tuple(pred_xy.shape)}"
        )
    if valid_mask.shape != (pred_xy.shape[0],):
        raise ValueError(
            f"valid_mask must have shape [K], got {tuple(valid_mask.shape)} "
            f"for K={pred_xy.shape[0]}"
        )
    if valid_mask.dtype != torch.bool:
        raise ValueError(f"valid_mask must be bool, got {valid_mask.dtype}")
    if pred_xy.device != target_xy.device or pred_xy.device != valid_mask.device:
        raise ValueError("pred_xy, target_xy, and valid_mask must be on the same device")
    if not torch.is_floating_point(pred_xy) or not torch.is_floating_point(target_xy):
        raise ValueError("pred_xy and target_xy must be floating point tensors")
    if robust_delta <= 0:
        raise ValueError(f"robust_delta must be positive, got {robust_delta}")

    if not valid_mask.any():
        return pred_xy.sum() * 0.0

    residual = pred_xy[valid_mask] - target_xy[valid_mask]
    error = torch.linalg.norm(residual, dim=-1)

    delta = pred_xy.new_tensor(robust_delta)
    quadratic = torch.minimum(error, delta)
    linear = error - quadratic
    loss = 0.5 * quadratic.square() / delta + linear

    return loss.mean()


def anchor_prior_loss(
    anchor_xyz: torch.Tensor,
    anchor_xyz_init: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Optional regularizer that keeps learnable anchors near MDE init.

    Args:
        anchor_xyz:
            Current anchor tensor [P, 3].
        anchor_xyz_init:
            Initial anchor tensor [P, 3].
        valid_mask:
            Optional bool tensor [P].

    Returns:
        Scalar tensor.

    TODO:
        Decide whether this prior is always enabled when learnable_anchors=True.
    """

    raise NotImplementedError("Compute optional anchor initialization prior.")


class TrajectorySupervisor(nn.Module):
    """Stateful helper for L_traj inside train.py.

    Responsibilities:
        1. Load preprocessed tracks and anchors.
        2. Store tensors on the training device.
        3. Sample visible track-frame pairs for current viewpoint.
        4. Compute differentiable L_traj and diagnostics.

    Expected train.py usage:
        traj_supervisor = TrajectorySupervisor(config, device="cuda")

        ...

        loss_traj, traj_log = traj_supervisor.compute_loss(
            iteration=iteration,
            viewpoint_cam=viewpoint_cam,
            deform_model=deform,
        )

        loss = image_loss + config.lambda_traj * loss_traj

    TODO:
        If anchors are learnable, expose self.anchor_xyz as nn.Parameter and
        make sure train.py adds it to an optimizer.
    """

    def __init__(self, config: TrajectorySupervisorConfig, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.device = torch.device(device)

        tracks_npz = load_tracks_npz(config.tracks_path)
        anchors_npz = load_anchor_npz(config.anchors_path, config.strategy)

        self.strategy = config.strategy
        self.frame_names = [str(name) for name in tracks_npz["frame_names"]]
        self.frame_name_to_index = {
            frame_name: frame_idx for frame_idx, frame_name in enumerate(self.frame_names)
        }

        tracks = torch.from_numpy(tracks_npz["tracks"]).to(self.device)
        visibility = torch.from_numpy(tracks_npz["visibility"]).to(self.device)
        valid_anchor_mask = torch.from_numpy(anchors_npz["valid"]).to(self.device)
        anchor_t0_idx = torch.from_numpy(anchors_npz["anchor_t0_idx"]).to(self.device)

        time_ids = torch.from_numpy(tracks_npz["time_ids"]).to(self.device)
        max_warp_id = torch.from_numpy(tracks_npz["max_warp_id"]).to(self.device)
        frame_fids = normalize_time_ids(time_ids, max_warp_id)

        anchor_warp = torch.from_numpy(anchors_npz["anchor_warp"]).to(self.device)
        anchor_fid0 = normalize_time_ids(anchor_warp, max_warp_id)

        anchor_xyz_init = torch.from_numpy(anchors_npz["anchor_xyz"]).to(self.device)
        if config.learnable_anchors:
            self.anchor_xyz = nn.Parameter(anchor_xyz_init.clone())
        else:
            self.register_buffer("anchor_xyz", anchor_xyz_init.clone())

        self.register_buffer("tracks", tracks)
        self.register_buffer("visibility", visibility)
        self.register_buffer("frame_fids", frame_fids)
        self.register_buffer("max_warp_id", max_warp_id)
        self.register_buffer("valid_anchor_mask", valid_anchor_mask)
        self.register_buffer("anchor_t0_idx", anchor_t0_idx)
        self.register_buffer("anchor_fid0", anchor_fid0)
        self.register_buffer("anchor_xyz_init", anchor_xyz_init.clone())

    def is_enabled(self, iteration: int) -> bool:
        """Return whether L_traj should be active at this iteration."""

        return self.config.lambda_traj > 0.0 and iteration >= self.config.start_iter

    def frame_index_from_camera(self, viewpoint_cam) -> int:
        """Map the current training Camera to a track frame index.

        Args:
            viewpoint_cam:
                Camera selected by train.py for this iteration.

        Returns:
            Integer frame index t used to index tracks[t].

        Prefer exact DyCheck frame-name matching.  The fid fallback keeps the
        method usable for synthetic MiniCam-style callers.
        """

        image_name = str(viewpoint_cam.image_name)
        if image_name in self.frame_name_to_index:
            return self.frame_name_to_index[image_name]

        fid = viewpoint_cam.fid.to(self.frame_fids.device).flatten()[0]
        frame_dist = torch.abs(self.frame_fids.flatten() - fid)
        return int(torch.argmin(frame_dist).item())

    def compute_loss(
        self,
        iteration: int,
        viewpoint_cam,
        deform_model,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute L_traj for one training iteration.

        Args:
            iteration:
                Current optimization iteration.
            viewpoint_cam:
                Current Camera from train.py.
            deform_model:
                DeformModel instance used by the backbone.

        Returns:
            loss_traj:
                Scalar tensor. Zero tensor when disabled or no valid samples.
            diagnostics:
                Dict for TensorBoard/progress logging.

        The main path intentionally stays compact; validation lives in the
        loading/projection/loss helpers.
        """

        zero_loss = self.tracks.new_zeros(())
        if not self.is_enabled(iteration):
            return zero_loss, {"traj_enabled": 0.0}

        frame_idx = self.frame_index_from_camera(viewpoint_cam)
        track_ids = sample_track_indices(
            self.valid_anchor_mask,
            self.visibility[frame_idx],
            self.config.num_tracks_per_iter,
        )

        if track_ids.numel() == 0:
            return zero_loss, {
                "traj_enabled": 1.0,
                "traj_frame_idx": float(frame_idx),
                "traj_num_samples": 0.0,
                "traj_num_valid": 0.0,
                "traj_mean_reproj_px": 0.0,
            }

        anchor_xyz = self.anchor_xyz[track_ids]
        target_xy = self.tracks[frame_idx, track_ids]

        # fid_t is shared by all selected anchors; fid0 can be per-anchor.
        fid_t = self.frame_fids[frame_idx].expand(track_ids.numel(), 1)
        fid0 = self.anchor_fid0[track_ids]

        anchor_xyz_t = deform_anchors_between_times(
            deform_model=deform_model,
            anchor_xyz=anchor_xyz,
            fid_t=fid_t,
            fid0=fid0,
        )

        pred_xy, in_front = project_world_to_pixel(anchor_xyz_t, viewpoint_cam)

        width, height = viewpoint_cam.image_width, viewpoint_cam.image_height
        in_bounds_x = (pred_xy[:, 0] >= 0.0) & (pred_xy[:, 0] < width)
        in_bounds_y = (pred_xy[:, 1] >= 0.0) & (pred_xy[:, 1] < height)
        valid_mask = in_front & in_bounds_x & in_bounds_y

        loss_traj = masked_reprojection_loss(
            pred_xy=pred_xy,
            target_xy=target_xy,
            valid_mask=valid_mask,
            robust_delta=self.config.robust_delta,
        )

        with torch.no_grad():
            reproj_error = torch.linalg.norm(pred_xy - target_xy, dim=-1)
            mean_reproj = reproj_error[valid_mask].mean() if valid_mask.any() else zero_loss

        traj_log = {
            "traj_enabled": 1.0,
            "traj_frame_idx": float(frame_idx),
            "traj_num_samples": float(track_ids.numel()),
            "traj_num_valid": float(valid_mask.sum().item()),
            "traj_mean_reproj_px": float(mean_reproj.item()),
        }

        return loss_traj, traj_log
