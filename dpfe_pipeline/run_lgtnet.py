#!/usr/bin/env python3
"""
Stage 3 (LGT-Net variant): Run LGT-Net on indoor clip keyframes.

Replaces run_horizonnet.py. Operates only on SLAM keyframes (much faster)
so it must run AFTER Stage 4 (SLAM) and Stage 5 (setup_scene).

Outputs HorizonNet-compatible (3, 1024) .npy files named by keyframe ID:
    clip_dir/vo_final/hn_mp3d/{kf_id}.npy

Usage (from the repo root, lgtnet env active):
    conda run -n lgtnet python dpfe_pipeline/run_lgtnet.py \\
        --clip_dir data/videos/085/clips/clip_000

    # All clips under a video dir:
    conda run -n lgtnet python dpfe_pipeline/run_lgtnet.py \\
        --clips_dir data/videos/085/clips
"""

import argparse
import sys
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

from dpfe_paths import LGTNET_DIR
LGTNET_CFG = LGTNET_DIR / "src/config/mp3d.yaml"
LGTNET_CKPT = LGTNET_DIR / "checkpoints/SWG_Transformer_LGT_Net/mp3d/best.pkl"

sys.path.insert(0, str(LGTNET_DIR))

from config.defaults import get_config
from models.build import build_model
from utils.misc import tensor2np
from postprocessing.post_process import post_process


def _load_model(device):
    import argparse as _ap
    import logging
    cfg_args = _ap.Namespace(cfg=str(LGTNET_CFG), mode='test')
    config = get_config(cfg_args)
    config.defrost()
    config.CKPT.DIR = str(LGTNET_CKPT.parent)
    if device == 'cpu':
        config.TRAIN.DEVICE = 'cpu'
    config.freeze()
    logging.getLogger('lgt_inference').setLevel(logging.WARNING)
    model, _, _, _ = build_model(config, logging.getLogger('lgt_inference'))
    model.eval()
    model.to(device)
    return model


def _corners_to_prob(corners_xyz, width=1024, sigma=10):
    prob = np.zeros(width, dtype=np.float32)
    if len(corners_xyz) == 0:
        # manhattan post-processing found no corners (e.g. open lobby, glass walls);
        # place 4 evenly-spaced blobs so DFPE always has peaks to segment walls
        cols = [0, width // 4, width // 2, 3 * width // 4]
    else:
        lons = np.arctan2(corners_xyz[:, 0], corners_xyz[:, 2])
        cols = ((lons / (2 * np.pi) + 0.5) * width).astype(int) % width
    for col in cols:
        prob[col] = 1.0
    prob = np.concatenate([prob, prob, prob])
    prob = gaussian_filter1d(prob, sigma=sigma)
    prob = prob[width:2 * width]
    if prob.max() > 0:
        prob /= prob.max()
    return prob


@torch.no_grad()
def _infer(model, img_path, device):
    img = np.array(Image.open(img_path).resize((1024, 512), Image.BILINEAR))[..., :3]
    tensor = torch.from_numpy(
        (img / 255.0).astype(np.float32).transpose(2, 0, 1)[None]
    ).to(device)
    dt = model(tensor)
    depth_256 = tensor2np(dt['depth'][0])
    ratio = float(max(tensor2np(dt['ratio'][0])[0], 1e-3))
    try:
        corners_xyz = post_process(depth_256[None], type_name='manhattan')[0]
    except Exception:
        corners_xyz = np.zeros((0, 3))
    return depth_256, ratio, corners_xyz


def _to_npy(depth_256, ratio, corners_xyz):
    xs = np.linspace(0, 1, 1024)
    depth = np.interp(xs, np.linspace(0, 1, 256), depth_256).astype(np.float32)
    depth = np.clip(depth, 1e-3, None)
    row1 = np.arctan(1.0 / depth).astype(np.float32)
    row0 = -np.arctan(1.0 / (ratio * depth)).astype(np.float32)
    row2 = _corners_to_prob(corners_xyz)
    return np.stack([row0, row1, row2], axis=0)


def run_lgtnet(clip_dir: Path, device: str):
    kf_list_path = clip_dir / "vo_final" / "keyframe_list.txt"
    if not kf_list_path.exists():
        print(f"  [LGT] No keyframe_list.txt in {clip_dir.name} — run SLAM + setup_scene first")
        return

    rgb_dir = clip_dir / "rgb"
    hn_dir  = clip_dir / "vo_final" / "lgt_mp3d"
    hn_dir.mkdir(parents=True, exist_ok=True)

    # Map kf_id → video frame id
    jpg_files = sorted(rgb_dir.glob("frame_*.jpg"))
    if not jpg_files:
        print(f"  [LGT] No frames in {rgb_dir}")
        return
    clip_start = int(jpg_files[0].stem.split("_")[1])

    kf_ids = [int(x.strip()) for x in kf_list_path.read_text().splitlines() if x.strip()]

    # Check which kf_ids still need inference
    todo = [k for k in kf_ids if not (hn_dir / f"{k}.npy").exists()]
    if not todo:
        print(f"  [LGT] {clip_dir.name}: all {len(kf_ids)} keyframes already done, skipping")
        return

    print(f"  [LGT] {clip_dir.name}: {len(todo)} / {len(kf_ids)} keyframes to process")
    model = _load_model(device)

    missing = []
    for kf_id in tqdm(todo, desc=f"  {clip_dir.name}"):
        video_fid = clip_start + kf_id - 1
        img_path  = rgb_dir / f"frame_{video_fid:06d}.jpg"
        out_path  = hn_dir  / f"{kf_id}.npy"

        if not img_path.exists():
            missing.append(kf_id)
            continue

        depth, ratio, corners = _infer(model, img_path, device)
        np.save(str(out_path), _to_npy(depth, ratio, corners))

    print(f"  [LGT] Done: {len(todo) - len(missing)} saved")
    if missing:
        print(f"  [LGT] WARNING: {len(missing)} images not found (kf_ids: {missing[:5]}...)")


def main():
    parser = argparse.ArgumentParser(description="Stage 3 (LGT-Net): Layout inference on SLAM keyframes")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--clip_dir",  type=Path, help="Single clip directory")
    group.add_argument("--clips_dir", type=Path, help="Parent clips/ directory (processes all clip_NNN/)")
    parser.add_argument("--device", default="cuda",
                        help="cuda or cpu (default: cuda)")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("  CUDA not available, falling back to CPU")
        args.device = "cpu"

    clips = [args.clip_dir] if args.clip_dir else sorted(args.clips_dir.glob("clip_*"))
    for clip_dir in clips:
        print(f"\n{'='*50}")
        run_lgtnet(clip_dir, args.device)

    print("\nStage 3 (LGT-Net) complete.")


if __name__ == "__main__":
    main()
