#!/usr/bin/env python3
"""
Stage 3: Run HorizonNet on indoor clip frames + write dummy pcl.ply.

For each clip_NNN/ under a video's clips/ directory:

  a) HorizonNet inference
     Reads clip_NNN/rgb/frame_{id:06d}.jpg
     Writes clip_NNN/vo_final/hn_mp3d/{id}.npy   (integer frame id, no padding)

  b) Dummy pcl.ply (frm_ref.txt and label.json are built in Stage 5 from SLAM)

Usage:
    conda run -n dfpe python dpfe_pipeline/run_horizonnet.py \
        --clips_dir data/videos/097/clips \
        --weights   HorizonNet/ckpt/resnet50_rnn__mp3d.pth
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dpfe_paths import HORIZONNET_DIR
sys.path.insert(0, str(HORIZONNET_DIR))

from model import HorizonNet
from misc.utils import load_trained_model

W, H = 1024, 512


# ── HorizonNet ────────────────────────────────────────────────────────────────

def _infer(net, img_path: Path, device) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    if img.size != (W, H):
        img = img.resize((W, H), Image.BICUBIC)
    x = torch.FloatTensor(
        np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        y_bon, y_cor = net(x)
    bon = y_bon.cpu().numpy()[0]                        # (2, 1024)
    cor = torch.sigmoid(y_cor).cpu().numpy()[0, 0]     # (1024,)
    return np.stack([bon[0], bon[1], cor], axis=0).astype(np.float32)


def run_horizonnet(clip_dir: Path, weights: Path, device):
    rgb_dir  = clip_dir / "rgb"
    hn_dir   = clip_dir / "vo_final" / "hn_mp3d"
    hn_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(rgb_dir.glob("frame_*.jpg"))
    if not frames:
        print(f"  [HN] No frames in {rgb_dir}, skipping.")
        return

    net = load_trained_model(HorizonNet, str(weights)).to(device)
    net.eval()

    print(f"  [HN] {clip_dir.name}: {len(frames)} frames → {hn_dir}")
    for img_path in tqdm(frames, desc=f"  {clip_dir.name}"):
        # frame_000394.jpg  →  394
        frame_id = int(img_path.stem.split("_")[1])
        out = hn_dir / f"{frame_id}.npy"
        if out.exists():
            continue
        result = _infer(net, img_path, device)
        np.save(str(out), result)


def write_pcl_ply(clip_dir: Path):
    """Minimal PLY with xyz+rgb — read_ply in DFPE reads cols 3:6 as colour."""
    (clip_dir / "pcl.ply").write_text(
        "ply\nformat ascii 1.0\n"
        "element vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
        "0 0 0 128 128 128\n"
        "1 0 0 128 128 128\n"
        "0 1 0 128 128 128\n"
        "0 0 1 128 128 128\n"
    )
    print(f"  [dummy] pcl.ply")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 3: HorizonNet + dummy DFPE scaffold")
    parser.add_argument("--clips_dir", type=Path, required=True,
                        help="Directory containing clip_NNN/ subdirectories")
    parser.add_argument("--weights", type=Path,
                        default=HORIZONNET_DIR / "ckpt/resnet50_rnn__mp3d.pth")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    clips = sorted(args.clips_dir.glob("clip_*"))
    if not clips:
        print(f"No clip_* directories found in {args.clips_dir}")
        return

    for clip_dir in clips:
        print(f"\n{'='*50}")
        print(f"Clip: {clip_dir}")
        run_horizonnet(clip_dir, args.weights, device)
        write_pcl_ply(clip_dir)

    print("\nStage 3 complete.")


if __name__ == "__main__":
    main()
