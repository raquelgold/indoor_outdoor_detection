#!/usr/bin/env python3
"""
Visualise HorizonNet predictions overlaid on the source frames.

For each clip_NNN/ draws:
  - red line   : floor boundary
  - blue line  : ceiling boundary
  - green tick : wall corners (corner prob > CORNER_THRESH)

Saves N_SAMPLES images to clip_NNN/viz/.

Usage:
    python dpfe_pipeline/visualize_horizonnet.py --clips_dir data/videos/097/clips
    python dpfe_pipeline/visualize_horizonnet.py --clips_dir data/videos/098/clips
"""

import argparse
import math
import numpy as np
import cv2
from pathlib import Path

N_SAMPLES      = 12      # frames to visualise per clip
CORNER_THRESH  = 0.5     # corner probability threshold
HN_W, HN_H    = 1024, 512


def phi_to_row(phi: np.ndarray, h: int) -> np.ndarray:
    """Equirectangular phi (radians, +up) → pixel row."""
    return ((0.5 - phi / math.pi) * h).astype(int).clip(0, h - 1)


def draw_prediction(frame: np.ndarray, npy: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]

    floor_phi  = npy[0]   # (1024,)
    ceil_phi   = npy[1]   # (1024,)
    corner_prob = npy[2]  # (1024,)

    floor_rows = phi_to_row(floor_phi, h)
    ceil_rows  = phi_to_row(ceil_phi,  h)

    # Scale HorizonNet's 1024-wide predictions to actual frame width
    xs = np.linspace(0, w - 1, HN_W).astype(int)

    img = frame.copy()

    # Draw floor (red) and ceiling (blue) as polylines
    for color, rows in [(( 50,  50, 220), floor_rows),   # red  (BGR)
                        ((220,  80,   0), ceil_rows)]:   # blue (BGR)
        pts = np.stack([xs, rows], axis=1).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=2)

    # Draw corner ticks (green) where probability exceeds threshold
    corner_xs = np.where(corner_prob > CORNER_THRESH)[0]
    for cx in corner_xs:
        px = xs[cx]
        cv2.line(img, (px, 0), (px, h - 1), (30, 200, 30), 1)

    return img


def visualize_clip(clip_dir: Path, n_samples: int):
    rgb_dir = clip_dir / "rgb"
    hn_dir  = clip_dir / "vo_final" / "hn_mp3d"
    viz_dir = clip_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    frames = sorted(rgb_dir.glob("frame_*.jpg"))
    if not frames:
        print(f"  No frames in {rgb_dir}")
        return

    # Pick evenly-spaced samples
    indices = np.linspace(0, len(frames) - 1, n_samples, dtype=int)
    samples = [frames[i] for i in indices]

    saved = 0
    for jpg_path in samples:
        frame_id = int(jpg_path.stem.split("_")[1])
        npy_path = hn_dir / f"{frame_id}.npy"
        if not npy_path.exists():
            print(f"  Missing {npy_path.name}, skipping")
            continue

        frame = cv2.imread(str(jpg_path))
        npy   = np.load(str(npy_path))
        out   = draw_prediction(frame, npy)

        out_path = viz_dir / f"viz_{frame_id:06d}.jpg"
        cv2.imwrite(str(out_path), out)
        saved += 1

    print(f"  {clip_dir.name}: {saved} images → {viz_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", type=Path, required=True)
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    for clip_dir in sorted(args.clips_dir.glob("clip_*")):
        visualize_clip(clip_dir, args.n_samples)


if __name__ == "__main__":
    main()
