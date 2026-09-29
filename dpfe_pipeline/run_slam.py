#!/usr/bin/env python3
"""
Stage 4: Run stella_vslam on an indoor clip to produce camera trajectories.

Outputs (in clip_dir/slam_output/):
  frame_trajectory.txt      ← dense per-frame poses  (used by setup_scene.py → frm_ref.txt)
  keyframe_trajectory.txt   ← sparse keyframe poses  (used by setup_scene.py → cam_pose_estimated.csv)
  map.msg                   ← saved map (for inspection / re-use)

--start-timestamp 0 is set so timestamps = frame_index / fps, making
setup_scene.py frame-id conversion (round(ts * fps) + 1) give clean 1-based IDs.

Usage:
    python dpfe_pipeline/run_slam.py --clip_dir data/videos/097/clips/clip_000
    python dpfe_pipeline/run_slam.py --clip_dir data/videos/098/clips/clip_000
"""

import argparse
import subprocess
import sys
from pathlib import Path

from dpfe_paths import SLAM_BIN, ORB_VOCAB as VOCAB, SLAM_CONFIG_DIR

CONFIG     = SLAM_CONFIG_DIR / "config_fps30_5760.yaml"

CONFIG_BY_RESOLUTION_FPS = {
    (5760, 2880, 30): CONFIG,
    (5760, 2880, 24): SLAM_CONFIG_DIR / "config.yaml",
    (1920, 960, 30):  SLAM_CONFIG_DIR / "config_fps30_1920.yaml",
}


def _pick_config(rgb_dir: Path, config: Path, fps: float = 30.0) -> Path:
    """Auto-select the config matching the actual frame resolution + declared
    fps, unless the caller passed an explicit override (config != CONFIG default)."""
    if config != CONFIG:
        return config
    frames = sorted(rgb_dir.glob("*.jpg"))
    if not frames:
        return config
    import cv2
    img = cv2.imread(str(frames[0]))
    if img is None:
        return config
    h, w = img.shape[:2]
    matched = CONFIG_BY_RESOLUTION_FPS.get((w, h, round(fps)))
    if matched and matched != config:
        print(f"  Detected frame resolution {w}x{h} @ {fps}fps -> using {matched.name}")
        return matched
    return config


def run_slam(clip_dir: Path, config: Path = CONFIG, extra_args: list = None, fps: float = 30.0):
    clip_dir   = clip_dir.resolve()
    rgb_dir    = clip_dir / "rgb"
    slam_out   = clip_dir / "slam_output"
    slam_out.mkdir(parents=True, exist_ok=True)

    config = _pick_config(rgb_dir, config, fps)

    map_path = slam_out / "map.msg"

    cmd = [
        str(SLAM_BIN),
        "--vocab",           str(VOCAB),
        "--img-dir",         str(rgb_dir),
        "--config",          str(config),
        "--start-timestamp", "0",
        "--no-sleep",
        "--auto-term",
        "--viewer",          "none",
        "--eval-log-dir",    str(slam_out),
        "--map-db-out",      str(map_path),
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"SLAM: {clip_dir.name}")
    print(f"  rgb:    {rgb_dir}  ({len(list(rgb_dir.glob('*.jpg')))} frames)")
    print(f"  output: {slam_out}")
    print(f"  cmd:    {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(clip_dir))

    if result.returncode != 0:
        print(f"\n  SLAM failed (exit {result.returncode})")
        sys.exit(result.returncode)

    # Verify expected outputs exist
    for fname in ("frame_trajectory.txt", "keyframe_trajectory.txt"):
        out = slam_out / fname
        if out.exists():
            lines = sum(1 for _ in open(out))
            print(f"  {fname}: {lines} lines")
        else:
            print(f"  WARNING: {fname} not found — SLAM may have lost tracking entirely")

    print(f"\n  Done: {clip_dir.name}")
    return slam_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Run stella_vslam on an indoor clip")
    parser.add_argument("--clip_dir", type=Path, required=True,
                        help="Clip directory containing rgb/")
    parser.add_argument("--config", type=Path, default=CONFIG,
                        help="stella_vslam config yaml")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Source video fps (used for config auto-selection)")
    args = parser.parse_args()

    run_slam(args.clip_dir, args.config, fps=args.fps)
