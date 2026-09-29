#!/usr/bin/env python3
"""
Stage 6: Run 360DFPE on each indoor clip.

For each clip produces:
  {results_dir}/{scene}_{version}/
    final_fp.png           ← estimated floor plan
    scale_recovery.jpg
    gt_rooms.png
    results.csv

DataManager expects:
  {mp3d_fpe_dir}/{scene}/{version}/
    frm_ref.txt
    label.json
    pcl.ply
    vo_final/
      cam_pose_estimated.csv
      keyframe_list.txt
      hn_mp3d/{kf_id}.npy

We set mp3d_fpe_dir = {video_dir}  and create a symlink
  {video_dir}/{version_str}/ -> clips/clip_{N}/
so DFPE finds everything at the right path.

Usage:
    conda run -n dfpe python dpfe_pipeline/run_dfpe.py \\
        --video_dirs data/videos/097 data/videos/098
"""

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path

from dpfe_paths import DFPE_ROOT, RESULTS_DEFAULT
BASE_CONFIG = DFPE_ROOT / "config" / "config.yaml"


def write_config(cfg_out: Path, mp3d_fpe_dir: Path, ly_model: str = "hn_mp3d"):
    """Copy the base config, overriding path.mp3d_fpe_dir and data.ly_model."""
    lines = BASE_CONFIG.read_text().splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("path.mp3d_fpe_dir"):
            new_lines.append(f"path.mp3d_fpe_dir: {mp3d_fpe_dir}")
        elif line.startswith("data.ly_model"):
            new_lines.append(f"data.ly_model: '{ly_model}'")
        else:
            new_lines.append(line)
    cfg_out.write_text("\n".join(new_lines) + "\n")
    print(f"  Config: {cfg_out}  (ly_model={ly_model})")


def make_scene_symlink(video_dir: Path, clip_dir: Path, scene: str, version: str):
    """
    Create {video_dir}/{scene}/{version} -> ../../clips/clip_NNN
    so DataManager finds the data at {mp3d_fpe_dir}/{scene}/{version}.
    """
    scene_parent = video_dir / scene
    scene_link = scene_parent / version
    scene_parent.mkdir(exist_ok=True)

    # Relative path from scene_parent/ to clips/clip_NNN
    rel_target = Path("..") / "clips" / clip_dir.name

    if scene_link.is_symlink():
        scene_link.unlink()
    elif scene_link.exists():
        print(f"  WARNING: {scene_link} exists as a real dir, skipping symlink creation")
        return scene_link

    scene_link.symlink_to(rel_target)
    print(f"  Symlink: {scene_link} -> {rel_target}")
    return scene_link


def run_dfpe_on_clip(video_dir: Path, clip_dir: Path, clip_idx: int, results_base: Path,
                     ly_model: str = "hn_mp3d"):
    video_dir = video_dir.resolve()
    clip_dir = clip_dir.resolve()
    results_base = results_base.resolve()

    scene = video_dir.name        # e.g. "097"
    version = str(clip_idx)      # "0", "1", ...
    scene_name = f"{scene}_{version}"  # e.g. "097_0"

    print(f"\n{'='*60}")
    print(f"DFPE: {scene_name}  ({clip_dir.name})")

    # 1. Config — must be absolute so main_eval_scene.py (cwd=DFPE_ROOT) finds it
    cfg_path = results_base / f"dfpe_config_{scene}.yaml"
    write_config(cfg_path, video_dir, ly_model)

    # 2. Symlink so DFPE finds the data at {video_dir}/{scene}/{version}/
    make_scene_symlink(video_dir, clip_dir, scene, version)

    # 3. main_eval_scene.py appends "{scene}_{version}" to --results itself,
    #    so pass results_base (not a pre-created subdir).
    results_base.mkdir(parents=True, exist_ok=True)
    print(f"  Results: {results_base / scene_name}")

    # 4. Run main_eval_scene.py inside the conda dfpe env
    cmd = [
        "conda", "run", "-n", "dfpe",
        "python", str(DFPE_ROOT / "main_eval_scene.py"),
        "--scene_name", scene_name,
        "--results",    str(results_base),
        "--cfg",        str(cfg_path),
    ]

    print(f"  CMD: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(DFPE_ROOT))

    if result.returncode != 0:
        print(f"\n  DFPE failed (exit {result.returncode}) for {scene_name}")
        return False

    print(f"\n  Done: {scene_name} → {results_base / scene_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Stage 6: Run 360DFPE on indoor clips")
    parser.add_argument(
        "--video_dirs", type=Path, nargs="+", required=True,
        help="Per-video pipeline dirs, e.g. data/videos/097 data/videos/098"
    )
    parser.add_argument(
        "--results_root", type=Path, default=RESULTS_DEFAULT,
        help="Root dir for DFPE outputs"
    )
    parser.add_argument(
        "--ly_model", default="hn_mp3d",
        help="Layout model directory name inside vo_final/ (default: hn_mp3d). "
             "Use lgt_mp3d for LGT-Net."
    )
    args = parser.parse_args()

    args.results_root.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for video_dir in args.video_dirs:
        video_dir = video_dir.resolve()
        clips_dir = video_dir / "clips"
        if not clips_dir.exists():
            print(f"WARNING: no clips/ in {video_dir}, skipping")
            continue

        clip_dirs = sorted(clips_dir.glob("clip_*"))
        for clip_idx, clip_dir in enumerate(clip_dirs):
            ok = run_dfpe_on_clip(video_dir, clip_dir, clip_idx, args.results_root, args.ly_model)
            if not ok:
                all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
