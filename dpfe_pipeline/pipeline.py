#!/usr/bin/env python3
"""
Full indoor floor-plan pipeline.

Pre-requisites in {video_dir}/:
  transition_yolo.csv    (frame_id, direction, method)
  transition_sfuda.csv   (transition_frame, type, ...)

Two layout models are supported via --layout_model:

  horizonnet (default):
    1  merge_transitions     transitions_confirmed.csv
    2  extract_indoor_clips  clips/clip_NNN/rgb/
    3  HorizonNet            clips/clip_NNN/vo_final/hn_mp3d/  (all frames)
    4  SLAM                  clips/clip_NNN/slam_output/
    5  setup_scene           clips/clip_NNN/{frm_ref.txt, …}
    6  DFPE                  dfpe_results/{video_id}_0/
    7  generate_maps

  lgtnet:
    1  merge_transitions     transitions_confirmed.csv
    2  extract_indoor_clips  clips/clip_NNN/rgb/
    3  SLAM                  clips/clip_NNN/slam_output/
    4  setup_scene           clips/clip_NNN/{frm_ref.txt, …}
    5  LGT-Net               clips/clip_NNN/vo_final/hn_mp3d/  (keyframes only)
    6  DFPE                  dfpe_results/{video_id}_0/
    7  generate_maps

Usage (from the repo root, venv active):
    python dpfe_pipeline/pipeline.py \\
        --video ~/lock_direction/VID_20260603_161558_00_097.mp4 \\
        --video_dir data/videos/097

    # Use LGT-Net instead of HorizonNet
    python dpfe_pipeline/pipeline.py ... --layout_model lgtnet

    # Resume from stage 3
    python dpfe_pipeline/pipeline.py ... --from_stage 3

    # Run only stages 1-2 (useful for new videos)
    python dpfe_pipeline/pipeline.py ... --to_stage 2
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── Python interpreters ────────────────────────────────────────────────────────
HERE       = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from dpfe_paths import VENV_PY, VIDEOS_DIR, RESULTS_DEFAULT
CONDA_DFPE   = ["conda", "run", "-n", "dfpe",   "python"]
CONDA_LGTNET = ["conda", "run", "-n", "lgtnet", "python"]


# ── helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list, cwd=None, desc=""):
    label = desc or " ".join(str(c) for c in cmd[:4])
    print(f"\n  >> {label}")
    result = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        print(f"\n  ERROR: stage failed (exit {result.returncode})")
        sys.exit(result.returncode)


def clip_dirs(video_dir: Path) -> list:
    clips = video_dir / "clips"
    if not clips.exists():
        return []
    return sorted(clips.glob("clip_*"))


def stage_done(tag: str, paths) -> bool:
    """Return True if all paths exist (stage output is already there)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    missing = [p for p in paths if not Path(p).exists()]
    if not missing:
        print(f"  [skip] stage {tag}: outputs already exist")
        return True
    return False


# ── stages ───────────────────────────────────────────────────────────────────

def stage1_merge(video_dir: Path):
    out = video_dir / "transitions_confirmed.csv"
    if stage_done("1 merge_transitions", out):
        return
    run(
        [VENV_PY, HERE / "merge_transitions.py",
         "--yolo_csv",   video_dir / "transition_yolo.csv",
         "--sfuda_csv",  video_dir / "transition_sfuda.csv",
         "--output_csv", out],
        desc="Stage 1: merge transitions",
    )


def stage2_extract(video_dir: Path, video_path: Path):
    out = video_dir / "clips" / "indoor_segments.csv"
    if stage_done("2 extract_indoor_clips", out):
        return
    run(
        [VENV_PY, HERE / "extract_indoor_clips.py",
         "--video",           video_path,
         "--transitions_csv", video_dir / "transitions_confirmed.csv",
         "--output_dir",      video_dir / "clips"],
        desc="Stage 2: extract indoor clips",
    )


def _run_horizonnet(video_dir: Path):
    clips = clip_dirs(video_dir)
    if not clips:
        print("  no clips found, skipping HorizonNet")
        return
    all_npy_done = all(
        list((c / "vo_final" / "hn_mp3d").glob("*.npy"))
        for c in clips
    )
    if all_npy_done:
        print("  [skip] HorizonNet: npy files already exist")
    else:
        run(
            CONDA_DFPE + [HERE / "run_horizonnet.py",
                          "--clips_dir", video_dir / "clips"],
            desc="HorizonNet inference",
        )
    all_viz_done = all(list((c / "viz").glob("viz_*.jpg")) for c in clips)
    if all_viz_done:
        print("  [skip] HorizonNet viz: already done")
    else:
        run(
            [VENV_PY, HERE / "visualize_horizonnet.py",
             "--clips_dir", video_dir / "clips"],
            desc="HorizonNet visualisation",
        )


def _run_slam(video_dir: Path, fps: float):
    for clip_dir in clip_dirs(video_dir):
        traj = clip_dir / "slam_output" / "frame_trajectory.txt"
        if traj.exists():
            print(f"  [skip] SLAM: {clip_dir.name} already done")
            continue
        run(
            [VENV_PY, HERE / "run_slam.py", "--clip_dir", clip_dir, "--fps", str(fps)],
            desc=f"SLAM {clip_dir.name}",
        )


def _run_setup(video_dir: Path, fps: float):
    for clip_dir in clip_dirs(video_dir):
        if (clip_dir / "frm_ref.txt").exists():
            print(f"  [skip] setup_scene: {clip_dir.name} already done")
            continue
        slam_dir = clip_dir / "slam_output"
        if not slam_dir.exists():
            print(f"  WARNING: no slam_output for {clip_dir.name}, skipping")
            continue
        run(
            [VENV_PY, HERE / "setup_scene.py",
             "--slam_dir", slam_dir, "--clip_dir", clip_dir, "--fps", str(fps)],
            desc=f"setup_scene {clip_dir.name}",
        )


def _run_lgtnet(video_dir: Path):
    clips = clip_dirs(video_dir)
    if not clips:
        print("  no clips found, skipping LGT-Net")
        return
    all_done = all(
        (c / "vo_final" / "keyframe_list.txt").exists() and all(
            (c / "vo_final" / "lgt_mp3d" / f"{k}.npy").exists()
            for k in [int(x) for x in
                      (c / "vo_final" / "keyframe_list.txt").read_text().splitlines()
                      if x.strip()]
        )
        for c in clips
    )
    if all_done:
        print("  [skip] LGT-Net: all keyframe npy files already exist in lgt_mp3d/")
        return
    run(
        CONDA_LGTNET + [HERE / "run_lgtnet.py",
                        "--clips_dir", video_dir / "clips"],
        desc="LGT-Net inference",
    )


# ── named stage sequences per layout model ────────────────────────────────────

def _horizonnet_stages(video_dir, video_path, results_root, fps):
    return {
        1: lambda: stage1_merge(video_dir),
        2: lambda: stage2_extract(video_dir, video_path),
        3: lambda: _run_horizonnet(video_dir),
        4: lambda: _run_slam(video_dir, fps),
        5: lambda: _run_setup(video_dir, fps),
        6: lambda: stage6_dfpe(video_dir, results_root),
        7: lambda: stage7_maps(video_dir, results_root),
    }


def _lgtnet_stages(video_dir, video_path, results_root, fps):
    return {
        1: lambda: stage1_merge(video_dir),
        2: lambda: stage2_extract(video_dir, video_path),
        3: lambda: _run_slam(video_dir, fps),
        4: lambda: _run_setup(video_dir, fps),
        5: lambda: _run_lgtnet(video_dir),
        6: lambda: stage6_dfpe(video_dir, results_root, ly_model="lgt_mp3d"),
        7: lambda: stage7_maps(video_dir, results_root),
    }


def stage6_dfpe(video_dir: Path, results_root: Path, ly_model: str = "hn_mp3d"):
    clips = clip_dirs(video_dir)
    vid_id = video_dir.name
    all_done = all(
        (results_root / f"{vid_id}_{i}" / f"{vid_id}_final_fp.png").exists()
        for i in range(len(clips))
    )
    if all_done and clips:
        print("  [skip] Stage 6 DFPE: results already exist")
        return
    run(
        CONDA_DFPE + [HERE / "run_dfpe.py",
                      "--video_dirs",    video_dir,
                      "--results_root",  results_root,
                      "--ly_model",      ly_model],
        desc="Stage 6: DFPE",
    )


def stage7_maps(video_dir: Path, results_root: Path):
    vid_id = video_dir.name
    clips = clip_dirs(video_dir)
    all_done = all(
        (results_root / f"{vid_id}_{i}" / f"{vid_id}_{i}_map_final_fp.png").exists()
        for i in range(len(clips))
    )
    if all_done and clips:
        print("  [skip] Stage 7 maps: already exist")
        return
    for i in range(len(clips)):
        scene_dir = results_root / f"{vid_id}_{i}"
        run(
            [VENV_PY, HERE / "generate_maps.py",
             "--results_dir", scene_dir],
            desc=f"Stage 7: maps {vid_id}_{i}",
        )


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end indoor floor-plan pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video", type=Path, required=True,
        help="Input video file (.mp4)")
    parser.add_argument(
        "--video_dir", type=Path, default=None,
        help="Per-video pipeline dir (default: data/videos/{video_id}). "
             "Must contain transition_yolo.csv and transition_sfuda.csv.")
    parser.add_argument(
        "--results_root", type=Path,
        default=RESULTS_DEFAULT,
        help="Where to write DFPE floor-plan outputs")
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Video frame rate (default 30)")
    parser.add_argument(
        "--layout_model", choices=["horizonnet", "lgtnet"], default="horizonnet",
        help="Layout estimation model (default: horizonnet). "
             "horizonnet: stages 3=HorizonNet 4=SLAM 5=setup. "
             "lgtnet: stages 3=SLAM 4=setup 5=LGT-Net.")
    parser.add_argument(
        "--from_stage", type=int, default=1, metavar="N",
        help="Start from this stage (1–7); earlier stages are skipped")
    parser.add_argument(
        "--to_stage", type=int, default=7, metavar="N",
        help="Stop after this stage (1–7)")
    args = parser.parse_args()

    # Resolve video_dir from video filename if not given
    if args.video_dir is None:
        # Extract numeric ID from filename, e.g. VID_..._097.mp4 → "097"
        stem = args.video.stem              # "VID_20260603_161558_00_097"
        vid_id = stem.rsplit("_", 1)[-1]   # "097"
        args.video_dir = VIDEOS_DIR / vid_id

    video_dir   = args.video_dir.resolve()
    video_path  = args.video.resolve()
    results_root = args.results_root.resolve()
    fps          = args.fps

    print(f"\n{'='*60}")
    print(f"Pipeline: {video_path.name}")
    print(f"  video_dir:    {video_dir}")
    print(f"  results_root: {results_root}")
    print(f"  layout_model: {args.layout_model}")
    print(f"  stages:       {args.from_stage}–{args.to_stage}")
    print(f"{'='*60}")

    video_dir.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    if args.layout_model == "lgtnet":
        stages = _lgtnet_stages(video_dir, video_path, results_root, fps)
    else:
        stages = _horizonnet_stages(video_dir, video_path, results_root, fps)

    for n in range(args.from_stage, min(args.to_stage, 7) + 1):
        print(f"\n{'─'*60}")
        print(f"Stage {n}")
        stages[n]()

    print(f"\n{'='*60}")
    print(f"Done. Results in {results_root}/{video_dir.name}_*/")


if __name__ == "__main__":
    main()
