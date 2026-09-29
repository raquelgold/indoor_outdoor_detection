#!/usr/bin/env python3
"""
Stage 5: Build DFPE scene directory from SLAM output.

Given SLAM output (frame_trajectory.txt + keyframe_trajectory.txt) and a
clip directory (already containing rgb/ and vo_final/hn_mp3d/ from Stages 2–3),
produces the full scaffold that direct_360_FPE needs:

  clip_dir/
    frm_ref.txt                  ← dense per-frame poses from frame_trajectory.txt
    label.json                   ← single room = padded convex hull of keyframe XZ
    pcl.ply                      ← minimal dummy point cloud (xyz + rgb)
    vo_final/
      cam_pose_estimated.csv     ← sparse keyframe poses from keyframe_trajectory.txt
      keyframe_list.txt          ← keyframe IDs (one per line)

DFPE indexes frm_ref.txt as: poses_gt[kf - 1]  (0-based into row order).
So frm_ref.txt must have rows 1..max(keyframe_id) in order.

Usage:
    python dpfe_pipeline/setup_scene.py \
        --slam_dir  data/videos/097/clips/clip_000/slam_output \
        --clip_dir  data/videos/097/clips/clip_000 \
        --fps       30
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull

FPS_DEFAULT = 30.0
ROOM_PADDING = 1.0  # metres of padding outward from convex hull


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_tum(path: Path) -> list:
    """Read TUM-format file → list of [ts, tx, ty, tz, qx, qy, qz, qw]."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        vals = [float(v) for v in line.split()]
        if len(vals) >= 8:
            rows.append(vals)
    return rows


def _to_fid(ts: float, ts0: float, fps: float) -> int:
    """SLAM timestamp → 1-based frame ID."""
    return round((ts - ts0) * fps) + 1


# ── vo_final/cam_pose_estimated.csv + keyframe_list.txt ──────────────────────

def write_vo_files(slam_dir: Path, clip_dir: Path, fps: float) -> list:
    rows = _parse_tum(slam_dir / "keyframe_trajectory.txt")
    ts0 = rows[0][0]

    kf_data = sorted(
        [(_to_fid(r[0], ts0, fps), r[1:]) for r in rows],
        key=lambda x: x[0],
    )
    kf_ids = [fid for fid, _ in kf_data]

    vo_dir = clip_dir / "vo_final"
    vo_dir.mkdir(parents=True, exist_ok=True)

    with open(vo_dir / "cam_pose_estimated.csv", "w") as f:
        for fid, (tx, ty, tz, qx, qy, qz, qw) in kf_data:
            f.write(f"{fid} {tx:.10g} {ty:.10g} {tz:.10g} "
                    f"{qx:.10g} {qy:.10g} {qz:.10g} {qw:.10g}\n")
    print(f"  cam_pose_estimated.csv  ({len(kf_ids)} keyframes)")

    (vo_dir / "keyframe_list.txt").write_text(
        "\n".join(str(k) for k in kf_ids) + "\n"
    )
    print(f"  keyframe_list.txt")

    return kf_ids


# ── frm_ref.txt ───────────────────────────────────────────────────────────────

def write_frm_ref(slam_dir: Path, clip_dir: Path, fps: float, max_kf_id: int):
    rows = _parse_tum(slam_dir / "frame_trajectory.txt")
    ts0 = rows[0][0]

    # Map every tracked frame to its pose
    poses = {}
    for r in rows:
        fid = _to_fid(r[0], ts0, fps)
        poses[fid] = r[1:]  # [tx ty tz qx qy qz qw]

    # Fill gaps (SLAM dropout) with nearest tracked frame
    tracked = sorted(poses)
    for fid in range(1, max_kf_id + 1):
        if fid not in poses:
            nearest = min(tracked, key=lambda x: abs(x - fid))
            poses[fid] = poses[nearest]

    out = clip_dir / "frm_ref.txt"
    with open(out, "w") as f:
        for fid in range(1, max_kf_id + 1):
            tx, ty, tz, qx, qy, qz, qw = poses[fid]
            f.write(f"{fid} {tx:.10g} {ty:.10g} {tz:.10g} "
                    f"{qx:.10g} {qy:.10g} {qz:.10g} {qw:.10g}\n")
    print(f"  frm_ref.txt  ({max_kf_id} rows, "
          f"{max_kf_id - len(tracked)} gap-filled)")


# ── label.json ────────────────────────────────────────────────────────────────

def write_label_json(slam_dir: Path, clip_dir: Path, fps: float):
    rows = _parse_tum(slam_dir / "keyframe_trajectory.txt")

    # XZ plane is the floor in SLAM coordinates (Y = height)
    xz = np.array([[r[1], r[3]] for r in rows])

    # Build room polygon as padded convex hull of keyframe positions
    if len(xz) >= 3:
        hull = ConvexHull(xz)
        hull_pts = xz[hull.vertices]
    else:
        # Fewer than 3 points — use bounding box
        mn, mx = xz.min(axis=0), xz.max(axis=0)
        hull_pts = np.array([mn, [mx[0], mn[1]], mx, [mn[0], mx[1]]])

    centroid = hull_pts.mean(axis=0)
    padded = []
    for pt in hull_pts:
        d = pt - centroid
        norm = np.linalg.norm(d)
        padded.append((pt + ROOM_PADDING * d / norm if norm > 0 else pt).tolist())

    # axis_corners: the two hull points furthest apart
    dists = np.linalg.norm(hull_pts[:, None] - hull_pts[None, :], axis=-1)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    axis_corners = [hull_pts[i].tolist(), hull_pts[j].tolist()]

    label = {
        "version": "v0.0",
        "room_corners": [[[str(round(p[0], 6)), str(round(p[1], 6))] for p in padded]],
        "axis_corners": [
            [str(round(axis_corners[0][0], 6)), str(round(axis_corners[0][1], 6))],
            [str(round(axis_corners[1][0], 6)), str(round(axis_corners[1][1], 6))],
        ],
    }

    (clip_dir / "label.json").write_text(json.dumps(label, indent=2))
    print(f"  label.json  (1 room, {len(padded)}-corner convex hull + {ROOM_PADDING}m pad)")


# ── hn_mp3d keyframe symlinks ─────────────────────────────────────────────────

def link_hn_npy(clip_dir: Path, kf_ids: list):
    """
    HorizonNet npy files are named by VIDEO frame ID.
    Create per-kf_id symlinks: hn_mp3d/{kf_id}.npy → {video_frame_id}.npy
    so DFPE can find the right prediction for each keyframe.
    LGT-Net uses its own separate lgt_mp3d/ directory and does not touch this.
    """
    rgb_dir = clip_dir / "rgb"
    jpg_files = sorted(rgb_dir.glob("frame_*.jpg"))
    if not jpg_files:
        print("  WARNING: no frames in rgb/, skipping hn_mp3d symlinks")
        return
    clip_start = int(jpg_files[0].stem.split("_")[1])

    hn_dir = clip_dir / "vo_final" / "hn_mp3d"
    created = 0
    for kf_id in kf_ids:
        video_fid = clip_start + kf_id - 1
        src = Path(f"{video_fid}.npy")   # relative — stays valid after mv
        link = hn_dir / f"{kf_id}.npy"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
        created += 1
    print(f"  hn_mp3d symlinks: {created} keyframes (clip_start={clip_start})")


# ── pcl.ply ───────────────────────────────────────────────────────────────────

def write_pcl_ply(clip_dir: Path):
    # read_ply in DFPE reads columns 3:6 as RGB, so colour properties are required
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
    print(f"  pcl.ply  (dummy, 4 pts with RGB)")


# ── entry point ───────────────────────────────────────────────────────────────

def setup_scene(slam_dir: Path, clip_dir: Path, fps: float):
    print(f"\n{'='*50}")
    print(f"Setting up DFPE scene: {clip_dir.name}")
    print(f"  SLAM output: {slam_dir}")

    kf_ids = write_vo_files(slam_dir, clip_dir, fps)
    link_hn_npy(clip_dir, kf_ids)
    write_frm_ref(slam_dir, clip_dir, fps, max(kf_ids))
    write_label_json(slam_dir, clip_dir, fps)
    write_pcl_ply(clip_dir)

    print(f"  Scene ready — {len(kf_ids)} keyframes, max id={max(kf_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5: Build DFPE scene from SLAM output")
    parser.add_argument("--slam_dir", type=Path, required=True,
                        help="SLAM output dir with frame_trajectory.txt and keyframe_trajectory.txt")
    parser.add_argument("--clip_dir", type=Path, required=True,
                        help="Clip dir already containing rgb/ and vo_final/hn_mp3d/")
    parser.add_argument("--fps",     type=float, default=FPS_DEFAULT)
    args = parser.parse_args()

    setup_scene(args.slam_dir, args.clip_dir, args.fps)
