#!/usr/bin/env python3
"""
Generate the top-down SLAM-recovery animation GIF from a DFPE coords JSON.

For each {scene}_coords.json produces {scene}_map_animation.gif in the same
directory. Each GIF frame shows the wall-plane segments accumulated so far
(cyan), the camera trail (dim white) and the current camera position.

Usage:
    python dpfe_pipeline/generate_map_animation.py \\
        --results_dir outputs/dfpe_results/097_0

    # Or point at the JSON directly
    python dpfe_pipeline/generate_map_animation.py \\
        --coords_json outputs/dfpe_results/097_0/097_0_coords.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection


def build_animation(data, out_path, fps=20, step=1):
    """
    data     : parsed JSON dict
    out_path : output GIF path
    fps      : frames per second in the GIF
    step     : render every Nth keyframe (1 = all frames)
    """
    planes_by_frame  = data["wall_planes"]["frames"]       # list of {frame_idx, segments}
    cam_positions    = data["camera_trajectory"]["positions"]  # list of {frame_idx, pos_xz}
    scene            = data["scene"]
    vo               = data["vo_scale"]

    # Align both lists by frame index (they should already match)
    cam_by_idx = {e["frame_idx"]: e["pos_xz"] for e in cam_positions}

    # Work out world bounds from all segments + all cam positions
    all_seg_pts = np.array(
        [p for f in planes_by_frame for seg in f["segments"] for p in seg]
    )
    all_cam_pts = np.array([e["pos_xz"] for e in cam_positions])
    all_pts     = np.vstack([all_seg_pts, all_cam_pts])

    pad   = 0.6
    xmin, xmax = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    zmin, zmax = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
    span  = max(xmax - xmin, zmax - zmin)
    xmid  = (xmin + xmax) / 2
    zmid  = (zmin + zmax) / 2
    xlim  = (xmid - span / 2, xmid + span / 2)
    zlim  = (zmid - span / 2, zmid + span / 2)

    # Sub-sample frames
    frames_to_render = planes_by_frame[::step]
    n_render = len(frames_to_render)

    # Segments and cam positions accumulated up to each rendered frame
    # Pre-compute cumulative segment lists and cam trails for speed
    cum_segs = []        # list[list of (2,2) arrays]
    cum_cams = []        # list of (x,z) tuples (trail)
    seg_pool = []
    cam_pool = []

    all_frame_idxs = sorted({f["frame_idx"] for f in planes_by_frame} |
                            set(cam_by_idx.keys()))

    rendered_frame_idxs = {f["frame_idx"] for f in frames_to_render}
    frame_lookup = {f["frame_idx"]: f["segments"] for f in planes_by_frame}

    # Walk through all frame indices in order, only save snapshots at rendered frames
    seg_pool_running = []
    cam_pool_running = []
    for fidx in sorted(all_frame_idxs):
        if fidx in frame_lookup:
            seg_pool_running.extend(frame_lookup[fidx])
        if fidx in cam_by_idx:
            cam_pool_running.append(cam_by_idx[fidx])
        if fidx in rendered_frame_idxs:
            cum_segs.append(list(seg_pool_running))
            cum_cams.append(list(cam_pool_running))

    # ------------------------------------------------------------------ #
    # Set up the figure
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal")
    ax.set_xlabel("X  (metres)", color="white", fontsize=10)
    ax.set_ylabel("Z  (metres)", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334466")

    title = ax.set_title("", color="white", fontsize=11, pad=10)

    seg_col  = LineCollection([], colors="#00d4ff", linewidths=0.9, alpha=0.55)
    trail_ln,  = ax.plot([], [], color="white",   alpha=0.25, linewidth=0.8)
    cam_dot,   = ax.plot([], [], "o", color="white", markersize=6, zorder=10)
    ax.add_collection(seg_col)

    # Progress bar (thin rectangle at the bottom)
    bar_bg  = ax.axhline(zlim[0] + 0.05, color="#334466", linewidth=4,
                         xmin=0.01, xmax=0.99, zorder=8)
    bar_fg, = ax.plot([], [], color="#00d4ff", linewidth=4, zorder=9)
    bar_y   = zlim[0] + 0.05 * (zlim[1] - zlim[0])

    def init():
        seg_col.set_segments([])
        trail_ln.set_data([], [])
        cam_dot.set_data([], [])
        bar_fg.set_data([], [])
        title.set_text("")
        return seg_col, trail_ln, cam_dot, bar_fg, title

    def update(i):
        segs = cum_segs[i]
        cams = cum_cams[i]
        fidx = frames_to_render[i]["frame_idx"]

        # Wall segments as a LineCollection
        lc_segs = [np.array(s) for s in segs]   # each (2,2): [[x1,z1],[x2,z2]]
        seg_col.set_segments(lc_segs)

        # Camera trail
        if cams:
            xs = [c[0] for c in cams]
            zs = [c[1] for c in cams]
            trail_ln.set_data(xs, zs)
            cam_dot.set_data([xs[-1]], [zs[-1]])
        else:
            trail_ln.set_data([], [])
            cam_dot.set_data([], [])

        # Progress bar
        progress = (i + 1) / n_render
        bar_x0   = xlim[0] + 0.01 * (xlim[1] - xlim[0])
        bar_x1   = xlim[0] + (0.01 + 0.98 * progress) * (xlim[1] - xlim[0])
        bar_fg.set_data([bar_x0, bar_x1], [bar_y, bar_y])

        title.set_text(
            f"{scene}  —  wall planes  ·  frame {fidx}\n"
            f"VO scale = {vo:.4f}   ·   {len(segs)} segments so far"
        )
        return seg_col, trail_ln, cam_dot, bar_fg, title

    ani = animation.FuncAnimation(
        fig, update, frames=n_render,
        init_func=init, blit=True, interval=1000 / fps,
    )

    print(f"Rendering {n_render} frames → {out_path}")
    ani.save(out_path, writer="pillow", fps=fps,
             savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    print(f"Saved  {out_path}")


def generate_animation(coords_json: Path, fps: float = 20.0, max_frames: int = 150):
    out_dir = coords_json.parent
    with open(coords_json) as f:
        data = json.load(f)
    scene = data["scene"]
    n_frames = len(data["wall_planes"]["frames"])
    step = max(1, n_frames // max_frames)
    out_path = out_dir / f"{scene}_map_animation.gif"
    build_animation(data, str(out_path), fps=fps, step=step)


def main():
    parser = argparse.ArgumentParser(description="Generate DFPE map-recovery animation GIF")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--coords_json", type=Path,
                        help="Path to a specific *_coords.json file")
    group.add_argument("--results_dir", type=Path,
                        help="Results directory; all *_coords.json files inside are processed")
    args = parser.parse_args()

    if args.coords_json:
        generate_animation(args.coords_json)
    else:
        jsons = sorted(args.results_dir.glob("*_coords.json"))
        if not jsons:
            print(f"  No *_coords.json found in {args.results_dir}")
            return
        for j in jsons:
            generate_animation(j)


if __name__ == "__main__":
    main()
