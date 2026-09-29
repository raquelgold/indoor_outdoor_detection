#!/usr/bin/env python3
"""
Stage 7: Generate top-down map images from DFPE coords JSON.

For each {scene}_coords.json produces three PNGs in the same directory:
  {scene}_map_scale_recovery.png  – wall boundary occupancy heatmap
  {scene}_map_final_fp.png        – estimated room corner polygons
  {scene}_map_wall_planes.png     – RANSAC-fitted wall segments

With --unified, also:
  {scene}_map_unified_fp.png      – all wall segments as one space, hull corners labelled

Usage:
    python dpfe_pipeline/generate_maps.py \\
        --results_dir outputs/dfpe_results/097_0

    # Or point at the JSON directly
    python dpfe_pipeline/generate_maps.py \\
        --coords_json outputs/dfpe_results/097_0/097_0_coords.json
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from pathlib import Path
from scipy.spatial import ConvexHull

ROOM_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
]


def _equal_padded_axis(ax, xs, zs, pad=0.5):
    xmin, xmax = min(xs) - pad, max(xs) + pad
    zmin, zmax = min(zs) - pad, max(zs) + pad
    span = max(xmax - xmin, zmax - zmin)
    xmid = (xmin + xmax) / 2
    zmid = (zmin + zmax) / 2
    ax.set_xlim(xmid - span / 2, xmid + span / 2)
    ax.set_ylim(zmid - span / 2, zmid + span / 2)
    ax.set_aspect("equal")


def plot_wall_planes(data: dict, out_path: Path):
    segments = data["wall_planes"]["segments"]
    all_pts = np.array([p for seg in segments for p in seg])
    xs_all, zs_all = all_pts[:, 0], all_pts[:, 1]

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    n = len(segments)
    alpha = max(0.03, min(0.12, 8.0 / n))
    for (x1, z1), (x2, z2) in segments:
        ax.plot([x1, x2], [z1, z2], color="#00d4ff", alpha=alpha,
                linewidth=1.0, solid_capstyle="round")

    ax.set_xlabel("X  (metres)", color="white", fontsize=11)
    ax.set_ylabel("Z  (metres)", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334466")

    scene = data["scene"]
    vo = data["vo_scale"]
    ax.set_title(
        f"{scene}  —  RANSAC Wall Planes\n"
        f"VO scale = {vo:.4f}   ·   {n} segments",
        color="white", fontsize=12, pad=12,
    )

    _equal_padded_axis(ax, xs_all, zs_all, pad=0.4)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_scale_recovery(data: dict, out_path: Path):
    boundaries = data["scale_recovery"]["wall_boundaries"]
    all_pts = np.array([p for poly in boundaries for p in poly])
    xs, zs = all_pts[:, 0], all_pts[:, 1]

    grid_size = 0.1
    pad = 10 * grid_size
    x_bins = np.arange(xs.min() - pad, xs.max() + pad + grid_size, grid_size)
    z_bins = np.arange(zs.min() - pad, zs.max() + pad + grid_size, grid_size)
    grid, x_edges, z_edges = np.histogram2d(xs, zs, bins=(x_bins, z_bins))
    grid = np.clip(grid, 0, 20) / 20.0

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    im = ax.imshow(
        grid.T, origin="lower",
        extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
        cmap="hot", interpolation="nearest", aspect="equal", vmin=0, vmax=1,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("normalised wall density", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("X  (metres)", color="white", fontsize=11)
    ax.set_ylabel("Z  (metres)", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334466")

    scene = data["scene"]
    vo = data["vo_scale"]
    n = len(boundaries)
    ax.set_title(
        f"{scene}  —  Scale Recovery  Occupancy Grid\n"
        f"VO scale = {vo:.4f}   ·   {n} frames   ·   cell = {int(grid_size*100)} cm",
        color="white", fontsize=12, pad=12,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_final_fp(data: dict, out_path: Path):
    rooms = data["final_fp"]["rooms"]
    scene = data["scene"]

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f5f5f0")

    all_xs, all_zs = [], []
    legend_handles = []

    for room in rooms:
        corners = np.array(room["corners_xz"])
        rid = room["room_id"]
        color = ROOM_COLORS[rid % len(ROOM_COLORS)]
        closed = np.vstack([corners, corners[0]])
        xs, zs = corners[:, 0], corners[:, 1]
        all_xs.extend(xs); all_zs.extend(zs)

        ax.add_patch(Polygon(corners, closed=True,
                             facecolor=color, alpha=0.35,
                             edgecolor=color, linewidth=2))
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2)
        ax.scatter(xs, zs, color=color, s=60, zorder=5,
                   edgecolors="white", linewidths=0.8)

        for x, z in zip(xs, zs):
            ax.annotate(f"({x:.2f}, {z:.2f})", xy=(x, z),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=6.5, color=color, zorder=6)

        cx, cz = corners[:, 0].mean(), corners[:, 1].mean()
        ax.text(cx, cz, f"R{rid}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=color, zorder=7)

        legend_handles.append(
            mpatches.Patch(facecolor=color, alpha=0.6,
                           label=f"Room {rid}  ({len(corners)} corners)")
        )

    ax.set_xlabel("X  (metres)", fontsize=11)
    ax.set_ylabel("Z  (metres)", fontsize=11)
    ax.set_title(
        f"{scene}  —  Estimated Floor Plan\n"
        f"{len(rooms)} room(s)  ·  world XZ coordinates",
        fontsize=12, pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.4, color="#aaaaaa")
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.85)
    _equal_padded_axis(ax, all_xs, all_zs, pad=0.8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_unified_floor_plan(data: dict, out_path: Path):
    """
    Single unified floor plan from wall_planes segments — no room colours,
    just clean wall lines for the whole space with endpoint coordinates labelled.
    """
    segments = data["wall_planes"]["segments"]
    scene    = data["scene"]
    vo       = data["vo_scale"]

    all_pts = np.array([p for seg in segments for p in seg])
    xs_all, zs_all = all_pts[:, 0], all_pts[:, 1]

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- draw every wall segment ---
    n = len(segments)
    alpha = max(0.04, min(0.18, 10.0 / n))
    for (x1, z1), (x2, z2) in segments:
        ax.plot([x1, x2], [z1, z2],
                color="#1a1a2e", alpha=alpha,
                linewidth=1.2, solid_capstyle="round")

    # --- label only the true convex hull vertices ---
    all_endpoints = np.array([p for seg in segments for p in seg])  # (2N, 2)
    hull = ConvexHull(all_endpoints)
    hull_pts = all_endpoints[hull.vertices]   # the actual outer boundary corners

    for x, z in hull_pts:
        ax.scatter(x, z, s=30, color="#e15759", zorder=6, linewidths=0)
        ax.annotate(
            f"({x:.2f}, {z:.2f})",
            xy=(x, z),
            xytext=(6, 6), textcoords="offset points",
            fontsize=7.5, color="#c0392b", fontweight="bold",
            zorder=7,
        )

    ax.set_xlabel("X  (metres)", fontsize=11)
    ax.set_ylabel("Z  (metres)", fontsize=11)
    ax.set_title(
        f"{scene}  —  Unified Floor Plan  (wall planes)\n"
        f"VO scale = {vo:.4f}   ·   {n} wall segments   ·   whole space, no room separation",
        fontsize=11, pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.3, color="#aaaaaa")
    _equal_padded_axis(ax, xs_all, zs_all, pad=0.6)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def generate_maps(coords_json: Path, unified: bool = False):
    out_dir = coords_json.parent
    with open(coords_json) as f:
        data = json.load(f)
    scene = data["scene"]
    print(f"  Maps for {scene}:")
    plot_scale_recovery(data, out_dir / f"{scene}_map_scale_recovery.png")
    plot_wall_planes(data,    out_dir / f"{scene}_map_wall_planes.png")
    plot_final_fp(data,       out_dir / f"{scene}_map_final_fp.png")
    if unified:
        plot_unified_floor_plan(data, out_dir / f"{scene}_map_unified_fp.png")


def main():
    parser = argparse.ArgumentParser(description="Stage 7: Generate top-down maps from DFPE coords JSON")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--coords_json", type=Path,
                       help="Path to a specific *_coords.json file")
    group.add_argument("--results_dir", type=Path,
                       help="Results directory; all *_coords.json files inside are processed")
    parser.add_argument("--unified", action="store_true",
                        help="Also render the single unified floor plan (no room separation)")
    args = parser.parse_args()

    if args.coords_json:
        generate_maps(args.coords_json, args.unified)
    else:
        jsons = sorted(args.results_dir.glob("*_coords.json"))
        if not jsons:
            print(f"  No *_coords.json found in {args.results_dir}")
            return
        for j in jsons:
            generate_maps(j, args.unified)


if __name__ == "__main__":
    main()
