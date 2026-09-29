"""
Post-process DFPE room polygons to remove overlaps using Shapely.

Strategy:
  - Sort rooms by area, largest first (most spatial coverage = most evidence)
  - Iterate in that order; each room gets all previously-claimed area subtracted
  - If clipping splits a room into multiple pieces, keep the largest piece
  - Rooms that shrink below min_area_m2 are dropped

Rewrites the final_fp section of the coords JSON and regenerates
*_map_final_fp.png.  Everything else (scale_recovery, wall_planes, etc.) is
unchanged.

Usage:
  python clip_room_overlaps.py
"""

import json
import os
import sys
import copy
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "dpfe_pipeline"))
from generate_maps import plot_final_fp
from pathlib import Path

MIN_AREA_M2 = 0.1   # drop rooms smaller than this after clipping

COORD_FILES = {
    "video033_0": "test_video033_lgt/video033_0/video033_0_coords.json",
    "video035_0": "test_video035_lgt/video035_0/video035_0_coords.json",
}


def largest_polygon(geom):
    """Return the largest polygon from a Polygon or MultiPolygon."""
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return None


def clip_rooms(rooms):
    """
    rooms: list of dicts with 'room_id' and 'corners_xz' (list of [x,z])
    Returns a new list with non-overlapping polygons.
    """
    # Build Shapely polygons and sort by area descending
    polys = []
    for r in rooms:
        pts = [(c[0], c[1]) for c in r["corners_xz"]]
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)   # fix self-intersections
        polys.append((r["room_id"], poly))

    polys.sort(key=lambda x: x[1].area, reverse=True)

    claimed      = Polygon()   # union of all rooms processed so far
    clipped_rooms = []

    for room_id, poly in polys:
        overlap = poly.intersection(claimed)

        if overlap.is_empty or overlap.area < 1e-6:
            # No overlap at all — keep as-is
            clipped = poly
        else:
            clipped = poly.difference(claimed)
            if clipped.is_empty:
                print(f"  Room {room_id}: fully consumed by larger rooms — dropped")
                claimed = claimed.union(poly)
                continue
            clipped = largest_polygon(clipped)
            if clipped is None or clipped.area < MIN_AREA_M2:
                print(f"  Room {room_id}: shrank below {MIN_AREA_M2} m² after clipping — dropped")
                claimed = claimed.union(poly)
                continue
            removed = poly.area - clipped.area
            print(f"  Room {room_id}: clipped {removed:.3f} m² of overlap  "
                  f"({poly.area:.3f} → {clipped.area:.3f} m²)")

        claimed = claimed.union(poly)   # original polygon stays authoritative

        coords = [[round(x, 6), round(z, 6)] for x, z in clipped.exterior.coords[:-1]]
        clipped_rooms.append({"room_id": room_id, "corners_xz": coords})

    # Re-number room_ids 0..N-1 in original order
    id_order = {r["room_id"]: i for i, r in enumerate(
        sorted(clipped_rooms, key=lambda r: r["room_id"]))}
    for r in clipped_rooms:
        r["room_id"] = id_order[r["room_id"]]

    return clipped_rooms


def main():
    base = os.path.join(REPO_ROOT, "direct_360_FPE")

    for scene_ver, rel_path in COORD_FILES.items():
        json_path = os.path.join(base, rel_path)
        out_dir   = os.path.dirname(json_path)
        scene     = scene_ver   # e.g. "video035_0"

        print(f"\n{'='*60}")
        print(f"Clipping overlaps for {scene}")
        print(f"{'='*60}")

        with open(json_path) as f:
            data = json.load(f)

        original_rooms = data["final_fp"]["rooms"]
        print(f"  Rooms before clipping: {len(original_rooms)}")

        clipped_rooms = clip_rooms(original_rooms)
        print(f"  Rooms after clipping:  {len(clipped_rooms)}")

        # Write clipped data to a separate JSON — never touch the original
        data_clipped = copy.deepcopy(data)
        data_clipped["final_fp"]["rooms"]     = clipped_rooms
        data_clipped["final_fp"]["num_rooms"] = len(clipped_rooms)

        clipped_json = os.path.join(out_dir, f"{scene}_coords_clipped.json")
        with open(clipped_json, "w") as f:
            json.dump(data_clipped, f, indent=2)
        print(f"  Written {clipped_json}")

        # Save clipped floor plan as a separate file
        plot_final_fp(data_clipped,
                      Path(out_dir) / f"{scene}_map_final_fp_clipped.png")


if __name__ == "__main__":
    main()
