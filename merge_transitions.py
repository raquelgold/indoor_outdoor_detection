#!/usr/bin/env python3
"""
Merge transition events from multiple sources using proximity-based consensus.

Sources per video folder:
  transition_yolo.csv   columns: frame_id, direction, method
                        methods: objects_detection, building_area
  transition_sfuda.csv  columns: transition_frame, type, ...

A transition is confirmed when >= MIN_SOURCES distinct sources detect
the same direction within RADIUS frames of each other (chain clustering).

Output columns: direction, frame_min, frame_max, sources, n_sources

compute_indoor_segments() turns the confirmed transitions into indoor frame
ranges (the final per-frame indoor/outdoor decision).

Usage:
    python merge_transitions.py --yolo_csv data/videos/097/transition_yolo.csv \\
        --sfuda_csv data/videos/097/transition_sfuda.csv \\
        --output_csv data/videos/097/transitions_confirmed.csv
"""

import argparse
import pandas as pd
from pathlib import Path

RADIUS = 100
MIN_SOURCES = 2


def _normalize_direction(raw: str) -> str:
    return "IN_TO_OUT" if "IN_TO_OUT" in raw.upper() else "OUT_TO_IN"


def _load_yolo(csv_path: Path) -> list:
    df = pd.read_csv(csv_path)
    return [
        {"frame": int(row.frame_id),
         "direction": _normalize_direction(row.direction),
         "source": row.method}
        for row in df.itertuples(index=False)
    ]


def _load_sfuda(csv_path: Path) -> list:
    df = pd.read_csv(csv_path)
    return [
        {"frame": int(row.transition_frame),
         "direction": _normalize_direction(row.type),
         "source": "sfuda"}
        for row in df.itertuples(index=False)
    ]


def _cluster(events: list, radius: int, min_sources: int) -> list:
    """
    Chain-cluster events sorted by frame.
    A new event joins the current cluster if it is within `radius` frames
    of the last event added; otherwise it starts a new cluster.
    Clusters with >= min_sources distinct sources are confirmed.
    """
    if not events:
        return []

    events = sorted(events, key=lambda e: e["frame"])
    clusters = [[events[0]]]
    for ev in events[1:]:
        if ev["frame"] - clusters[-1][-1]["frame"] <= radius:
            clusters[-1].append(ev)
        else:
            clusters.append([ev])

    confirmed = []
    for cluster in clusters:
        sources = {e["source"] for e in cluster}
        if len(sources) >= min_sources:
            confirmed.append({
                "direction": cluster[0]["direction"],
                "frame_min": min(e["frame"] for e in cluster),
                "frame_max": max(e["frame"] for e in cluster),
                "sources": ",".join(sorted(sources)),
                "n_sources": len(sources),
            })
    return confirmed


def _overlaps(a: dict, b: dict) -> bool:
    return a["frame_min"] <= b["frame_max"] and b["frame_min"] <= a["frame_max"]


def _has_sfuda(cluster: dict) -> bool:
    return "sfuda" in cluster["sources"].split(",")


def _pick_winner(a: dict, b: dict) -> tuple:
    """Return (winner, loser) between two overlapping opposite-direction clusters."""
    a_sfuda, b_sfuda = _has_sfuda(a), _has_sfuda(b)
    if a_sfuda != b_sfuda:
        return (a, b) if a_sfuda else (b, a)
    if a["n_sources"] != b["n_sources"]:
        return (a, b) if a["n_sources"] > b["n_sources"] else (b, a)
    a_span = a["frame_max"] - a["frame_min"]
    b_span = b["frame_max"] - b["frame_min"]
    return (a, b) if a_span <= b_span else (b, a)


def _resolve_overlaps(confirmed: list) -> list:
    """
    Two opposite-direction clusters with overlapping frame ranges describe
    the same ambiguous crossing, not two real transitions. Keep only one:
    prefer whichever side sfuda voted for; if neither/both, prefer more
    sources; tie-break on the tighter (more confident) frame span.
    """
    items = sorted(confirmed, key=lambda e: e["frame_min"])
    keep = [True] * len(items)
    for i in range(len(items)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(items)):
            if not keep[j] or items[i]["direction"] == items[j]["direction"]:
                continue
            if not _overlaps(items[i], items[j]):
                continue
            winner, _ = _pick_winner(items[i], items[j])
            if winner is items[i]:
                keep[j] = False
            else:
                keep[i] = False
                break
    return [it for it, k in zip(items, keep) if k]


def merge_transitions(
    yolo_csv: Path,
    sfuda_csv: Path,
    output_csv: Path = None,
    radius: int = RADIUS,
    min_sources: int = MIN_SOURCES,
) -> pd.DataFrame:
    events = []

    if yolo_csv.exists():
        events.extend(_load_yolo(yolo_csv))
    else:
        print(f"  Warning: {yolo_csv.name} not found")

    if sfuda_csv.exists():
        events.extend(_load_sfuda(sfuda_csv))
    else:
        print(f"  Warning: {sfuda_csv.name} not found")

    confirmed = []
    for direction in ("OUT_TO_IN", "IN_TO_OUT"):
        dir_events = [e for e in events if e["direction"] == direction]
        confirmed.extend(_cluster(dir_events, radius, min_sources))

    confirmed = _resolve_overlaps(confirmed)

    result = (
        pd.DataFrame(confirmed, columns=["direction", "frame_min", "frame_max", "sources", "n_sources"])
        .sort_values("frame_min")
        .reset_index(drop=True)
    )

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        print(f"  Saved {len(result)} confirmed transition(s) → {output_csv}")

    return result


def compute_indoor_segments(transitions_df: pd.DataFrame, total_frames: int) -> list:
    """
    Indoor frame ranges [(start, end), ...] from confirmed transitions, with
    conservative bounds: OUT_TO_IN starts at frame_max, IN_TO_OUT ends at
    frame_min - 1, and a first IN_TO_OUT means the video starts indoors.
    Same rule as the DFPE repo's extract_indoor_clips.py (keep the two in sync).
    """
    df = transitions_df.sort_values("frame_min").reset_index(drop=True)
    if df.empty:
        return []

    # Infer initial state from direction of first transition
    indoor_start = 0 if df.iloc[0]["direction"] == "IN_TO_OUT" else None

    segments = []
    for row in df.itertuples(index=False):
        if row.direction == "OUT_TO_IN":
            indoor_start = row.frame_max
        elif row.direction == "IN_TO_OUT":
            if indoor_start is not None:
                segments.append((indoor_start, row.frame_min - 1))
                indoor_start = None

    if indoor_start is not None:
        segments.append((indoor_start, total_frames - 1))

    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge transition sources with proximity consensus")
    parser.add_argument("--yolo_csv",   type=Path, required=True)
    parser.add_argument("--sfuda_csv",  type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--radius",      type=int, default=RADIUS,
                        help="Max frame gap for two events to be in the same cluster")
    parser.add_argument("--min_sources", type=int, default=MIN_SOURCES,
                        help="Min distinct sources required to confirm a transition")
    args = parser.parse_args()

    df = merge_transitions(args.yolo_csv, args.sfuda_csv, args.output_csv,
                           args.radius, args.min_sources)
    print(df.to_string(index=False))
