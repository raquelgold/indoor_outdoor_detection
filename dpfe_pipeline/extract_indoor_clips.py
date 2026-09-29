#!/usr/bin/env python3
"""
Stage 2: Extract indoor frame clips from a video based on confirmed transitions.

Reads transitions_confirmed.csv, derives indoor frame ranges, and writes
frames as JPEGs into:
  <output_dir>/clip_<N>/rgb/frame_{frame_id:06d}.jpg

Also writes indoor_segments.csv:
  clip, start_frame, end_frame

Conservative segment boundaries:
  OUT_TO_IN transition → indoor starts at frame_max (definitely indoor by then)
  IN_TO_OUT transition → indoor ends at frame_min - 1 (still indoor before first detection)
"""

import argparse
import cv2
import pandas as pd
from pathlib import Path


def compute_indoor_segments(transitions_df: pd.DataFrame, total_frames: int) -> list:
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


def extract_clip(video_path: Path, start: int, end: int, out_dir: Path) -> int:
    rgb_dir = out_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    written = 0
    for frame_id in range(start, end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(rgb_dir / f"frame_{frame_id:06d}.jpg"), frame)
        written += 1

    cap.release()
    return written


def extract_indoor_clips(video_path: Path, transitions_csv: Path, output_dir: Path):
    df = pd.read_csv(transitions_csv)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"  Video: {video_path.name} — {total_frames} total frames")

    segments = compute_indoor_segments(df, total_frames)
    if not segments:
        print("  No indoor segments found.")
        return []

    print(f"  Indoor segments: {len(segments)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (start, end) in enumerate(segments):
        clip_dir = output_dir / f"clip_{i:03d}"
        n = extract_clip(video_path, start, end, clip_dir)
        print(f"    clip_{i:03d}: frames {start}–{end}  ({n} frames written) → {clip_dir}/rgb/")

    manifest = pd.DataFrame(
        [{"clip": f"clip_{i:03d}", "start_frame": s, "end_frame": e}
         for i, (s, e) in enumerate(segments)]
    )
    manifest_path = output_dir / "indoor_segments.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"  Saved segment manifest → {manifest_path}")

    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract indoor frame clips from confirmed transitions")
    parser.add_argument("--video",           type=Path, required=True, help="Input video file")
    parser.add_argument("--transitions_csv", type=Path, required=True, help="transitions_confirmed.csv")
    parser.add_argument("--output_dir",      type=Path, required=True, help="Where to write clip_NNN/ folders")
    args = parser.parse_args()

    extract_indoor_clips(args.video, args.transitions_csv, args.output_dir)
