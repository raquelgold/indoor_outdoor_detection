#!/usr/bin/env python3
"""
Create a side-by-side video:
  LEFT   : original video (downscaled), with INDOOR/OUTDOOR banner
  RIGHT  : VSLAM GIF animation (frame-synced) while indoor,
           final DFPE floor plan map while outdoor

Usage:
    source ~/venv/bin/activate
    python dpfe_pipeline/visualize_video_map_animated.py \\
        --video      /path/to/VID_..._085.mp4 \\
        --video_dir  data/videos/085 \\
        --output     data/videos/085/indoor_map_viz_animated.mp4

    # Custom results root if not the default
    python dpfe_pipeline/visualize_video_map_animated.py \\
        --video ... --video_dir ... \\
        --results_root outputs/dfpe_results
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

PANEL_H = 480
FONT = cv2.FONT_HERSHEY_SIMPLEX
from dpfe_paths import RESULTS_DEFAULT


def load_gif_frames(gif_path: Path, w: int, h: int) -> list[np.ndarray]:
    """Load all GIF frames as BGR numpy arrays, letterboxed to (w, h)."""
    gif = Image.open(str(gif_path))
    frames = []
    try:
        while True:
            frame_rgba = gif.convert("RGBA")
            frame_rgb = np.array(frame_rgba)[:, :, :3]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            ih, iw = frame_bgr.shape[:2]
            scale = min(w / iw, h / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            y0 = (h - nh) // 2
            x0 = (w - nw) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = resized
            frames.append(canvas)

            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames


def load_image(png_path: Path, w: int, h: int) -> np.ndarray:
    img = cv2.imread(str(png_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {png_path}")
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def add_banner(frame: np.ndarray, text: str, color: tuple) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    bh = max(28, h // 18)
    cv2.rectangle(out, (0, 0), (w, bh), (0, 0, 0), -1)
    scale = bh / 32
    thick = max(1, int(scale * 1.5))
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thick)
    x = (w - tw) // 2
    y = int(bh * 0.75)
    cv2.putText(out, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return out


def build_video(video_path: Path, video_dir: Path, results_root: Path, output: Path):
    seg_csv = video_dir / "clips" / "indoor_segments.csv"
    if not seg_csv.exists():
        raise FileNotFoundError(f"indoor_segments.csv not found: {seg_csv}")
    segs = pd.read_csv(seg_csv)

    vid_id = video_dir.name  # e.g. "085"

    # Build list of (start, end, clip_idx) per segment
    segments = []
    for row in segs.itertuples(index=False):
        clip_idx = int(row.clip.split("_")[1])
        segments.append((int(row.start_frame), int(row.end_frame), clip_idx))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_w = int(PANEL_H * src_w / src_h)
    map_w = video_w
    out_w = video_w + map_w
    out_h = PANEL_H

    # Load GIF frames and final map per segment
    print("Loading assets...")
    seg_assets = []
    for start, end, clip_idx in segments:
        gif_path = results_root / f"{vid_id}_{clip_idx}" / f"{vid_id}_{clip_idx}_map_animation.gif"
        map_path = results_root / f"{vid_id}_{clip_idx}" / f"{vid_id}_{clip_idx}_map_final_fp.png"
        gif_frames = load_gif_frames(gif_path, map_w, out_h)
        final_map = load_image(map_path, map_w, out_h)
        seg_assets.append((start, end, gif_frames, final_map))
        print(f"  Segment frames {start}–{end}: {len(gif_frames)} GIF frames loaded")

    # Build a lookup: which segment (if any) does this video frame belong to?
    def get_right_panel(frame_idx: int) -> np.ndarray:
        for start, end, gif_frames, final_map in seg_assets:
            if start <= frame_idx <= end:
                # Map indoor progress → GIF frame index
                progress = (frame_idx - start) / max(end - start, 1)
                gif_idx = int(progress * len(gif_frames))
                if gif_idx >= len(gif_frames) - 1:
                    # Wireframe animation has finished building — show the
                    # actual finished DFPE map for the rest of the indoor
                    # segment (also covers videos that end while still indoors).
                    return final_map, True
                return gif_frames[gif_idx], True
        # Outdoor after the last indoor segment: show the final map
        if seg_assets and frame_idx > seg_assets[-1][1]:
            _, _, _, final_map = seg_assets[-1]
            return final_map, False
        # Outdoor before any indoor segment: blank panel
        blank = np.zeros((out_h, map_w, 3), dtype=np.uint8)
        return blank, False

    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (out_w, out_h))

    print(f"Writing {total} frames → {output}  ({out_w}×{out_h} @ {fps:.0f} fps)")

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        left = cv2.resize(frame, (video_w, out_h), interpolation=cv2.INTER_AREA)

        right, is_indoor = get_right_panel(frame_idx)
        if is_indoor:
            left = add_banner(left, "INDOOR", (100, 230, 100))
        else:
            left = add_banner(left, "OUTDOOR", (100, 100, 230))

        combined = np.hstack([left, right])
        writer.write(combined)

        if frame_idx % 300 == 0:
            pct = 100 * frame_idx / max(total, 1)
            print(f"  {frame_idx}/{total}  ({pct:.0f}%)")

    cap.release()
    writer.release()
    print(f"Done → {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side video: VSLAM GIF animation (indoor) + final DFPE map (outdoor)")
    parser.add_argument("--video", type=Path, required=True,
                        help="Original video file (.mp4)")
    parser.add_argument("--video_dir", type=Path, required=True,
                        help="Pipeline dir for this video (has clips/indoor_segments.csv)")
    parser.add_argument("--results_root", type=Path, default=RESULTS_DEFAULT,
                        help="DFPE results root (default: outputs/dfpe_results)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .mp4 path (default: {video_dir}/indoor_map_viz_animated.mp4)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.video_dir / "indoor_map_viz_animated.mp4"

    build_video(args.video, args.video_dir, args.results_root, args.output)


if __name__ == "__main__":
    main()
