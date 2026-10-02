#!/usr/bin/env python3
"""
Render a video showing every indoor/outdoor classifier's per-frame decision.

Top banner: one big INDOOR/OUTDOOR block per method, then "= FINAL":
  objects_detection  – find_indoor_objects.py (indoor-like vs outdoor-like object counts)
  building_area      – find_indoor_area.py    (House/Building box > AREA_PCT_THRESHOLD)
  SegFormer (SFUDA)  – 360SFUDA/suspicious_frames.py (sky/veg/car %, 3 s smoothing)
  FINAL (pipeline)   – the final decision: merge_transitions.py confirms transitions
                       that >= 2 sources agree on, then compute_indoor_segments turns
                       them into indoor frame ranges. Always uses all 3 sources,
                       including with --only others.

On the frame, the YOLO boxes those methods actually use:
  green  = INDOOR_LIKE  (conf >= MIN_CONFIDENCE)
  orange = OUTDOOR_LIKE (conf >= MIN_CONFIDENCE)
  blue   = House/Building (conf >= CONFIDENCE_THRESHOLD); thick when area >= threshold

Inputs are the outputs of pipeline_lock_direction.py (--detections_dir) and of
360SFUDA/pipeline_master.py (--sfuda_csv = its frame_object_stats.csv).

--only building : just building_area + House/Building boxes (no FINAL block;
                  --sfuda_csv not needed)
--only others   : objects_detection + SegFormer + indoor/outdoor-like boxes + FINAL

Usage:
    ~/venv/bin/python visualize_classifications.py \\
        --video          ~/dfpe_data/VID_20251226_110816_00_010.mp4 \\
        --detections_dir ~/pipeline_output/VID_20251226_110816_00_010 \\
        --sfuda_csv      /mnt/sfuda_out/010/frame_object_stats.csv \\
        --output         /mnt/sfuda_out/010/classifications_010.mp4
"""

import argparse
import importlib.util
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
SFUDA_DIR = Path(os.environ.get("SFUDA_DIR", Path.home() / "360SFUDA"))

# BGR colours
INDOOR_COLOR = (80, 175, 76)
OUTDOOR_COLOR = (40, 120, 230)
BUILDING_COLOR = (230, 150, 40)
PANEL_BG = (30, 30, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _load_module(script_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _frame_number(frame_id):
    return int(str(frame_id).split("_")[-1])


def load_yolo_labels(detections_dir: Path, name: str) -> dict:
    df = pd.read_csv(detections_dir / name)
    df["frame"] = df["frame_id"].apply(_frame_number)
    return df.set_index("frame").to_dict("index")


def load_sfuda_labels(stats_csv: Path, fps: float) -> dict:
    """Per-frame smoothed SFUDA state, computed exactly as suspicious_frames.analyze_csv does."""
    sf = _load_module(SFUDA_DIR / "suspicious_frames.py", "suspicious_frames")
    frames = sf.get_env_stats_per_frame(pd.read_csv(stats_csv))

    # Phase 1 + 2 of analyze_csv (raw status, then 3 s stable-island smoothing)
    first = frames.iloc[0]
    is_outdoor = (first.sky_area > 15.0 or first.veg_area > 15.0 or first.car_area > 5.0)
    prev_sky = first.sky_area
    raw = []
    for fid in frames.index:
        d = frames.loc[fid]
        is_outdoor = sf.determine_scene_type(d.sky_area, d.veg_area, d.car_area, prev_sky, is_outdoor)
        raw.append(is_outdoor)
        prev_sky = d.sky_area
    smoothed = sf.stable_island_smoothing(np.array(raw), fps, min_duration_sec=3.0)

    return {
        int(fid): {"outdoor": bool(s), "raw_outdoor": bool(r),
                   "sky": d.sky_area, "veg": d.veg_area, "car": d.car_area}
        for (fid, d), s, r in zip(frames.iterrows(), smoothed, raw)
    }


def load_final_labels(yolo_transitions: Path, sfuda_report: Path, n_frames: int) -> np.ndarray:
    """Per-frame final indoor flag: merged transitions -> indoor frame ranges."""
    merge = _load_module(HERE / "merge_transitions.py", "merge_transitions")
    confirmed = merge.merge_transitions(yolo_transitions, sfuda_report)
    segments = merge.compute_indoor_segments(confirmed, n_frames)
    print(f"  Pipeline indoor segments (FINAL): {segments or 'none'}")
    indoor = np.zeros(n_frames, dtype=bool)
    for start, end in segments:
        indoor[start:end + 1] = True
    return indoor


def draw_boxes(img, dets, scale, objects_mod, area_mod, show_building=True, show_objects=True):
    for det in dets:
        name, conf = det["class_name"], det["conf"]
        if name in area_mod.HOUSE_BUILDING_CLASSES:
            if not show_building or conf < area_mod.CONFIDENCE_THRESHOLD:
                continue
            color = BUILDING_COLOR
            thick = 5 if det["area_pct"] > area_mod.AREA_PCT_THRESHOLD else 2
            label = f"{name} {conf:.2f} ({det['area_pct']:.2f}%)"
        elif not show_objects:
            continue
        elif name in objects_mod.INDOOR_LIKE and conf >= objects_mod.MIN_CONFIDENCE:
            color, thick, label = INDOOR_COLOR, 2, f"{name} {conf:.2f}"
        elif name in objects_mod.OUTDOOR_LIKE and conf >= objects_mod.MIN_CONFIDENCE:
            color, thick, label = OUTDOOR_COLOR, 2, f"{name} {conf:.2f}"
        else:
            continue
        x1, y1, x2, y2 = (int(det[k] * scale) for k in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        ty = max(y1, th + 6)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(img, label, (x1 + 3, ty - 4), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _label_style(label):
    if label is None:
        return "NO DATA", (110, 110, 110)
    return label.upper(), INDOOR_COLOR if label == "indoor" else OUTDOOR_COLOR


def _centered_text(img, text, cx, y, scale, thick, color=(255, 255, 255)):
    tw = cv2.getTextSize(text, FONT, scale, thick)[0][0]
    cv2.putText(img, text, (cx - tw // 2, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_panel(width, frame_idx, n_frames, fps, rows, legend, final=None, show_final=False):
    """
    rows  : [(method, label)], label = "indoor" / "outdoor" / None
    final : the pipeline's final label for this frame (drawn after "=" when show_final)
    """
    header_h, banner_h, margin, gap, eq_w = 44, 140, 15, 15, 80
    panel = np.full((header_h + banner_h, width, 3), PANEL_BG, np.uint8)

    header = f"Frame {frame_idx} / {n_frames - 1}    t = {frame_idx / fps:5.1f} s"
    cv2.putText(panel, header, (margin, 32), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    x = width - 20 - sum(60 + cv2.getTextSize(t, FONT, 0.6, 1)[0][0] for t, _ in legend)
    for text, color in legend:
        cv2.rectangle(panel, (x, 14), (x + 22, 36), color, -1)
        cv2.putText(panel, text, (x + 30, 32), FONT, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        x += 60 + cv2.getTextSize(text, FONT, 0.6, 1)[0][0]

    # Banner: one big block per method, then "= FINAL"
    n_blocks = len(rows) + (1 if show_final else 0)
    avail = width - 2 * margin - gap * (len(rows) - 1) - (eq_w if show_final else 0)
    block_w = avail // n_blocks
    big = 2.8 if n_blocks == 1 else 2.0
    y0, y1 = header_h, header_h + banner_h - 12
    x = margin
    for method, label in rows:
        text, color = _label_style(label)
        cv2.rectangle(panel, (x, y0), (x + block_w, y1), color, -1)
        _centered_text(panel, method, x + block_w // 2, y0 + 32, 0.8, 2)
        _centered_text(panel, text, x + block_w // 2, y1 - 20, big, 5)
        x += block_w + gap
    if show_final:
        x -= gap
        _centered_text(panel, "=", x + eq_w // 2, (y0 + y1) // 2 + 22, 2.4, 6)
        x += eq_w
        text, color = _label_style(final)
        cv2.rectangle(panel, (x, y0), (x + block_w, y1), color, -1)
        cv2.rectangle(panel, (x + 2, y0 + 2), (x + block_w - 2, y1 - 2), (255, 255, 255), 5)
        _centered_text(panel, "FINAL (pipeline)", x + block_w // 2, y0 + 32, 0.8, 2)
        _centered_text(panel, text, x + block_w // 2, y1 - 20, big, 5)

    return panel


def main():
    parser = argparse.ArgumentParser(description="Video showing all indoor/outdoor classifications per frame")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detections_dir", type=Path, required=True,
                        help="pipeline_lock_direction.py output dir for this video")
    parser.add_argument("--sfuda_csv", type=Path,
                        help="360SFUDA frame_object_stats.csv for this video (not needed with --only building)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920, help="Output frame width (default 1920)")
    parser.add_argument("--yolo_transitions", type=Path,
                        help="YOLO transitions.csv for the FINAL block (default: {detections_dir}/transitions.csv)")
    parser.add_argument("--sfuda_report", type=Path,
                        help="SFUDA transition_report.csv for the FINAL block (default: next to --sfuda_csv)")
    parser.add_argument("--only", choices=["building", "others"],
                        help="Show only building_area, or only the other methods (default: all)")
    args = parser.parse_args()
    show_building = args.only != "others"
    show_others = args.only != "building"
    if show_others and args.sfuda_csv is None:
        parser.error("--sfuda_csv is required unless --only building")

    objects_mod = _load_module(HERE / "object_detection" / "find_indoor_objects.py", "find_indoor_objects")
    area_mod = _load_module(HERE / "object_detection" / "find_indoor_area.py", "find_indoor_area")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = args.width / src_w
    out_w, vid_h = args.width, int(round(src_h * scale)) // 2 * 2

    print("Loading classifications...")
    objects = load_yolo_labels(args.detections_dir, "indoor_outdoor_objects_classification.csv")
    area = load_yolo_labels(args.detections_dir, "indoor_outdoor_classification.csv")
    sfuda = load_sfuda_labels(args.sfuda_csv, fps) if show_others else {}
    final = load_final_labels(args.yolo_transitions or args.detections_dir / "transitions.csv",
                              args.sfuda_report or args.sfuda_csv.parent / "transition_report.csv",
                              n_frames) if show_others else None
    dets = pd.read_csv(args.detections_dir / "oiv7_detections.csv")
    dets["frame"] = dets["frame_id"].apply(_frame_number)
    dets_by_frame = {f: g.to_dict("records") for f, g in dets.groupby("frame")}

    legend = ([("indoor-like", INDOOR_COLOR), ("outdoor-like", OUTDOOR_COLOR)] if show_others else []) \
        + ([("House/Building", BUILDING_COLOR)] if show_building else [])
    n_rows = (1 if show_building else 0) + (2 if show_others else 0)
    panel_h = draw_panel(out_w, 0, n_frames, fps, [("", None)] * n_rows, legend).shape[0]
    out_h = (vid_h + panel_h) // 2 * 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(args.output)],
        stdin=subprocess.PIPE)

    for idx in tqdm(range(n_frames), desc="Rendering"):
        ok, frame = cap.read()
        if not ok:
            break
        img = cv2.resize(frame, (out_w, vid_h), interpolation=cv2.INTER_AREA)
        draw_boxes(img, dets_by_frame.get(idx, []), scale, objects_mod, area_mod, show_building, show_others)

        o, a, s = objects.get(idx), area.get(idx), sfuda.get(idx)
        rows = []
        if show_others:
            rows.append(("objects_detection", o and o["classification"]))
        if show_building:
            rows.append(("building_area", a and a["classification"]))
        if show_others:
            rows.append(("SegFormer (SFUDA)", s and ("outdoor" if s["outdoor"] else "indoor")))
        final_label = None if final is None else ("indoor" if final[idx] else "outdoor")
        panel = draw_panel(out_w, idx, n_frames, fps, rows, legend, final_label, show_final=show_others)
        canvas = np.vstack([panel, img])[:out_h]
        ffmpeg.stdin.write(canvas.tobytes())

    cap.release()
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg encoding failed")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
