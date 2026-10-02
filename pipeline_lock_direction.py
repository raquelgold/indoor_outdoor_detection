#!/usr/bin/env python3
"""
YOLO-OIV7 indoor/outdoor transition detection for a folder of 360° videos
(or a single video with --video).

Usage:
    ~/venv/bin/python pipeline_lock_direction.py --video X.mp4       # one video
    ~/venv/bin/python pipeline_lock_direction.py                     # every video in --videos_dir
    ~/venv/bin/python pipeline_lock_direction.py --output_dir DIR    # custom output root

For each video:
  1. Process frames in memory, run OIV7 inference
  2. Save YOLO label .txt files (input for extract_classes_oiv7)
  3. Save oiv7_detections.csv (input for find_indoor_area / find_indoor_objects)
  4. Run extract_classes_oiv7 → per-frame class summaries
  5. Run find_indoor_area   → indoor_outdoor_classification.csv
  6. Run find_indoor_objects → indoor_outdoor_objects_classification.csv
  7. Run detect_transitions  → transitions.csv (copy to data/videos/{id}/transition_yolo.csv)

Output structure:
  pipeline_output/
    {video_stem}/
      labels/                          <- YOLO .txt label files
        {video_stem}_{frame:06d}.txt
        {video_stem}/                  <- extract_classes_oiv7 summary dir
          per_frame_summary_*.txt
      oiv7_detections.csv
      indoor_outdoor_classification.csv
      indoor_outdoor_objects_classification.csv
"""

import argparse
import importlib.util
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────────────────────
VIDEOS_DIR = Path.home() / "lock_direction" / "new_lock_direction"
OUTPUT_DIR = Path.home() / "pipeline_output"
MODEL_PATH = Path.home() / "yolov8l-oiv7.pt"

HERE = Path(__file__).resolve().parent
EXTRACT_SCRIPT = HERE / "extract_classes_yolo" / "extract_classes_oiv7.py"
FIND_AREA_SCRIPT = HERE / "object_detection" / "find_indoor_area.py"
FIND_OBJECTS_SCRIPT = HERE / "object_detection" / "find_indoor_objects.py"
DETECT_TRANSITIONS_SCRIPT = HERE / "object_detection" / "detect_transitions.py"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".webm"}

YOLO_CONF = 0.01   # low threshold; classifiers do their own filtering
YOLO_IOU = 0.7
SKIP_EXISTING = True  # skip video if oiv7_detections.csv already exists
SAVE_LABELS = False   # set True to write per-frame YOLO .txt files (needed only by extract_classes_oiv7)
# ─────────────────────────────────────────────────────────────────────────────


def _load_module(script_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_inference(video_path: Path, video_out: Path, model: YOLO):
    """
    Stream video frames, run YOLO on each frame.
    Writes:
      - video_out/labels/{stem}_{frame:06d}.txt  (YOLO format with conf)
      - video_out/oiv7_detections.csv
    Returns path to detections CSV and path to labels dir.
    """
    labels_dir = video_out / "labels"
    if SAVE_LABELS:
        labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = float(w * h) or 1.0

    stem = video_path.stem
    rows = []
    frame_idx = 0

    with tqdm(total=total, desc=f"  {video_path.name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False
            )
            result = results[0]

            # Write YOLO label file (cls x_c y_c w h conf)
            if SAVE_LABELS:
                label_file = labels_dir / f"{stem}_{frame_idx:06d}.txt"
                with open(label_file, "w") as lf:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            xc, yc, bw, bh = box.xywhn[0].tolist()
                            lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf:.6f}\n")

            # Collect detections for CSV
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    area_pct = max(0.0, x2 - x1) * max(0.0, y2 - y1) / frame_area * 100.0
                    rows.append({
                        "frame_id": f"frame_{frame_idx:06d}",
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "conf": round(conf, 6),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "area_pct": round(area_pct, 6),
                    })

            frame_idx += 1
            pbar.update(1)

    cap.release()

    detections_csv = video_out / "oiv7_detections.csv"
    pd.DataFrame(rows).to_csv(detections_csv, index=False)

    print(f"    {frame_idx} frames, {len(rows)} detections → {detections_csv.name}")
    return detections_csv, labels_dir


def run_extract_classes(labels_dir: Path):
    """Run extract_classes_oiv7 on the per-video labels directory."""
    print("  Running extract_classes_oiv7...")
    mod = _load_module(EXTRACT_SCRIPT, "extract_classes_oiv7")
    mod.LABELS_DIR = str(labels_dir)
    mod.main()


def run_find_indoor_area(detections_csv: Path, video_out: Path):
    print("  Running find_indoor_area...")
    mod = _load_module(FIND_AREA_SCRIPT, "find_indoor_area")
    out_csv = video_out / "indoor_outdoor_classification.csv"
    mod.classify_frames(detections_csv, out_csv)


def run_find_indoor_objects(detections_csv: Path, video_out: Path):
    print("  Running find_indoor_objects...")
    mod = _load_module(FIND_OBJECTS_SCRIPT, "find_indoor_objects")
    out_csv = video_out / "indoor_outdoor_objects_classification.csv"
    mod.classify_frames_by_objects(detections_csv, out_csv)


def run_detect_transitions(video_out: Path):
    print("  Running detect_transitions...")
    mod = _load_module(DETECT_TRANSITIONS_SCRIPT, "detect_transitions")
    objects_csv = video_out / "indoor_outdoor_objects_classification.csv"
    area_csv = video_out / "indoor_outdoor_classification.csv"
    out_csv = video_out / "transitions.csv"
    mod.detect_transitions(objects_csv, area_csv, out_csv)


def find_videos(directory: Path):
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(directory.glob(f"*{ext}"))
        videos.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(set(videos))


def main():
    parser = argparse.ArgumentParser(description="YOLO-OIV7 indoor/outdoor transition detection")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--video", type=Path, help="Process a single video file")
    src.add_argument("--videos_dir", type=Path, default=VIDEOS_DIR,
                     help=f"Process every video in this folder (default: {VIDEOS_DIR})")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR,
                        help=f"Output root; one {{video_stem}}/ folder per video (default: {OUTPUT_DIR})")
    parser.add_argument("--model", type=Path, default=MODEL_PATH,
                        help=f"YOLO OIV7 weights (default: {MODEL_PATH})")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        if not args.video.is_file():
            raise FileNotFoundError(f"Video not found: {args.video}")
        videos = [args.video]
    else:
        videos = find_videos(args.videos_dir)
        if not videos:
            print(f"No videos found in {args.videos_dir}")
            return
        print(f"Found {len(videos)} video(s) in {args.videos_dir}")

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    print(f"Loading OIV7 model from {args.model}...")
    model = YOLO(str(args.model))
    print()

    ok, failed = [], []

    for video_path in videos:
        print(f"{'='*60}")
        print(f"Video: {video_path.name}")
        video_out = output_dir / video_path.stem
        video_out.mkdir(parents=True, exist_ok=True)

        detections_csv = video_out / "oiv7_detections.csv"
        labels_dir = video_out / "labels"

        try:
            # Step 1–3: inference → label files + detections CSV
            if SKIP_EXISTING and detections_csv.exists():
                print(f"  Skipping inference (detections CSV exists)")
            else:
                detections_csv, labels_dir = run_inference(video_path, video_out, model)

            # Step 4: extract_classes_oiv7 class summaries (skipped when SAVE_LABELS=False)
            if SAVE_LABELS:
                run_extract_classes(labels_dir)

            # Step 5: indoor area classifier
            run_find_indoor_area(detections_csv, video_out)

            # Step 6: indoor objects classifier
            run_find_indoor_objects(detections_csv, video_out)

            # Step 7: detect indoor/outdoor transitions
            run_detect_transitions(video_out)

            print(f"  ✓ Done: {video_path.name}")
            ok.append(video_path.name)

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append(video_path.name)

        print()

    print(f"{'='*60}")
    print(f"Pipeline complete: {len(ok)} ok, {len(failed)} failed")
    if failed:
        print("Failed:")
        for name in failed:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
