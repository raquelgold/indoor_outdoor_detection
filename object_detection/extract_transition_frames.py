#!/usr/bin/env python3
"""
For every video in lock_direction/new_lock_direction, run the "objects_detection"
and "building_area" indoor/outdoor methods independently (same logic as
find_indoor_objects.py / find_indoor_area.py + detect_transitions.py), then save
each confirmed transition frame as a PNG with all YOLO detections for that frame
drawn on top.

Reuses pipeline_output/{stem}/ outputs (oiv7_detections.csv, the two
classification CSVs, transitions.csv) when they were computed from the exact
same video file (matched by size against the top-level lock_direction/ copy),
to avoid re-running YOLO inference and re-classifying videos already processed
by pipeline_lock_direction.py.

Output: /mnt/transition_frames/{stem}/
  oiv7_detections.csv                        (only written if freshly computed)
  indoor_outdoor_classification.csv          (building_area per-frame, only if freshly computed)
  indoor_outdoor_objects_classification.csv  (objects_detection per-frame, only if freshly computed)
  transitions.csv                            (only if freshly computed)
  <method>/frame_NNNNNN_<DIRECTION>.png                                   (the transition frame itself)
  <method>/context/frame_NNNNNN_<DIRECTION>/frame_MMMMMM_offset<+-K>.png  (CONTEXT_WINDOW frames before/after)
  <method>/between/frame_MMMMMM_between_NNNNNN_PPPPPP.png                 (midpoint frame between consecutive transitions)
plus a top-level transitions_summary.csv listing every saved PNG.

Context/between frames are computed independently per method (building_area
and objects_detection each get their own context/between frames from their
own transition list), and detections on them are filtered/drawn using that
same method's class set.
"""

import importlib.util
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

NEW_VIDEOS_DIR = Path("/home/geolocation/lock_direction/new_lock_direction")
EXISTING_LOCK_DIR = Path("/home/geolocation/lock_direction")
EXISTING_PIPELINE_OUTPUT = Path("/home/geolocation/pipeline_output")
OUTPUT_ROOT = Path("/mnt/transition_frames")
MODEL_PATH = Path("/home/geolocation/yolov8l-oiv7.pt")

HERE = Path(__file__).resolve().parent
FIND_AREA_SCRIPT = HERE / "find_indoor_area.py"
FIND_OBJECTS_SCRIPT = HERE / "find_indoor_objects.py"
DETECT_TRANSITIONS_SCRIPT = HERE / "detect_transitions.py"

YOLO_CONF = 0.01
YOLO_IOU = 0.7
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
CONTEXT_WINDOW = 5  # frames saved before and after each transition frame

def _load_module(script_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


find_indoor_area = _load_module(FIND_AREA_SCRIPT, "find_indoor_area")
find_indoor_objects = _load_module(FIND_OBJECTS_SCRIPT, "find_indoor_objects")
detect_transitions_mod = _load_module(DETECT_TRANSITIONS_SCRIPT, "detect_transitions")

# Each method's PNG only draws the boxes that actually drive that method's
# classification (same class sets/thresholds as the imported modules above).
BUILDING_AREA_DRAW_CLASSES = find_indoor_area.HOUSE_BUILDING_CLASSES
OBJECTS_DETECTION_DRAW_CLASSES = find_indoor_objects.INDOOR_LIKE | find_indoor_objects.OUTDOOR_LIKE
OBJECTS_DETECTION_MIN_CONF = find_indoor_objects.MIN_CONFIDENCE

_model = None  # lazily loaded only if a video needs fresh YOLO inference


def find_reusable_pipeline_dir(video_path: Path):
    """
    Return pipeline_output/{stem}/ if it was computed from the exact same video
    (matched by file size against the top-level lock_direction/{stem}.mp4 copy),
    else None. pipeline_lock_direction.py always writes all four output files
    together, so the presence of oiv7_detections.csv implies the rest exist too.
    """
    stem = video_path.stem
    existing_dir = EXISTING_PIPELINE_OUTPUT / stem
    existing_video = EXISTING_LOCK_DIR / f"{stem}.mp4"
    if not (existing_dir / "oiv7_detections.csv").exists() or not existing_video.exists():
        return None
    try:
        if existing_video.stat().st_size != video_path.stat().st_size:
            return None
    except OSError:
        return None
    return existing_dir


def run_inference(video_path: Path, out_csv: Path, model: YOLO):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = float(w * h) or 1.0

    rows = []
    frame_idx = 0
    with tqdm(total=total, desc=f"  YOLO {video_path.name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
            result = results[0]
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
                        "x1": round(x1, 2), "y1": round(y1, 2),
                        "x2": round(x2, 2), "y2": round(y2, 2),
                        "area_pct": round(area_pct, 6),
                    })
            frame_idx += 1
            pbar.update(1)
    cap.release()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def get_pipeline_outputs(video_path: Path, video_out: Path):
    """
    Return (detections_csv, transitions_csv) for this video, reusing a matching
    pipeline_output/{stem}/ directory wholesale when available, else computing
    detections (fresh or reused) and running the real find_indoor_area /
    find_indoor_objects / detect_transitions functions into video_out.
    """
    reused_dir = find_reusable_pipeline_dir(video_path)
    if reused_dir is not None:
        print(f"  Reusing full pipeline_output: {reused_dir}")
        return reused_dir / "oiv7_detections.csv", reused_dir / "transitions.csv"

    global _model
    local_detections_csv = video_out / "oiv7_detections.csv"
    if local_detections_csv.exists():
        detections_csv = local_detections_csv
        print("  Reusing detections computed in a previous run of this script")
    else:
        if _model is None:
            print(f"Loading OIV7 model from {MODEL_PATH}...")
            _model = YOLO(str(MODEL_PATH))
        print("  No matching existing pipeline_output, running fresh YOLO inference")
        detections_csv = run_inference(video_path, local_detections_csv, _model)

    area_csv = video_out / "indoor_outdoor_classification.csv"
    objects_csv = video_out / "indoor_outdoor_objects_classification.csv"
    find_indoor_area.classify_frames(detections_csv, area_csv)
    find_indoor_objects.classify_frames_by_objects(detections_csv, objects_csv)

    transitions_csv = video_out / "transitions.csv"
    detect_transitions_mod.detect_transitions(objects_csv, area_csv, transitions_csv)
    return detections_csv, transitions_csv


def draw_detections(frame, dets: pd.DataFrame):
    annotated = frame.copy()
    for _, d in dets.iterrows():
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        label = f"{d.class_name} {d.conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        label_y = max(y1 - 10, label_size[1] + 10)
        cv2.rectangle(
            annotated,
            (x1, label_y - label_size[1] - 6),
            (x1 + label_size[0] + 6, label_y + 6),
            color, -1,
        )
        cv2.putText(
            annotated, label, (x1 + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
    return annotated


def _filter_dets(det_df: pd.DataFrame, method: str, frame_id: int) -> pd.DataFrame:
    frame_key = f"frame_{frame_id:06d}"
    dets = det_df[det_df["frame_id"] == frame_key]
    if method == "building_area":
        dets = dets[dets["class_name"].isin(BUILDING_AREA_DRAW_CLASSES)]
    elif method == "objects_detection":
        dets = dets[
            dets["class_name"].isin(OBJECTS_DETECTION_DRAW_CLASSES)
            & (dets["conf"] >= OBJECTS_DETECTION_MIN_CONF)
        ]
    return dets


def _save_frame(cap, det_df: pd.DataFrame, method: str, frame_id: int, png_path: Path,
                 video_path: Path, summary_rows: list, frame_type: str,
                 direction: str = "", offset=None, reference_frame_id: str = ""):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if not ret:
        print(f"  WARNING: could not read frame {frame_id} from {video_path.name}")
        return

    dets = _filter_dets(det_df, method, frame_id)
    annotated = draw_detections(frame, dets)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png_path), annotated)
    print(f"    Saved {png_path.relative_to(OUTPUT_ROOT)} ({len(dets)} boxes)")

    summary_rows.append({
        "video": video_path.name,
        "method": method,
        "frame_id": frame_id,
        "direction": direction,
        "frame_type": frame_type,
        "offset": offset if offset is not None else "",
        "reference_frame_id": reference_frame_id,
        "n_detections_drawn": len(dets),
        "png_path": str(png_path),
    })


def extract_transition_pngs(video_path: Path, detections_csv: Path, transitions_csv: Path,
                             video_out: Path, summary_rows: list):
    det_df = pd.read_csv(detections_csv)
    if det_df.empty:
        print(f"  No detections for {video_path.name}, skipping transition extraction")
        return

    transitions_df = pd.read_csv(transitions_csv)
    if transitions_df.empty:
        print(f"  No transitions found for {video_path.name}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for method, group in transitions_df.groupby("method"):
        group = group.sort_values("frame_id").reset_index(drop=True)
        method_dir = video_out / method

        for _, row in group.iterrows():
            frame_id = int(row.frame_id)
            direction = row.direction
            frame_key = f"frame_{frame_id:06d}"

            # the transition frame itself
            _save_frame(
                cap, det_df, method, frame_id, method_dir / f"{frame_key}_{direction}.png",
                video_path, summary_rows, frame_type="transition", direction=direction,
                reference_frame_id=str(frame_id),
            )

            # CONTEXT_WINDOW frames before and after, clipped to video bounds
            ctx_dir = method_dir / "context" / f"{frame_key}_{direction}"
            for offset in list(range(-CONTEXT_WINDOW, 0)) + list(range(1, CONTEXT_WINDOW + 1)):
                ctx_frame_id = frame_id + offset
                if ctx_frame_id < 0 or ctx_frame_id >= total_frames:
                    continue
                sign = "+" if offset > 0 else ""
                ctx_path = ctx_dir / f"frame_{ctx_frame_id:06d}_offset{sign}{offset}.png"
                _save_frame(
                    cap, det_df, method, ctx_frame_id, ctx_path,
                    video_path, summary_rows, frame_type="context", direction=direction,
                    offset=offset, reference_frame_id=str(frame_id),
                )

        # midpoint frame between each consecutive pair of this method's transitions
        between_dir = method_dir / "between"
        for i in range(len(group) - 1):
            frame_a = int(group.loc[i, "frame_id"])
            frame_b = int(group.loc[i + 1, "frame_id"])
            mid_frame_id = (frame_a + frame_b) // 2
            if mid_frame_id == frame_a or mid_frame_id == frame_b:
                continue  # no room for a distinct frame between adjacent transitions
            mid_path = between_dir / f"frame_{mid_frame_id:06d}_between_{frame_a:06d}_{frame_b:06d}.png"
            _save_frame(
                cap, det_df, method, mid_frame_id, mid_path,
                video_path, summary_rows, frame_type="between",
                reference_frame_id=f"{frame_a}_{frame_b}",
            )

    cap.release()


def find_videos(directory: Path):
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    videos = find_videos(NEW_VIDEOS_DIR)
    print(f"Found {len(videos)} videos in {NEW_VIDEOS_DIR}")

    summary_csv = OUTPUT_ROOT / "transitions_summary.csv"
    summary_rows = pd.read_csv(summary_csv).to_dict("records") if summary_csv.exists() else []
    done_videos = {r["video"] for r in summary_rows}

    for video_path in videos:
        stem = video_path.stem
        video_out = OUTPUT_ROOT / stem

        if video_path.name in done_videos:
            print(f"Skipping {video_path.name} (already has saved transition frames)")
            continue

        print(f"\n{'='*60}\nVideo: {video_path.name}")
        video_out.mkdir(parents=True, exist_ok=True)

        try:
            detections_csv, transitions_csv = get_pipeline_outputs(video_path, video_out)
            extract_transition_pngs(video_path, detections_csv, transitions_csv, video_out, summary_rows)
        except Exception as e:
            print(f"  FAILED: {e}")

        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print(f"\nDone. Summary at {summary_csv}")


if __name__ == "__main__":
    main()
