#!/usr/bin/env python3
"""
Run OIV7 on all frames of a video and create two annotated videos focused on
House/Building detections, using the same logic as find_indoor_area.py.

Outputs:
1. high_conf_house_building.mp4  - only House/Building detections with
   area_pct > AREA_PCT_THRESHOLD and conf >= CONFIDENCE_THRESHOLD.
2. all_house_building.mp4        - all House/Building detections regardless
   of confidence or area.

Both outputs keep the full video length (no frame skipping); they just differ
in which detections are drawn on top.
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

# Same configuration as find_indoor_area.py
HOUSE_BUILDING_CLASSES = {"House", "Building"}
AREA_PCT_THRESHOLD = 98.0
CONFIDENCE_THRESHOLD = 0.25

# Model path (reuse from annotate_video_oiv7.py)
OIV7_MODEL_PATH = "/home/geolocation/yolov8l-oiv7.pt"


def annotate_frame_with_house_building(
    frame,
    results,
    model,
    draw_high_conf_only: bool,
) -> "cv2.Mat":
    """
    Annotate a frame with House/Building detections.

    Args:
        frame: Input frame (numpy array)
        results: YOLO results object
        model: YOLO model (for class names)
        draw_high_conf_only: If True, only draw detections that satisfy
            area_pct > AREA_PCT_THRESHOLD and conf >= CONFIDENCE_THRESHOLD.
            If False, draw all House/Building detections regardless of conf/area.

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    if results.boxes is None or len(results.boxes) == 0:
        return annotated

    img_h, img_w = frame.shape[:2]
    frame_area = float(img_w * img_h)

    for box in results.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = model.names.get(cls_id, str(cls_id))

        if cls_name not in HOUSE_BUILDING_CLASSES:
            continue

        # Box coords
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Compute area_pct like in detections CSV
        box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        area_pct = 0.0
        if frame_area > 0:
            area_pct = box_area / frame_area * 100.0

        if draw_high_conf_only:
            if not (area_pct > AREA_PCT_THRESHOLD and conf >= CONFIDENCE_THRESHOLD):
                continue

        # Draw bounding box
        color = (0, 255, 255) if draw_high_conf_only else (255, 0, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label: "<class_name> conf area_pct%"
        label = f"{cls_name} {conf:.2f} {area_pct:.1f}%"
        label_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_y = max(y1 - 10, label_size[1] + 10)

        # Label background
        cv2.rectangle(
            annotated,
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0] + 5, label_y + 5),
            color,
            -1,
        )

        # Label text
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return annotated


def process_video(
    video_path: Path,
    output_dir: Path,
    model_path: Path,
):
    """
    Process video: run OIV7 on all frames and create two annotated videos:
    - high_conf_house_building.mp4
    - all_house_building.mp4
    """
    print(f"Loading OIV7 model from: {model_path}")
    model = YOLO(str(model_path))

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = video_path.stem
    high_conf_video = output_dir / f"{video_name}_house_building_high_conf.mp4"
    all_conf_video = output_dir / f"{video_name}_house_building_all_conf.mp4"

    # Create video writers (same pattern as annotate_video_oiv7.py)
    codecs_to_try = ["mp4v", "XVID", "avc1"]
    writer_high = None
    writer_all = None

    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            tmp_high = cv2.VideoWriter(
                str(high_conf_video),
                fourcc,
                fps,
                (width, height),
            )
            tmp_all = cv2.VideoWriter(
                str(all_conf_video),
                fourcc,
                fps,
                (width, height),
            )
            if tmp_high.isOpened() and tmp_all.isOpened():
                writer_high = tmp_high
                writer_all = tmp_all
                print(f"  Using codec: {codec}")
                break
        except Exception:
            continue

    if writer_high is None or not writer_high.isOpened() or writer_all is None or not writer_all.isOpened():
        raise RuntimeError("Could not create video writers with any available codec")

    print("\nProcessing frames for House/Building annotations...")
    print(f"  High-confidence thresholds: area > {AREA_PCT_THRESHOLD}%, conf >= {CONFIDENCE_THRESHOLD}")

    frame_count = 0
    high_conf_frames = 0
    all_conf_frames = 0

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run OIV7 inference on this frame
            results = model.predict(
                source=frame,
                conf=0.01,  # low threshold; we'll filter ourselves
                iou=0.7,
                verbose=False,
            )
            result = results[0]

            # Annotated versions
            annotated_high = annotate_frame_with_house_building(
                frame, result, model, draw_high_conf_only=True
            )
            annotated_all = annotate_frame_with_house_building(
                frame, result, model, draw_high_conf_only=False
            )

            # Track how many frames actually had at least one box
            if (annotated_high != frame).any():
                high_conf_frames += 1
            if (annotated_all != frame).any():
                all_conf_frames += 1

            writer_high.write(annotated_high)
            writer_all.write(annotated_all)

            frame_count += 1
            pbar.update(1)

    cap.release()
    writer_high.release()
    writer_all.release()

    print("\nProcessing complete!")
    print(f"  Processed frames: {frame_count}")
    print(f"  Frames with high-conf House/Building: {high_conf_frames}")
    print(f"  Frames with any House/Building: {all_conf_frames}")
    print(f"  High-conf video: {high_conf_video}")
    print(f"  All-conf video:  {all_conf_video}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create two annotated videos highlighting House/Building frames "
            "using OIV7, mirroring the logic from find_indoor_area.py."
        )
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for annotated videos",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(OIV7_MODEL_PATH),
        help=f"Path to OIV7 model (default: {OIV7_MODEL_PATH})",
    )

    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    process_video(args.video, args.output_dir, args.model)

    print("\nDone!")


if __name__ == "__main__":
    main()



