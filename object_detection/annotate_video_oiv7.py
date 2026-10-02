#!/usr/bin/env python3
"""
Run OIV7 object detection on all frames of a video and create annotated videos.

Creates two output videos:
1. All detections annotated
2. Filtered detections (only INDOOR_LIKE/OUTDOOR_LIKE objects with confidence >= 0.1)
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Import categories from find_indoor_objects.py
INDOOR_LIKE = {
    "Chair", "Couch", "Sofa bed", "Bed", "Table", "Coffee table", "Desk",
    "Toilet", "Sink", "Refrigerator", "Television", "Laptop",
    "Microwave oven", "Oven", "Closet", "Cabinetry", "Bathroom cabinet",
    "Bathtub", "Shower", "Mirror", "Window", "Door", "Door handle"
}
OUTDOOR_LIKE = {
    "Tree", "Car", "Street light", "Stop sign", "Traffic light", "Traffic sign",
    "Road", "Bus", "Truck", "Bicycle",
}

# Configuration
OIV7_MODEL_PATH = "/home/geolocation/yolov8l-oiv7.pt"
FILTER_CONFIDENCE_THRESHOLD = 0.1  # Confidence threshold for filtered video


def annotate_frame_with_detections(frame, results, model, filter_classes=None, min_conf=0.0):
    """
    Annotate a frame with YOLO detections.
    
    Args:
        frame: Input frame (numpy array)
        results: YOLO results object
        model: YOLO model (for class names)
        filter_classes: Set of class names to include (None = all classes)
        min_conf: Minimum confidence threshold
    
    Returns:
        Annotated frame
    """
    annotated_frame = frame.copy()
    
    if results.boxes is None or len(results.boxes) == 0:
        return annotated_frame
    
    # Draw bounding boxes and labels
    for box in results.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = model.names.get(cls_id, str(cls_id))
        
        # Apply filters
        if conf < min_conf:
            continue
        if filter_classes is not None and cls_name not in filter_classes:
            continue
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw bounding box
        color = (0, 255, 0)  # Green for all detections
        if filter_classes:
            # Use different colors for indoor vs outdoor
            if cls_name in INDOOR_LIKE:
                color = (255, 0, 0)  # Blue for indoor
            elif cls_name in OUTDOOR_LIKE:
                color = (0, 0, 255)  # Red for outdoor
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with class name and confidence
        label = f"{cls_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = max(y1 - 10, label_size[1] + 10)
        
        # Draw label background
        cv2.rectangle(
            annotated_frame,
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0] + 5, label_y + 5),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    return annotated_frame


def process_video(video_path: Path, output_dir: Path, model_path: Path, filter_conf: float = None):
    """
    Process video: run OIV7 on all frames and create annotated videos.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for results
        model_path: Path to OIV7 model
        filter_conf: Confidence threshold for filtered video (default: FILTER_CONFIDENCE_THRESHOLD)
    """
    if filter_conf is None:
        filter_conf = FILTER_CONFIDENCE_THRESHOLD
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
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create video writers
    video_name = video_path.stem
    all_detections_video = output_dir / f"{video_name}_all_detections.mp4"
    filtered_detections_video = output_dir / f"{video_name}_filtered_detections.mp4"
    
    # Define codec and create VideoWriter objects
    # Try different codecs for better compatibility
    codecs_to_try = ['mp4v', 'XVID', 'avc1']
    fourcc = None
    writer_all = None
    writer_filtered = None
    
    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer_all = cv2.VideoWriter(
                str(all_detections_video),
                fourcc,
                fps,
                (width, height)
            )
            writer_filtered = cv2.VideoWriter(
                str(filtered_detections_video),
                fourcc,
                fps,
                (width, height)
            )
            if writer_all.isOpened() and writer_filtered.isOpened():
                print(f"  Using codec: {codec}")
                break
        except Exception as e:
            continue
    
    if writer_all is None or not writer_all.isOpened() or not writer_filtered.isOpened():
        raise RuntimeError("Could not create video writers with any available codec")
    
    # Filter classes: combine indoor and outdoor
    filter_classes = INDOOR_LIKE | OUTDOOR_LIKE
    
    print(f"\nProcessing frames...")
    print(f"  Filter classes: {len(filter_classes)} categories")
    print(f"  Filter confidence threshold: {filter_conf}")
    
    frame_count = 0
    total_detections_all = 0
    total_detections_filtered = 0
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run OIV7 inference
            results = model.predict(
                source=frame,
                conf=0.01,  # Low threshold to get all detections
                iou=0.7,
                verbose=False
            )
            
            result = results[0]
            
            # Count detections
            if result.boxes is not None and len(result.boxes) > 0:
                total_detections_all += len(result.boxes)
                
                # Count filtered detections
                filtered_count = 0
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, str(cls_id))
                    if conf >= filter_conf and cls_name in filter_classes:
                        filtered_count += 1
                total_detections_filtered += filtered_count
            
            # Create annotated frame with all detections
            annotated_all = annotate_frame_with_detections(
                frame, result, model, filter_classes=None, min_conf=0.0
            )
            
            # Create annotated frame with filtered detections
            annotated_filtered = annotate_frame_with_detections(
                frame, result, model,
                filter_classes=filter_classes,
                min_conf=filter_conf
            )
            
            # Write frames to videos
            writer_all.write(annotated_all)
            writer_filtered.write(annotated_filtered)
            
            frame_count += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    writer_all.release()
    writer_filtered.release()
    
    print(f"\nProcessing complete!")
    print(f"  Processed {frame_count} frames")
    print(f"  Total detections (all): {total_detections_all}")
    print(f"  Total detections (filtered): {total_detections_filtered}")
    print(f"  Average detections per frame (all): {total_detections_all/frame_count:.2f}")
    print(f"  Average detections per frame (filtered): {total_detections_filtered/frame_count:.2f}")
    print(f"  All detections video: {all_detections_video}")
    print(f"  Filtered detections video: {filtered_detections_video}")


def main():
    parser = argparse.ArgumentParser(
        description="Run OIV7 on all video frames and create annotated videos"
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for annotated videos"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(OIV7_MODEL_PATH),
        help=f"Path to OIV7 model (default: {OIV7_MODEL_PATH})"
    )
    parser.add_argument(
        "--filter_conf",
        type=float,
        default=FILTER_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for filtered video (default: {FILTER_CONFIDENCE_THRESHOLD})"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    # Process video
    process_video(args.video, args.output_dir, args.model, filter_conf=args.filter_conf)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

