#!/usr/bin/env python3
"""
Classify each frame in a video as indoor or outdoor based on detected objects.

A frame is classified by counting indoor-like vs outdoor-like objects detected.
Uses the same object categories as finding_suspicious.py.
"""

import argparse
import pandas as pd
from pathlib import Path


# Configuration - Object categories
INDOOR_LIKE = {
    "Chair", "Couch", "Sofa bed", "Bed", "Table", "Coffee table", "Desk",
    "Toilet", "Sink", "Refrigerator", "Television", "Laptop",
    "Microwave oven", "Oven", "Closet", "Cabinetry", "Bathroom cabinet",
    "Bathtub", "Shower", "Mirror", "Window", "Door", "Door handle", "Stairs", "Furniture"
}
OUTDOOR_LIKE = {
    "Tree", "Car", "Street light", "Stop sign", "Traffic light", "Traffic sign",
    "Road", "Bus", "Truck", "Bicycle"
}

# Classification thresholds
MIN_CONFIDENCE = 0.1  # Minimum confidence for detections to count
MIN_OBJECTS_REQUIRED = 1  # Minimum number of objects required to classify (0 = use majority even with 1 object)


def classify_frames_by_objects(detections_csv: Path, output_csv: Path = None):
    """
    Classify each frame as indoor or outdoor based on detected objects.
    
    Uses global constants INDOOR_LIKE, OUTDOOR_LIKE, and thresholds.
    
    Args:
        detections_csv: Path to OIV7 detections CSV file
        output_csv: Path to save classification results (optional)
    
    Returns:
        DataFrame with columns: frame_id, classification, indoor_score, outdoor_score
    """
    # Read detections CSV
    print(f"Reading detections from: {detections_csv}")
    df = pd.read_csv(detections_csv)
    
    if df.empty:
        print("Warning: Detections CSV is empty")
        return pd.DataFrame(columns=["frame_id", "classification", "indoor_score", "outdoor_score"])
    
    print(f"  Loaded {len(df)} detections across {df['frame_id'].nunique()} frames")
    
    # Convert frame_id to numeric if it's in "frame_00xxxx" format
    def extract_frame_number(x):
        if isinstance(x, str) and x.startswith("frame_"):
            try:
                return int(x.split("_")[1])
            except (ValueError, IndexError):
                return int(x)
        return int(x)
    df["frame_id"] = df["frame_id"].apply(extract_frame_number)
    
    # Filter detections by minimum confidence
    df_filtered = df[df["conf"] >= MIN_CONFIDENCE].copy()
    print(f"  After confidence filter (>= {MIN_CONFIDENCE}): {len(df_filtered)} detections")
    
    # Get all unique frame IDs
    all_frame_ids = sorted(df["frame_id"].unique())
    
    # Classify each frame
    classifications = []
    indoor_counts = []
    outdoor_counts = []
    last_classification = "outdoor"  # default before any frame is seen

    for frame_id in all_frame_ids:
        # Get all detections for this frame
        frame_detections = df_filtered[df_filtered["frame_id"] == frame_id]

        # Count indoor and outdoor objects
        indoor_objects = frame_detections[frame_detections["class_name"].isin(INDOOR_LIKE)]
        outdoor_objects = frame_detections[frame_detections["class_name"].isin(OUTDOOR_LIKE)]

        indoor_count = len(indoor_objects)
        outdoor_count = len(outdoor_objects)

        # Calculate weighted scores
        indoor_score = indoor_count
        outdoor_score = outdoor_count

        # Classify based on scores
        total_objects = indoor_count + outdoor_count

        if total_objects < MIN_OBJECTS_REQUIRED:
            classification = last_classification
        elif indoor_score > outdoor_score:
            classification = "indoor"
        elif outdoor_score > indoor_score:
            classification = "outdoor"
        else:
            # Tie — carry forward last known state
            classification = last_classification

        last_classification = classification
        
        classifications.append({
            "frame_id": f"frame_{frame_id:06d}",
            "classification": classification,
            "indoor_score": indoor_score,
            "outdoor_score": outdoor_score,
            "indoor_count": indoor_count,
            "outdoor_count": outdoor_count
        })
        
        indoor_counts.append(indoor_count)
        outdoor_counts.append(outdoor_count)
    
    result_df = pd.DataFrame(classifications)
    
    # Print statistics
    indoor_frames = len(result_df[result_df["classification"] == "indoor"])
    outdoor_frames = len(result_df[result_df["classification"] == "outdoor"])
    total_frames = len(all_frame_ids)
    
    print(f"\nClassification results:")
    print(f"  Total frames: {total_frames}")
    print(f"  Indoor frames: {indoor_frames} ({indoor_frames/total_frames*100:.1f}%)")
    print(f"  Outdoor frames: {outdoor_frames} ({outdoor_frames/total_frames*100:.1f}%)")
    print(f"\nObject detection statistics:")
    print(f"  Total indoor objects detected: {sum(indoor_counts)}")
    print(f"  Total outdoor objects detected: {sum(outdoor_counts)}")
    print(f"  Average indoor objects per frame: {sum(indoor_counts)/total_frames:.2f}")
    print(f"  Average outdoor objects per frame: {sum(outdoor_counts)/total_frames:.2f}")
    
    # Save to CSV if output path provided
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        print(f"\nSaved classification results to: {output_csv}")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Classify video frames as indoor or outdoor based on detected objects"
    )
    parser.add_argument(
        "--detections_csv",
        type=Path,
        required=True,
        help="Path to OIV7 detections CSV file"
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Path to save classification results CSV (default: detections_csv directory + 'indoor_outdoor_objects_classification.csv')"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.detections_csv.exists():
        raise FileNotFoundError(f"Detections CSV not found: {args.detections_csv}")
    
    # Set default output path if not provided
    if args.output_csv is None:
        args.output_csv = args.detections_csv.parent / "indoor_outdoor_objects_classification.csv"
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Indoor objects: {len(INDOOR_LIKE)} categories")
    print(f"  Outdoor objects: {len(OUTDOOR_LIKE)} categories")
    print(f"  Min confidence: {MIN_CONFIDENCE}")
    print(f"  Min objects required: {MIN_OBJECTS_REQUIRED}")
    print()
    
    # Classify frames
    result_df = classify_frames_by_objects(
        args.detections_csv,
        args.output_csv
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

