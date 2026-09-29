#!/usr/bin/env python3
"""
Classify each frame in a video as indoor or outdoor based on OIV7 detections.

A frame is classified as INDOOR if:
- There is a detection of "House" or "Building" 
- The detection has area_pct > 90%
- The detection has confidence >= CONFIDENCE_THRESHOLD

Otherwise, the frame is classified as OUTDOOR.
"""

import argparse
import pandas as pd
from pathlib import Path


# Configuration
HOUSE_BUILDING_CLASSES = {"House", "Building"}
AREA_PCT_THRESHOLD = 98.0  
CONFIDENCE_THRESHOLD = 0.25


def classify_frames(detections_csv: Path, output_csv: Path = None):
    """
    Classify each frame as indoor or outdoor based on OIV7 detections.
    
    Uses global constants AREA_PCT_THRESHOLD and CONFIDENCE_THRESHOLD.
    
    Args:
        detections_csv: Path to OIV7 detections CSV file
        output_csv: Path to save classification results (optional)
    
    Returns:
        DataFrame with columns: frame_id, classification
    """
    
    # Read detections CSV
    print(f"Reading detections from: {detections_csv}")
    df = pd.read_csv(detections_csv)
    
    if df.empty:
        print("Warning: Detections CSV is empty")
        return pd.DataFrame(columns=["frame_id", "classification"])
    
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
    
    # Filter for house/building detections meeting criteria
    house_building_mask = (
        (df["class_name"].isin(HOUSE_BUILDING_CLASSES)) &
        (df["area_pct"] > AREA_PCT_THRESHOLD) &
        (df["conf"] >= CONFIDENCE_THRESHOLD)
    )
    
    house_building_detections = df[house_building_mask]
    
    print(f"  Found {len(house_building_detections)} house/building detections meeting criteria:")
    print(f"    - Class: {HOUSE_BUILDING_CLASSES}")
    print(f"    - Area > {AREA_PCT_THRESHOLD}%")
    print(f"    - Confidence >= {CONFIDENCE_THRESHOLD}")
    
    # Get all unique frame IDs
    all_frame_ids = sorted(df["frame_id"].unique())
    
    # Get frame IDs with qualifying house/building detections
    indoor_frame_ids = set(house_building_detections["frame_id"].unique())
    
    # Classify each frame
    classifications = []
    for frame_id in all_frame_ids:
        classification = "indoor" if frame_id in indoor_frame_ids else "outdoor"
        classifications.append({
            "frame_id": f"frame_{frame_id:06d}",
            "classification": classification
        })
    
    result_df = pd.DataFrame(classifications)
    
    # Print statistics
    indoor_count = len(indoor_frame_ids)
    outdoor_count = len(all_frame_ids) - indoor_count
    print(f"\nClassification results:")
    print(f"  Total frames: {len(all_frame_ids)}")
    print(f"  Indoor frames: {indoor_count} ({indoor_count/len(all_frame_ids)*100:.1f}%)")
    print(f"  Outdoor frames: {outdoor_count} ({outdoor_count/len(all_frame_ids)*100:.1f}%)")
    
    # Save to CSV if output path provided
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        print(f"\nSaved classification results to: {output_csv}")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Classify video frames as indoor or outdoor based on OIV7 detections"
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
        help="Path to save classification results CSV (default: detections_csv directory + 'indoor_outdoor_classification.csv')"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.detections_csv.exists():
        raise FileNotFoundError(f"Detections CSV not found: {args.detections_csv}")
    
    # Set default output path if not provided
    if args.output_csv is None:
        args.output_csv = args.detections_csv.parent / "indoor_outdoor_classification.csv"
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Area threshold: > {AREA_PCT_THRESHOLD}%")
    print(f"  Confidence threshold: >= {CONFIDENCE_THRESHOLD}")
    print()
    
    # Classify frames
    result_df = classify_frames(
        args.detections_csv,
        args.output_csv
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

