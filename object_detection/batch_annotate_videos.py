#!/usr/bin/env python3
"""
Batch process all videos in a directory with OIV7 annotation.

Runs annotate_video_oiv7.py on all video files in the specified directory.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

# Path to the annotation script
ANNOTATE_SCRIPT = Path(__file__).parent / "annotate_video_oiv7.py"

# Supported video extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'}


def find_videos(directory: Path):
    """Find all video files in a directory (recursively)."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(directory.rglob(f"*{ext}"))
        videos.extend(directory.rglob(f"*{ext.upper()}"))
    return sorted(videos)


def process_video(video_path: Path, output_base_dir: Path, model_path: Path, filter_conf: float):
    """Process a single video using the annotation script."""
    # Create output directory for this video (using video name)
    video_output_dir = output_base_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        str(ANNOTATE_SCRIPT),
        "--video", str(video_path),
        "--output_dir", str(video_output_dir),
        "--model", str(model_path),
        "--filter_conf", str(filter_conf)
    ]
    
    # Run the annotation script
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return result.returncode == 0, result.stdout, result.stderr


def main():
    parser = argparse.ArgumentParser(
        description="Batch process all videos in a directory with OIV7 annotation"
    )
    parser.add_argument(
        "--videos_dir",
        type=Path,
        required=True,
        help="Directory containing video files to process"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Base output directory for annotated videos"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/geolocation/yolov8l-oiv7.pt"),
        help="Path to OIV7 model"
    )
    parser.add_argument(
        "--filter_conf",
        type=float,
        default=0.1,
        help="Confidence threshold for filtered video (default: 0.1)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos that already have output files"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {args.videos_dir}")
    
    if not args.videos_dir.is_dir():
        raise ValueError(f"Path is not a directory: {args.videos_dir}")
    
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    if not ANNOTATE_SCRIPT.exists():
        raise FileNotFoundError(f"Annotation script not found: {ANNOTATE_SCRIPT}")
    
    # Find all videos
    print(f"Searching for videos in: {args.videos_dir}")
    videos = find_videos(args.videos_dir)
    
    if not videos:
        print(f"No video files found in {args.videos_dir}")
        print(f"Supported extensions: {', '.join(VIDEO_EXTENSIONS)}")
        return
    
    print(f"Found {len(videos)} video(s) to process")
    print()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each video
    successful = []
    failed = []
    
    for video_path in tqdm(videos, desc="Processing videos"):
        # Check if already processed (if skip_existing is enabled)
        if args.skip_existing:
            video_output_dir = args.output_dir / video_path.stem
            all_detections_video = video_output_dir / f"{video_path.stem}_all_detections.mp4"
            filtered_detections_video = video_output_dir / f"{video_path.stem}_filtered_detections.mp4"
            
            if all_detections_video.exists() and filtered_detections_video.exists():
                print(f"\nSkipping {video_path.name} (already processed)")
                successful.append(video_path)
                continue
        
        print(f"\nProcessing: {video_path.name}")
        success, stdout, stderr = process_video(
            video_path, args.output_dir, args.model, args.filter_conf
        )
        
        if success:
            successful.append(video_path)
            print(f"  ✓ Success: {video_path.name}")
        else:
            failed.append(video_path)
            print(f"  ✗ Failed: {video_path.name}")
            if stderr:
                print(f"  Error: {stderr[:500]}")  # Print first 500 chars of error
    
    # Print summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total videos: {len(videos)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccessfully processed videos:")
        for video in successful:
            print(f"  - {video.name}")
    
    if failed:
        print(f"\nFailed videos:")
        for video in failed:
            print(f"  - {video.name}")
        print("\nCheck error messages above for details.")
        sys.exit(1)
    
    print(f"\nAll videos processed successfully!")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

