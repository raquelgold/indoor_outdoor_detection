#!/usr/bin/env python3
"""
Detect indoor/outdoor transition frames from classification CSVs.

A transition is confirmed only when at least MIN_RUN consecutive frames
show the new state. The reported frame_id is the first frame of that run.

Methods:
  objects_detection  — uses indoor_outdoor_objects_classification.csv
  building_area      — uses indoor_outdoor_classification.csv
"""

import pandas as pd
from pathlib import Path

MIN_RUN = 10  # consecutive frames required to confirm a transition


def _extract_frame_number(frame_id):
    if isinstance(frame_id, str) and frame_id.startswith("frame_"):
        return int(frame_id.split("_")[1])
    return int(frame_id)


def find_transitions(frame_ids: list, classifications: list) -> list:
    """
    Return list of dicts {frame_id, direction} for confirmed transitions.
    frame_ids and classifications must be sorted and aligned.
    """
    transitions = []
    n = len(frame_ids)
    if n == 0:
        return transitions

    current_state = classifications[0]
    i = 1

    while i <= n - MIN_RUN:
        if classifications[i] != current_state:
            new_state = classifications[i]
            if all(c == new_state for c in classifications[i: i + MIN_RUN]):
                direction = "OUT_TO_IN" if new_state == "indoor" else "IN_TO_OUT"
                transitions.append({"frame_id": frame_ids[i], "direction": direction})
                current_state = new_state
                i += MIN_RUN
            else:
                i += 1
        else:
            i += 1

    return transitions


def detect_transitions(objects_csv: Path, area_csv: Path, output_csv: Path = None) -> pd.DataFrame:
    """
    Detect transitions from both classification CSVs and combine into one table.

    Args:
        objects_csv: indoor_outdoor_objects_classification.csv
        area_csv:    indoor_outdoor_classification.csv
        output_csv:  where to save the result (optional)

    Returns:
        DataFrame with columns: frame_id, direction, method
    """
    rows = []

    for csv_path, method in [
        (objects_csv, "objects_detection"),
        (area_csv, "building_area"),
    ]:
        print(f"  [{method}] Reading {csv_path.name}")
        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"  [{method}] Warning: CSV is empty, skipping")
            continue

        df["_frame_num"] = df["frame_id"].apply(_extract_frame_number)
        df = df.sort_values("_frame_num").reset_index(drop=True)

        frame_ids = df["_frame_num"].tolist()
        classifications = df["classification"].tolist()

        transitions = find_transitions(frame_ids, classifications)
        print(f"  [{method}] Found {len(transitions)} transition(s)")

        for t in transitions:
            rows.append({
                "frame_id": t["frame_id"],
                "direction": t["direction"],
                "method": method,
            })

    result_df = pd.DataFrame(rows, columns=["frame_id", "direction", "method"])

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        print(f"  Saved transitions to: {output_csv}")

    return result_df
