"""
scale_cam_pose.py
─────────────────
Multiplies TX (col 1) and TZ (col 3) of cam_pose_estimated.csv
by their respective scale ratios, saves a corrected copy.
"""

import numpy as np
import os
DFPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "direct_360_FPE")

INPUT  = DFPE_DIR + "/mp3d_fpe_dataset/my_run/1LXtFkjw3qL/0/vo_final/cam_pose_estimated.csv"
OUTPUT = DFPE_DIR + "/mp3d_fpe_dataset/my_run/1LXtFkjw3qL/0/vo_final/cam_pose_estimated_scaled.csv"

# Per-axis ratios from ratio_scatter.py (Run1 results — most keyframes, lowest std)
TX_RATIO = 1.136
TZ_RATIO = 1.170

data = np.loadtxt(INPUT)
print(f"Loaded {len(data)} keyframes from {INPUT}")
print(f"Before — TX range: {data[:,1].min():.4f} → {data[:,1].max():.4f}")
print(f"Before — TZ range: {data[:,3].min():.4f} → {data[:,3].max():.4f}")

scaled = data.copy()
scaled[:, 1] *= TX_RATIO   # TX
scaled[:, 3] *= TZ_RATIO   # TZ

print(f"\nAfter  — TX range: {scaled[:,1].min():.4f} → {scaled[:,1].max():.4f}")
print(f"After  — TZ range: {scaled[:,3].min():.4f} → {scaled[:,3].max():.4f}")

# Save with same precision as original
np.savetxt(OUTPUT, scaled, fmt="%.10g")
print(f"\nSaved scaled poses to {OUTPUT}")
print("Next: point DFPE at this file by copying it over cam_pose_estimated.csv")
print("  cp", OUTPUT, INPUT.replace(".csv", "_original_backup.csv"))
print("  cp", OUTPUT, INPUT)