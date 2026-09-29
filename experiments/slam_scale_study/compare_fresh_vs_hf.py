"""
compare_fresh_vs_hf.py
──────────────────────
3-panel plot: HF original | run_fresh raw | overlay
Mirrors the style of the cam_pose_scaled_comparison figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
DFPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "direct_360_FPE")

ORIG_PATH  = DFPE_DIR + "/mp3d_fpe_dataset/huggingface/1LXtFkjw3qL/0/vo_final/cam_pose_estimated.csv"
SLAM_PATH  = DFPE_DIR + "/mp3d_fpe_dataset/run_scaled/1LXtFkjw3qL/0/vo_final/cam_pose_estimated.csv"
OUT_PATH   = DFPE_DIR + "/slam_output/run_fresh/compare_scaled_vs_hf.png"


def load(path):
    data = np.loadtxt(path)
    return data[:, 1], data[:, 3]   # TX, TZ


tx_o, tz_o = load(ORIG_PATH)
tx_s, tz_s = load(SLAM_PATH)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("cam_pose_estimated: HF Original vs run_fresh_scaled", fontsize=13)

# Panel 1 — HF original only
axes[0].plot(tx_o, tz_o, "b-", lw=0.9, alpha=0.8, label=f"HF original ({len(tx_o)} kf)")
axes[0].set_title("Huggingface original")
axes[0].set_xlabel("TX"); axes[0].set_ylabel("TZ")
axes[0].axis("equal"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)

# Panel 2 — run_fresh raw only
axes[1].plot(tx_s, tz_s, "r-", lw=0.9, alpha=0.8, label=f"run_fresh raw ({len(tx_s)} kf)")
axes[1].set_title("run_fresh_scaled")
axes[1].set_xlabel("TX"); axes[1].set_ylabel("TZ")
axes[1].axis("equal"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)

# Panel 3 — overlay
axes[2].plot(tx_o, tz_o, "b-", lw=0.9, alpha=0.7, label=f"HF original ({len(tx_o)} kf)")
axes[2].plot(tx_s, tz_s, "r-", lw=0.9, alpha=0.7, label=f"run_fresh raw ({len(tx_s)} kf)")
axes[2].set_title("Raw vs Original overlay")
axes[2].set_xlabel("TX"); axes[2].set_ylabel("TZ")
axes[2].axis("equal"); axes[2].grid(alpha=0.3); axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
