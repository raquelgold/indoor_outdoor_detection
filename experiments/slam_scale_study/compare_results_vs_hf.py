"""
compare_results_vs_hf.py
────────────────────────
2×2 grid comparing scale_recovery.jpg and final_fp.png
between the HuggingFace original run and run_no_scale.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
DFPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "direct_360_FPE")

HF_DIR  = DFPE_DIR + "/test/1LXtFkjw3qL_0"
NS_DIR  = DFPE_DIR + "/test_run_no_scale/1LXtFkjw3qL_0"
OUT     = DFPE_DIR + "/test_run_no_scale/1LXtFkjw3qL_0/comparison_vs_hf.png"

imgs = {
    ("scale_recovery", "HF original"):  f"{HF_DIR}/scale_recovery.jpg",
    ("scale_recovery", "run_no_scale"): f"{NS_DIR}/scale_recovery.jpg",
    ("final_fp",       "HF original"):  f"{HF_DIR}/1LXtFkjw3qL_0_final_fp.png",
    ("final_fp",       "run_no_scale"): f"{NS_DIR}/1LXtFkjw3qL_0_final_fp.png",
}

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("run_no_scale vs HuggingFace original — 1LXtFkjw3qL scene 0", fontsize=13)

titles = [
    ("scale_recovery", "HF original"),
    ("scale_recovery", "run_no_scale"),
    ("final_fp",       "HF original"),
    ("final_fp",       "run_no_scale"),
]

for ax, key in zip(axes.flat, titles):
    img = mpimg.imread(imgs[key])
    ax.imshow(img)
    ax.set_title(f"{key[0]}  —  {key[1]}", fontsize=11)
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
