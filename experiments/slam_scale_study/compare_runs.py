"""
compare_runs.py
───────────────
Produces:
  1. Per-run trajectory plots (all runs + original on one figure)
  2. TX / TZ / Yaw over time for all runs
  3. Pairwise comparison plots for every pair (run vs original, run vs run)
  4. One .txt report per pair with per-frame differences
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from itertools import combinations
import os
DFPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "direct_360_FPE")

# ── CONFIG ────────────────────────────────────────────────────────────────────
RUNS = {
    "Run1": DFPE_DIR + "/slam_output/run_1/keyframe_trajectory.txt",
    "Run2": DFPE_DIR + "/slam_output/run_2/keyframe_trajectory.txt",
    "Run3": DFPE_DIR + "/slam_output/run_3/keyframe_trajectory.txt",
    "Run4": DFPE_DIR + "/slam_output/run_4/keyframe_trajectory.txt",
}
ORIG_PATH = DFPE_DIR + "/mp3d_fpe_dataset/huggingface/1LXtFkjw3qL/0/vo_final/cam_pose_estimated.csv"
OUT_DIR   = DFPE_DIR + "/slam_output/analysis"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "Original": "#185FA5",
    "Run1":     "#E24B4A",
    "Run2":     "#1D9E75",
    "Run3":     "#BA7517",
    "Run4":     "#9B59B6",
}

# ── LOAD ──────────────────────────────────────────────────────────────────────
def load(path):
    data = np.loadtxt(path)
    ids  = data[:, 0]
    tx, ty, tz = data[:, 1], data[:, 2], data[:, 3]
    quat = data[:, 4:8]
    yaw = np.array([
        Rotation.from_quat(q).as_euler('yxz', degrees=True)[0]
        for q in quat
    ])
    # normalize to 0→1 so original (timestamps) and runs (frame IDs) share the same x axis
    progress = (ids - ids.min()) / (ids.max() - ids.min())
    return {"id": ids, "progress": progress, "tx": tx, "ty": ty, "tz": tz, "yaw": yaw}

all_data = {"Original": load(ORIG_PATH)}
for name, path in RUNS.items():
    all_data[name] = load(path)

run_names  = list(RUNS.keys())
all_names  = ["Original"] + run_names

# ── HELPER ────────────────────────────────────────────────────────────────────
def compare(a, b):
    # Match by nearest progress value (handles original timestamps vs frame IDs)
    # For run-vs-run: match on integer frame IDs (exact)
    # For original-vs-run: match by nearest progress
    a_prog = a["progress"]
    b_prog = b["progress"]

    # find closest point in b for each point in a
    matched_ia, matched_ib = [], []
    used = set()
    for i, pa in enumerate(a_prog):
        dists = np.abs(b_prog - pa)
        j = int(np.argmin(dists))
        if j not in used and dists[j] < 0.005:  # within 0.5% of trajectory
            matched_ia.append(i)
            matched_ib.append(j)
            used.add(j)

    ia = np.array(matched_ia)
    ib = np.array(matched_ib)
    shared_prog = a_prog[ia]
    return (shared_prog,
            a["tx"][ia], a["tz"][ia], a["yaw"][ia],
            b["tx"][ib], b["tz"][ib], b["yaw"][ib])

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — All trajectories
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("All trajectories", fontsize=13)

for name, d in all_data.items():
    c = COLORS.get(name, "gray")
    lw = 1.2 if name == "Original" else 0.8
    alpha = 0.5 if name == "Original" else 0.75
    label = f"{name} ({len(d['id'])} kf)"
    axes[0].plot(d["tx"], d["tz"], color=c, lw=lw, alpha=alpha, label=label)
    cx, cz = d["tx"] - d["tx"].mean(), d["tz"] - d["tz"].mean()
    axes[1].plot(cx, cz, color=c, lw=lw, alpha=alpha, label=label)

for ax, title in zip(axes, ["Top-down (TX vs TZ)", "Centered"]):
    ax.set_title(title); ax.set_xlabel("TX"); ax.set_ylabel("TZ")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/all_trajectories.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: all_trajectories.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — TX, TZ, Yaw over frame ID
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
fig.suptitle("TX / TZ / Yaw over time — all runs", fontsize=13)

for name, d in all_data.items():
    c = COLORS.get(name, "gray")
    lw = 1.2 if name == "Original" else 0.8
    alpha = 0.5 if name == "Original" else 0.75
    axes[0].plot(d["progress"], d["tx"],  color=c, lw=lw, alpha=alpha, label=name)
    axes[1].plot(d["progress"], d["tz"],  color=c, lw=lw, alpha=alpha, label=name)
    axes[2].plot(d["progress"], d["yaw"], color=c, lw=lw, alpha=alpha, label=name)

for ax, ylabel in zip(axes, ["TX (m)", "TZ (m)", "Yaw (°)"]):
    ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.legend(fontsize=8)
axes[2].set_xlabel("Trajectory progress (0=start, 1=end)")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/tx_tz_yaw_over_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: tx_tz_yaw_over_time.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Keyframe gap distribution
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Keyframe gap distribution", fontsize=13)
for name in run_names:
    d = all_data[name]
    gaps = np.diff(d["id"])
    ax.hist(gaps, bins=40, alpha=0.5, color=COLORS.get(name, "gray"),
            label=f"{name}  mean={gaps.mean():.1f}  std={gaps.std():.1f}  max={gaps.max()}")
ax.set_xlabel("Gap (frames)"); ax.set_ylabel("Count")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/keyframe_gaps.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: keyframe_gaps.png")

# ════════════════════════════════════════════════════════════════════════════
# PAIRWISE: run vs Original + run vs run
# ════════════════════════════════════════════════════════════════════════════
pairs_vs_orig = [("Original", r) for r in run_names]
pairs_vs_runs = list(combinations(run_names, 2))
all_pairs = pairs_vs_orig + pairs_vs_runs

for (nameA, nameB) in all_pairs:
    dA, dB = all_data[nameA], all_data[nameB]
    shared_prog, txA, tzA, yawA, txB, tzB, yawB = compare(dA, dB)

    if len(shared_prog) == 0:
        print(f"No shared keyframes: {nameA} vs {nameB} - skipping")
        continue

    tx_diff  = txA  - txB
    tz_diff  = tzA  - tzB
    yaw_diff = yawA - yawB
    euclid   = np.sqrt(tx_diff**2 + tz_diff**2)
    tag = f"{nameA}_vs_{nameB}"
    cA  = COLORS.get(nameA, "gray")
    cB  = COLORS.get(nameB, "gray")

    # ── plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{nameA} vs {nameB}  |  shared keyframes: {len(shared_prog)}", fontsize=13)

    # top-down trajectory
    axes[0,0].plot(dA["tx"], dA["tz"], color=cA, lw=0.8, alpha=0.8,
                   label=f"{nameA} ({len(dA['id'])} kf)")
    axes[0,0].plot(dB["tx"], dB["tz"], color=cB, lw=0.8, alpha=0.8,
                   label=f"{nameB} ({len(dB['id'])} kf)")
    axes[0,0].set_title("Trajectory top-down")
    axes[0,0].set_xlabel("TX"); axes[0,0].set_ylabel("TZ")
    axes[0,0].axis("equal"); axes[0,0].grid(alpha=0.3); axes[0,0].legend(fontsize=8)

    # centered trajectory
    axes[0,1].plot(dA["tx"]-dA["tx"].mean(), dA["tz"]-dA["tz"].mean(),
                   color=cA, lw=0.8, alpha=0.8, label=nameA)
    axes[0,1].plot(dB["tx"]-dB["tx"].mean(), dB["tz"]-dB["tz"].mean(),
                   color=cB, lw=0.8, alpha=0.8, label=nameB)
    axes[0,1].set_title("Centered trajectory")
    axes[0,1].set_xlabel("TX"); axes[0,1].set_ylabel("TZ")
    axes[0,1].axis("equal"); axes[0,1].grid(alpha=0.3); axes[0,1].legend(fontsize=8)

    # TX diff
    axes[0,2].plot(shared_prog, tx_diff, color=cA, lw=0.8, alpha=0.8)
    axes[0,2].axhline(0, color="black", lw=0.5, ls="--")
    axes[0,2].set_title(f"TX difference\nmean={np.abs(tx_diff).mean():.4f}m  max={np.abs(tx_diff).max():.4f}m")
    axes[0,2].set_xlabel("Progress"); axes[0,2].set_ylabel("TX diff (m)")
    axes[0,2].grid(alpha=0.3)

    # TZ diff
    axes[1,0].plot(shared_prog, tz_diff, color=cB, lw=0.8, alpha=0.8)
    axes[1,0].axhline(0, color="black", lw=0.5, ls="--")
    axes[1,0].set_title(f"TZ difference\nmean={np.abs(tz_diff).mean():.4f}m  max={np.abs(tz_diff).max():.4f}m")
    axes[1,0].set_xlabel("Progress"); axes[1,0].set_ylabel("TZ diff (m)")
    axes[1,0].grid(alpha=0.3)

    # Yaw diff
    axes[1,1].plot(shared_prog, yaw_diff, color="#9B59B6", lw=0.8, alpha=0.8)
    axes[1,1].axhline(0, color="black", lw=0.5, ls="--")
    axes[1,1].set_title(f"Yaw difference\nmean={np.abs(yaw_diff).mean():.2f}°  max={np.abs(yaw_diff).max():.2f}°")
    axes[1,1].set_xlabel("Progress"); axes[1,1].set_ylabel("Yaw diff (°)")
    axes[1,1].grid(alpha=0.3)

    # Euclidean diff
    axes[1,2].plot(shared_prog, euclid, color="#BA7517", lw=0.8, alpha=0.8)
    axes[1,2].axhline(euclid.mean(), color="red", lw=1, ls="--",
                      label=f"Mean={euclid.mean():.4f}m")
    axes[1,2].set_title(f"Euclidean diff (TX+TZ)\nmean={euclid.mean():.4f}m  max={euclid.max():.4f}m")
    axes[1,2].set_xlabel("Progress"); axes[1,2].set_ylabel("Distance (m)")
    axes[1,2].legend(fontsize=8); axes[1,2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {tag}.png")

    # ── txt report ────────────────────────────────────────────────────────
    txt_path = f"{OUT_DIR}/{tag}_diff.txt"
    with open(txt_path, "w") as f:
        f.write(f"Pairwise comparison: {nameA} vs {nameB}\n")
        f.write(f"Shared keyframes : {len(shared_prog)}\n")
        f.write(f"Only in {nameA:<10}: {len(dA['id']) - len(shared_prog)}\n")
        f.write(f"Only in {nameB:<10}: {len(dB['id']) - len(shared_prog)}\n")
        f.write(f"\nSummary (shared frames only):\n")
        f.write(f"  TX   mean_abs={np.abs(tx_diff).mean():.6f}m  max_abs={np.abs(tx_diff).max():.6f}m\n")
        f.write(f"  TZ   mean_abs={np.abs(tz_diff).mean():.6f}m  max_abs={np.abs(tz_diff).max():.6f}m\n")
        f.write(f"  Yaw  mean_abs={np.abs(yaw_diff).mean():.4f}°  max_abs={np.abs(yaw_diff).max():.4f}°\n")
        f.write(f"  Eucl mean={euclid.mean():.6f}m  max={euclid.max():.6f}m\n")
        f.write(f"\n{'frame_id':>10}  {'txA':>12}  {'tzA':>12}  {'yawA':>10}  "
                f"{'txB':>12}  {'tzB':>12}  {'yawB':>10}  "
                f"{'tx_diff':>12}  {'tz_diff':>12}  {'yaw_diff':>10}  {'euclidean':>12}\n")
        f.write("-" * 130 + "\n")
        for vals in zip(shared_prog, txA, tzA, yawA, txB, tzB, yawB,
                        tx_diff, tz_diff, yaw_diff, euclid):
            fid, txa, tza, ywa, txb, tzb, ywb, txd, tzd, yd, eu = vals
            f.write(f"{fid:>10.4f}  {txa:>12.6f}  {tza:>12.6f}  {ywa:>10.4f}  "
                    f"{txb:>12.6f}  {tzb:>12.6f}  {ywb:>10.4f}  "
                    f"{txd:>12.6f}  {tzd:>12.6f}  {yd:>10.4f}  {eu:>12.6f}\n")
    print(f"Saved: {tag}_diff.txt")

# ════════════════════════════════════════════════════════════════════════════
# TERMINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n── Per-run summary ──")
for name in run_names:
    d = all_data[name]
    gaps = np.diff(d["id"])
    print(f"  {name}: {len(d['id'])} kf | gap mean={gaps.mean():.1f} "
          f"std={gaps.std():.1f} max={gaps.max()} @ frame {d['id'][np.argmax(gaps)]}")

print(f"\nAll outputs saved to: {OUT_DIR}/")