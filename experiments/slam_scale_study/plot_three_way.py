"""
plot_three_way.py
─────────────────
All trajectory plots comparing:
  • HF original  (huggingface cam_pose_estimated.csv)
  • run_no_scale (raw SLAM, no scale correction)
  • run_scaled   (TX * TX_RATIO, TZ * TZ_RATIO)

Produces (saved to OUT_DIR):
  1. three_panel_topdown.png     — HF | no_scale | scaled side-by-side
  2. all_overlay.png             — all three overlaid (raw + centered)
  3. tx_tz_yaw_over_time.png     — TX / TZ / Yaw vs progress
  4. hf_vs_no_scale.png          — pairwise diff plot
  5. hf_vs_scaled.png            — pairwise diff plot
  6. no_scale_vs_scaled.png      — pairwise diff plot
  7. ratio_no_scale.png          — scale ratio scatter (no_scale vs HF)
  8. ratio_scaled.png            — scale ratio scatter (scaled vs HF)
  + .txt diff reports for each pair
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os
DFPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "direct_360_FPE")

# ── PATHS ─────────────────────────────────────────────────────────────────────
HF_PATH = DFPE_DIR + "/mp3d_fpe_dataset/huggingface/1LXtFkjw3qL/0/vo_final/cam_pose_estimated.csv"
NS_PATH = DFPE_DIR + "/mp3d_fpe_dataset/run_no_scale/1LXtFkjw3qL/0/vo_final/cam_pose_estimated.csv"
SC_PATH = DFPE_DIR + "/mp3d_fpe_dataset/run_scaled/1LXtFkjw3qL/0/vo_final/cam_pose_estimated.csv"
OUT_DIR = DFPE_DIR + "/slam_output/analysis/three_way"

COLORS  = {"HF original": "#185FA5", "run_no_scale": "#E24B4A", "run_scaled": "#1D9E75"}
PROGRESS_TOL = 0.005
MIN_ABS      = 0.05

os.makedirs(OUT_DIR, exist_ok=True)

# ── LOAD ──────────────────────────────────────────────────────────────────────
def load(path):
    data = np.loadtxt(path)
    ids  = data[:, 0]
    tx, ty, tz = data[:, 1], data[:, 2], data[:, 3]
    quat = data[:, 4:8]
    yaw  = np.array([Rotation.from_quat(q).as_euler('yxz', degrees=True)[0] for q in quat])
    progress = (ids - ids.min()) / (ids.max() - ids.min())
    return {"id": ids, "progress": progress, "tx": tx, "ty": ty, "tz": tz, "yaw": yaw}

datasets = {
    "HF original":  load(HF_PATH),
    "run_no_scale": load(NS_PATH),
    "run_scaled":   load(SC_PATH),
}

for name, d in datasets.items():
    print(f"{name:15s}: {len(d['id'])} kf  |  "
          f"TX [{d['tx'].min():.3f}, {d['tx'].max():.3f}]  "
          f"TZ [{d['tz'].min():.3f}, {d['tz'].max():.3f}]")

# ── PAIRWISE MATCH ─────────────────────────────────────────────────────────────
def match(a, b):
    matched_ia, matched_ib = [], []
    used = set()
    for i, pa in enumerate(a["progress"]):
        dists = np.abs(b["progress"] - pa)
        j = int(np.argmin(dists))
        if j not in used and dists[j] < PROGRESS_TOL:
            matched_ia.append(i); matched_ib.append(j); used.add(j)
    ia, ib = np.array(matched_ia), np.array(matched_ib)
    if len(ia) == 0:
        return None
    return {
        "prog": a["progress"][ia],
        "txA": a["tx"][ia], "tzA": a["tz"][ia], "yawA": a["yaw"][ia],
        "txB": b["tx"][ib], "tzB": b["tz"][ib], "yawB": b["yaw"][ib],
    }

# ── RATIO SCATTER helper ───────────────────────────────────────────────────────
def compute_ratios(txO, tzO, txR, tzR):
    tx_m = np.abs(txR) > MIN_ABS
    tx_r = txO[tx_m] / txR[tx_m]
    tx_r = tx_r[(tx_r > -10) & (tx_r < 10)]

    tz_m = np.abs(tzR) > MIN_ABS
    tz_r = tzO[tz_m] / tzR[tz_m]
    tz_r = tz_r[(tz_r > -10) & (tz_r < 10)]
    return tx_r, tz_r


# ════════════════════════════════════════════════════════════════════════════════
# 1. Three-panel top-down: HF | no_scale | scaled
# ════════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("cam_pose_estimated: HF Original  |  run_no_scale  |  run_scaled", fontsize=13)

labels = ["Huggingface original", "run_no_scale (raw)", "run_scaled"]
keys   = ["HF original", "run_no_scale", "run_scaled"]

for ax, key, title in zip(axes, keys, labels):
    d = datasets[key]
    c = COLORS[key]
    ax.plot(d["tx"], d["tz"], color=c, lw=0.9, alpha=0.85,
            label=f"{title} ({len(d['id'])} kf)")
    ax.set_title(title); ax.set_xlabel("TX"); ax.set_ylabel("TZ")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/three_panel_topdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: three_panel_topdown.png")


# ════════════════════════════════════════════════════════════════════════════════
# 2. All overlay (raw + centered)
# ════════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("All trajectories — overlay", fontsize=13)

for name, d in datasets.items():
    c  = COLORS[name]
    lw = 1.2 if name == "HF original" else 0.85
    al = 0.5 if name == "HF original" else 0.75
    label = f"{name} ({len(d['id'])} kf)"
    axes[0].plot(d["tx"], d["tz"], color=c, lw=lw, alpha=al, label=label)
    axes[1].plot(d["tx"] - d["tx"].mean(), d["tz"] - d["tz"].mean(),
                 color=c, lw=lw, alpha=al, label=label)

for ax, title in zip(axes, ["Top-down (raw)", "Centered"]):
    ax.set_title(title); ax.set_xlabel("TX"); ax.set_ylabel("TZ")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/all_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: all_overlay.png")


# ════════════════════════════════════════════════════════════════════════════════
# 3. TX / TZ / Yaw over trajectory progress
# ════════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
fig.suptitle("TX / TZ / Yaw over trajectory progress", fontsize=13)

for name, d in datasets.items():
    c  = COLORS[name]
    lw = 1.2 if name == "HF original" else 0.85
    al = 0.5 if name == "HF original" else 0.75
    axes[0].plot(d["progress"], d["tx"],  color=c, lw=lw, alpha=al, label=name)
    axes[1].plot(d["progress"], d["tz"],  color=c, lw=lw, alpha=al, label=name)
    axes[2].plot(d["progress"], d["yaw"], color=c, lw=lw, alpha=al, label=name)

for ax, ylabel in zip(axes, ["TX (m)", "TZ (m)", "Yaw (°)"]):
    ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.legend(fontsize=9)
axes[2].set_xlabel("Trajectory progress (0 = start, 1 = end)")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/tx_tz_yaw_over_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: tx_tz_yaw_over_time.png")


# ════════════════════════════════════════════════════════════════════════════════
# 4–6. Pairwise diff plots + txt reports
# ════════════════════════════════════════════════════════════════════════════════
pairs = [
    ("HF original",  "run_no_scale", "hf_vs_no_scale"),
    ("HF original",  "run_scaled",   "hf_vs_scaled"),
    ("run_no_scale", "run_scaled",   "no_scale_vs_scaled"),
]

for nameA, nameB, tag in pairs:
    dA, dB = datasets[nameA], datasets[nameB]
    m = match(dA, dB)
    if m is None:
        print(f"No shared keyframes: {nameA} vs {nameB} — skipping")
        continue

    tx_diff  = m["txA"]  - m["txB"]
    tz_diff  = m["tzA"]  - m["tzB"]
    yaw_diff = m["yawA"] - m["yawB"]
    euclid   = np.sqrt(tx_diff**2 + tz_diff**2)
    cA, cB   = COLORS[nameA], COLORS[nameB]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{nameA}  vs  {nameB}  |  matched keyframes: {len(m['prog'])}", fontsize=13)

    # top-down raw
    axes[0,0].plot(dA["tx"], dA["tz"], color=cA, lw=0.8, alpha=0.8,
                   label=f"{nameA} ({len(dA['id'])} kf)")
    axes[0,0].plot(dB["tx"], dB["tz"], color=cB, lw=0.8, alpha=0.8,
                   label=f"{nameB} ({len(dB['id'])} kf)")
    axes[0,0].set_title("Trajectory top-down")
    axes[0,0].set_xlabel("TX"); axes[0,0].set_ylabel("TZ")
    axes[0,0].axis("equal"); axes[0,0].grid(alpha=0.3); axes[0,0].legend(fontsize=8)

    # centered
    axes[0,1].plot(dA["tx"]-dA["tx"].mean(), dA["tz"]-dA["tz"].mean(),
                   color=cA, lw=0.8, alpha=0.8, label=nameA)
    axes[0,1].plot(dB["tx"]-dB["tx"].mean(), dB["tz"]-dB["tz"].mean(),
                   color=cB, lw=0.8, alpha=0.8, label=nameB)
    axes[0,1].set_title("Centered trajectory")
    axes[0,1].set_xlabel("TX"); axes[0,1].set_ylabel("TZ")
    axes[0,1].axis("equal"); axes[0,1].grid(alpha=0.3); axes[0,1].legend(fontsize=8)

    # TX diff
    axes[0,2].plot(m["prog"], tx_diff, color=cA, lw=0.8, alpha=0.8)
    axes[0,2].axhline(0, color="black", lw=0.5, ls="--")
    axes[0,2].set_title(f"TX difference\nmean={np.abs(tx_diff).mean():.4f}m  max={np.abs(tx_diff).max():.4f}m")
    axes[0,2].set_xlabel("Progress"); axes[0,2].set_ylabel("TX diff (m)"); axes[0,2].grid(alpha=0.3)

    # TZ diff
    axes[1,0].plot(m["prog"], tz_diff, color=cB, lw=0.8, alpha=0.8)
    axes[1,0].axhline(0, color="black", lw=0.5, ls="--")
    axes[1,0].set_title(f"TZ difference\nmean={np.abs(tz_diff).mean():.4f}m  max={np.abs(tz_diff).max():.4f}m")
    axes[1,0].set_xlabel("Progress"); axes[1,0].set_ylabel("TZ diff (m)"); axes[1,0].grid(alpha=0.3)

    # Yaw diff
    axes[1,1].plot(m["prog"], yaw_diff, color="#9B59B6", lw=0.8, alpha=0.8)
    axes[1,1].axhline(0, color="black", lw=0.5, ls="--")
    axes[1,1].set_title(f"Yaw difference\nmean={np.abs(yaw_diff).mean():.2f}°  max={np.abs(yaw_diff).max():.2f}°")
    axes[1,1].set_xlabel("Progress"); axes[1,1].set_ylabel("Yaw diff (°)"); axes[1,1].grid(alpha=0.3)

    # Euclidean
    axes[1,2].plot(m["prog"], euclid, color="#BA7517", lw=0.8, alpha=0.8)
    axes[1,2].axhline(euclid.mean(), color="red", lw=1, ls="--",
                      label=f"Mean={euclid.mean():.4f}m")
    axes[1,2].set_title(f"Euclidean diff (TX+TZ)\nmean={euclid.mean():.4f}m  max={euclid.max():.4f}m")
    axes[1,2].set_xlabel("Progress"); axes[1,2].set_ylabel("Distance (m)")
    axes[1,2].legend(fontsize=8); axes[1,2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {tag}.png")

    # txt report
    with open(f"{OUT_DIR}/{tag}_diff.txt", "w") as f:
        f.write(f"Pairwise comparison: {nameA} vs {nameB}\n")
        f.write(f"Matched keyframes : {len(m['prog'])}\n")
        f.write(f"Only in {nameA:<15}: {len(dA['id']) - len(m['prog'])}\n")
        f.write(f"Only in {nameB:<15}: {len(dB['id']) - len(m['prog'])}\n\n")
        f.write(f"Summary:\n")
        f.write(f"  TX   mean_abs={np.abs(tx_diff).mean():.6f}m  max_abs={np.abs(tx_diff).max():.6f}m\n")
        f.write(f"  TZ   mean_abs={np.abs(tz_diff).mean():.6f}m  max_abs={np.abs(tz_diff).max():.6f}m\n")
        f.write(f"  Yaw  mean_abs={np.abs(yaw_diff).mean():.4f}°  max_abs={np.abs(yaw_diff).max():.4f}°\n")
        f.write(f"  Eucl mean={euclid.mean():.6f}m  max={euclid.max():.6f}m\n")
    print(f"Saved: {tag}_diff.txt")


# ════════════════════════════════════════════════════════════════════════════════
# 7–8. Scale ratio scatter: no_scale vs HF, scaled vs HF
# ════════════════════════════════════════════════════════════════════════════════
ratio_pairs = [
    ("run_no_scale", "ratio_no_scale"),
    ("run_scaled",   "ratio_scaled"),
]

hf = datasets["HF original"]

for run_name, out_tag in ratio_pairs:
    run = datasets[run_name]
    c   = COLORS[run_name]
    m   = match(hf, run)
    if m is None:
        print(f"No matches for ratio plot: {run_name}")
        continue

    tx_r, tz_r = compute_ratios(m["txA"], m["tzA"], m["txB"], m["tzB"])

    print(f"\n{run_name}:")
    print(f"  TX ratio  mean={tx_r.mean():.4f}  std={tx_r.std():.4f}  n={len(tx_r)}")
    print(f"  TZ ratio  mean={tz_r.mean():.4f}  std={tz_r.std():.4f}  n={len(tz_r)}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{run_name} — scale ratio vs HF original\n"
                 f"TX: mean={tx_r.mean():.3f}  std={tx_r.std():.3f}  |  "
                 f"TZ: mean={tz_r.mean():.3f}  std={tz_r.std():.3f}", fontsize=12)

    for ax_row, (comp, ratio) in enumerate([("TX", tx_r), ("TZ", tz_r)]):
        orig_vals = m["txA"] if comp == "TX" else m["tzA"]
        run_vals  = m["txB"] if comp == "TX" else m["tzB"]
        mask      = np.abs(run_vals) > MIN_ABS
        ov, rv    = orig_vals[mask], run_vals[mask]
        r_filtered = ov / rv
        r_filtered = r_filtered[(r_filtered > -10) & (r_filtered < 10)]

        # scatter: orig vs run
        ax = axes[ax_row, 0]
        ax.scatter(ov, rv, s=10, alpha=0.6, color=c)
        lims = [min(ov.min(), rv.min()) - 0.3, max(ov.max(), rv.max()) + 0.3]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label="1:1 line")
        ax.plot(np.array(lims), np.array(lims) / ratio.mean(), 'r--', lw=1, alpha=0.7,
                label=f"fitted (÷{ratio.mean():.3f})")
        ax.set_title(f"{comp} orig vs {run_name}"); ax.set_xlabel(f"{comp} HF"); ax.set_ylabel(f"{comp} run")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.axis("equal")

        # ratio vs progress
        prog_mask = np.abs(m["txB"] if comp == "TX" else m["tzB"]) > MIN_ABS
        prog_vals = m["prog"][prog_mask]
        r_prog    = (m["txA"] if comp == "TX" else m["tzA"])[prog_mask] / \
                    (m["txB"] if comp == "TX" else m["tzB"])[prog_mask]
        r_prog    = r_prog[(r_prog > -10) & (r_prog < 10)]
        p_filtered = prog_vals[(
            (m["txA"] if comp == "TX" else m["tzA"])[prog_mask] /
            (m["txB"] if comp == "TX" else m["tzB"])[prog_mask] > -10) &
            ((m["txA"] if comp == "TX" else m["tzA"])[prog_mask] /
             (m["txB"] if comp == "TX" else m["tzB"])[prog_mask] < 10)]

        ax = axes[ax_row, 1]
        ax.scatter(p_filtered, r_prog, s=10, alpha=0.6, color=c,
                   label=f"mean={ratio.mean():.3f}  std={ratio.std():.3f}")
        ax.axhline(ratio.mean(), color="red", lw=1.2, ls="-", label=f"mean={ratio.mean():.3f}")
        ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5, label="ratio=1 (perfect)")
        ax.fill_between([0, 1], ratio.mean() - ratio.std(), ratio.mean() + ratio.std(),
                        alpha=0.15, color="red", label="±1 std")
        ax.set_title(f"{comp} ratio over progress")
        ax.set_xlabel("Trajectory progress (0→1)"); ax.set_ylabel(f"{comp} ratio (HF / run)")
        ax.set_ylim(-3, 5); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{out_tag}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_tag}.png")

print(f"\nAll outputs → {OUT_DIR}/")
