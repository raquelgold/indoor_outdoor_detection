"""
ratio_and_scale_run_no_scale.py
────────────────────────────────
1. Matches run_no_scale keyframes to HF original by trajectory progress
2. Computes TX and TZ scale ratios  (orig / run_no_scale)
3. Saves a 2×2 ratio plot
4. Writes the scaled cam_pose_estimated.csv → run_scaled/vo_final/
5. Writes keyframe_list.txt              → run_scaled/vo_final/
"""

import numpy as np
import matplotlib.pyplot as plt
import shutil, os
import os
DFPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "direct_360_FPE")

# ── paths ─────────────────────────────────────────────────────────────────────
ORIG_PATH = DFPE_DIR + "/mp3d_fpe_dataset/huggingface/1LXtFkjw3qL/0/vo_final/cam_pose_estimated.csv"
NS_PATH   = DFPE_DIR + "/mp3d_fpe_dataset/run_no_scale/1LXtFkjw3qL/0/vo_final/cam_pose_estimated.csv"
SC_DIR    = DFPE_DIR + "/mp3d_fpe_dataset/run_scaled/1LXtFkjw3qL/0/vo_final"
PLOT_OUT  = DFPE_DIR + "/slam_output/analysis/ratio_run_no_scale.png"

PROGRESS_TOL = 0.005   # within 0.5% of trajectory length to match by time
MIN_ABS      = 0.05    # ignore near-zero poses when computing ratio

os.makedirs(SC_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PLOT_OUT), exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
def load(path):
    data = np.loadtxt(path)
    ids  = data[:, 0]
    prog = (ids - ids.min()) / (ids.max() - ids.min())
    return {"raw": data, "progress": prog,
            "tx": data[:, 1], "tz": data[:, 3]}

orig = load(ORIG_PATH)
ns   = load(NS_PATH)
print(f"HF original  : {len(orig['tx'])} keyframes")
print(f"run_no_scale : {len(ns['tx'])} keyframes")

# ── match by progress ─────────────────────────────────────────────────────────
def match(a, b, tol=PROGRESS_TOL):
    """For each frame in a, find nearest-progress frame in b."""
    ia, ib = [], []
    used = set()
    for i, pa in enumerate(a["progress"]):
        dists = np.abs(b["progress"] - pa)
        j = int(np.argmin(dists))
        if j not in used and dists[j] < tol:
            ia.append(i); ib.append(j); used.add(j)
    ia, ib = np.array(ia), np.array(ib)
    return (a["progress"][ia],
            a["tx"][ia], a["tz"][ia],
            b["tx"][ib], b["tz"][ib])

prog, txO, tzO, txR, tzR = match(orig, ns)
print(f"Matched      : {len(prog)} keyframe pairs")

# ── compute ratios (orig / run_no_scale) ──────────────────────────────────────
def ratios(orig_vals, run_vals):
    mask = np.abs(run_vals) > MIN_ABS
    r    = orig_vals[mask] / run_vals[mask]
    ok   = (r > 0.1) & (r < 10)   # drop sign-flipped / extreme outliers
    return r[ok], mask, ok

tx_ratio, tx_mask, tx_ok = ratios(txO, txR)
tz_ratio, tz_mask, tz_ok = ratios(tzO, tzR)

TX_SCALE = float(tx_ratio.mean())
TZ_SCALE = float(tz_ratio.mean())

print(f"\nScale ratios (orig / run_no_scale):")
print(f"  TX  mean={TX_SCALE:.4f}  std={tx_ratio.std():.4f}  n={len(tx_ratio)}")
print(f"  TZ  mean={TZ_SCALE:.4f}  std={tz_ratio.std():.4f}  n={len(tz_ratio)}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    f"Scale ratio — run_no_scale vs HF original\n"
    f"TX: mean={TX_SCALE:.4f}  std={tx_ratio.std():.4f}  |  "
    f"TZ: mean={TZ_SCALE:.4f}  std={tz_ratio.std():.4f}",
    fontsize=12
)

for row, (label, o_vals, r_vals, ratio, mask, ok) in enumerate([
    ("TX", txO, txR, tx_ratio, tx_mask, tx_ok),
    ("TZ", tzO, tzR, tz_ratio, tz_mask, tz_ok),
]):
    prog_m = prog[mask][ok]

    # left: orig vs run scatter
    ax = axes[row, 0]
    ax.scatter(o_vals[mask][ok], r_vals[mask][ok], s=10, alpha=0.6, color="#E24B4A")
    lim = [min(o_vals[mask][ok].min(), r_vals[mask][ok].min()) - 0.2,
           max(o_vals[mask][ok].max(), r_vals[mask][ok].max()) + 0.2]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.4, label="1:1 (perfect)")
    ax.plot(np.array(lim), np.array(lim) / ratio.mean(), "r--", lw=1,
            label=f"fitted scale ÷{ratio.mean():.3f}")
    ax.set_title(f"{label} — original vs run_no_scale")
    ax.set_xlabel(f"{label} original (m)"); ax.set_ylabel(f"{label} run_no_scale (m)")
    ax.axis("equal"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # right: ratio over trajectory progress
    ax = axes[row, 1]
    ax.scatter(prog_m, ratio, s=10, alpha=0.6, color="#E24B4A",
               label=f"ratio per frame  (n={len(ratio)})")
    ax.axhline(ratio.mean(), color="red", lw=1.5, label=f"mean={ratio.mean():.4f}")
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5, label="ratio=1 (no scale needed)")
    ax.fill_between([0, 1], ratio.mean() - ratio.std(), ratio.mean() + ratio.std(),
                    alpha=0.15, color="red", label=f"±1 std ({ratio.std():.4f})")
    ax.set_title(f"{label} ratio (orig/run) over trajectory")
    ax.set_xlabel("Trajectory progress (0→1)"); ax.set_ylabel("Ratio")
    ax.set_ylim(max(0, ratio.mean() - 4*ratio.std()), ratio.mean() + 4*ratio.std())
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved → {PLOT_OUT}")

# ── write scaled cam_pose_estimated.csv → run_scaled ─────────────────────────
ns_data = ns["raw"].copy()
ns_data[:, 1] *= TX_SCALE   # TX
ns_data[:, 3] *= TZ_SCALE   # TZ

sc_pose_path = os.path.join(SC_DIR, "cam_pose_estimated.csv")
np.savetxt(sc_pose_path, ns_data, fmt="%.10g")
print(f"Scaled poses → {sc_pose_path}")
print(f"  TX before: [{ns['tx'].min():.3f}, {ns['tx'].max():.3f}]  "
      f"after: [{ns_data[:,1].min():.3f}, {ns_data[:,1].max():.3f}]")
print(f"  TZ before: [{ns['tz'].min():.3f}, {ns['tz'].max():.3f}]  "
      f"after: [{ns_data[:,3].min():.3f}, {ns_data[:,3].max():.3f}]")

# ── write keyframe_list.txt → run_scaled ─────────────────────────────────────
kf_ids = ns_data[:, 0].astype(int)
kf_list_path = os.path.join(SC_DIR, "keyframe_list.txt")
with open(kf_list_path, "w") as f:
    for fid in kf_ids:
        if fid != 0:
            f.write(f"{fid}\n")
print(f"Keyframe list → {kf_list_path}  ({len(kf_ids)} entries)")

print("\nDone. run_scaled is ready for DFPE.")
