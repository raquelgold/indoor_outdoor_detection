"""
ratio_scatter.py
────────────────
For each run, matches keyframes to the original by nearest trajectory progress,
then computes TX_orig / TX_run and TZ_orig / TZ_run and plots a scatterplot.
Produces:
  1. Combined plot with all runs
  2. Individual plot per run
"""

import numpy as np
import matplotlib.pyplot as plt
import os
DFPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "direct_360_FPE")

# ── CONFIG ────────────────────────────────────────────────────────────────────
RUNS = {
    "Run1": DFPE_DIR + "/slam_output/run_1/keyframe_trajectory.txt",
    "Run2": DFPE_DIR + "/slam_output/run_2/keyframe_trajectory.txt",
    "Run3": DFPE_DIR + "/slam_output/run_3/keyframe_trajectory.txt",
    "Run4": DFPE_DIR + "/slam_output/run_4/keyframe_trajectory.txt",
}
ORIG_PATH    = DFPE_DIR + "/mp3d_fpe_dataset/huggingface/1LXtFkjw3qL/0/vo_final/cam_pose_estimated.csv"
OUT_DIR      = DFPE_DIR + "/slam_output/analysis"
PROGRESS_TOL = 0.005
MIN_ABS      = 0.05
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
COLORS = {"Run1": "#E24B4A", "Run2": "#1D9E75", "Run3": "#BA7517", "Run4": "#9B59B6"}

def load(path):
    data = np.loadtxt(path)
    ids  = data[:, 0]
    progress = (ids - ids.min()) / (ids.max() - ids.min())
    return {"progress": progress, "tx": data[:, 1], "tz": data[:, 3]}

def match_to_orig(orig, run, verbose=True):
    def normalize(arr):
        r = arr.max() - arr.min()
        return (arr - arr.min()) / r if r > 0 else arr * 0

    orig_tx_n = normalize(orig["tx"])
    orig_tz_n = normalize(orig["tz"])
    run_tx_n  = normalize(run["tx"])
    run_tz_n  = normalize(run["tz"])

    matched_io, matched_ir, match_method = [], [], []
    used = set()

    for i, pa in enumerate(orig["progress"]):
        prog_dists = np.abs(run["progress"] - pa)

        j = int(np.argmin(prog_dists))
        if j not in used and prog_dists[j] < PROGRESS_TOL:
            matched_io.append(i); matched_ir.append(j)
            match_method.append("time"); used.add(j)
            continue

        window_mask = prog_dists < 0.15
        candidates = [j for j in np.where(window_mask)[0] if j not in used]
        if len(candidates) == 0:
            continue
        spatial_dists = np.sqrt(
            (orig_tx_n[i] - run_tx_n[candidates])**2 +
            (orig_tz_n[i] - run_tz_n[candidates])**2
        )
        best_idx = int(np.argmin(spatial_dists))
        best_j   = candidates[best_idx]
        if spatial_dists[best_idx] < 0.1:
            matched_io.append(i); matched_ir.append(best_j)
            match_method.append("position"); used.add(best_j)

    io = np.array(matched_io)
    ir = np.array(matched_ir)
    method = np.array(match_method)
    if verbose:
        n_time = np.sum(method == "time")
        n_pos  = np.sum(method == "position")
        print(f"  Matched {len(io)} kf  ({n_time} by time, {n_pos} by position)")
    return orig["progress"][io], orig["tx"][io], orig["tz"][io], run["tx"][ir], run["tz"][ir]

def compute_ratios(txO, tzO, txR, tzR):
    """Return filtered ratios and progress-aligned masks."""
    tx_mask = np.abs(txR) > MIN_ABS
    tx_ratio = txO[tx_mask] / txR[tx_mask]
    tx_ok    = (tx_ratio > -10) & (tx_ratio < 10)
    tx_ratio = tx_ratio[tx_ok]

    tz_mask = np.abs(tzR) > MIN_ABS
    tz_ratio = tzO[tz_mask] / tzR[tz_mask]
    tz_ok    = (tz_ratio > -10) & (tz_ratio < 10)
    tz_ratio = tz_ratio[tz_ok]

    return tx_ratio, tz_ratio, tx_mask, tx_ok, tz_mask, tz_ok

orig = load(ORIG_PATH)

# ════════════════════════════════════════════════════════════════════════════
# 1. COMBINED plot (all runs together)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("TX and TZ ratio: Original / Run  (all runs combined)\n"
             "Tight cluster = consistent scale  |  Scattered = drift dominated", fontsize=12)

for ax_row, (component, ylabel) in enumerate([("tx", "TX ratio (orig/run)"),
                                               ("tz", "TZ ratio (orig/run)")]):
    for ax_col, use_progress in enumerate([False, True]):
        ax = axes[ax_row, ax_col]
        ax.set_title(f"{ylabel} — {'vs progress' if use_progress else 'scatterplot'}")

        for name, path in RUNS.items():
            run = load(path)
            prog, txO, tzO, txR, tzR = match_to_orig(orig, run, verbose=False)
            tx_ratio, tz_ratio, tx_mask, tx_ok, tz_mask, tz_ok = compute_ratios(txO, tzO, txR, tzR)

            orig_vals = txO if component == "tx" else tzO
            run_vals  = txR if component == "tx" else tzR
            ratio     = tx_ratio if component == "tx" else tz_ratio
            mask      = tx_mask  if component == "tx" else tz_mask
            ok        = tx_ok    if component == "tx" else tz_ok
            prog_m    = prog[mask][ok]

            c     = COLORS.get(name, "gray")
            label = f"{name}  n={len(ratio)}  mean={ratio.mean():.3f}  std={ratio.std():.3f}"

            if use_progress:
                ax.scatter(prog_m, ratio, s=8, alpha=0.5, color=c, label=label)
                ax.set_xlabel("Trajectory progress (0→1)")
                ax.set_ylabel("Ratio")
                ax.set_ylim(-3, 5)
            else:
                ax.scatter(orig_vals[mask][ok], run_vals[mask][ok],
                           s=8, alpha=0.5, color=c, label=label)
                ax.set_xlabel(f"{component.upper()} original")
                ax.set_ylabel(f"{component.upper()} run")
                lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                        max(ax.get_xlim()[1], ax.get_ylim()[1])]
                ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.4, label="1:1 line")

        if use_progress:
            ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.5, label="ratio=1")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ratio_scatter_combined.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ratio_scatter_combined.png")

# ════════════════════════════════════════════════════════════════════════════
# 2. INDIVIDUAL plots per run
# ════════════════════════════════════════════════════════════════════════════
print("\n── Per-run ratio summary ──")
print(f"{'Run':<8}  {'TX mean':>10}  {'TX std':>8}  {'TZ mean':>10}  {'TZ std':>8}  {'n':>6}")
print("-" * 58)

for name, path in RUNS.items():
    run = load(path)
    print(f"\n{name}:")
    prog, txO, tzO, txR, tzR = match_to_orig(orig, run, verbose=True)
    tx_ratio, tz_ratio, tx_mask, tx_ok, tz_mask, tz_ok = compute_ratios(txO, tzO, txR, tzR)

    print(f"  TX  mean={tx_ratio.mean():.4f}  std={tx_ratio.std():.4f}  n={len(tx_ratio)}")
    print(f"  TZ  mean={tz_ratio.mean():.4f}  std={tz_ratio.std():.4f}  n={len(tz_ratio)}")

    c = COLORS.get(name, "gray")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{name} — TX and TZ ratio vs Original\n"
                 f"TX: mean={tx_ratio.mean():.3f}  std={tx_ratio.std():.3f}  |  "
                 f"TZ: mean={tz_ratio.mean():.3f}  std={tz_ratio.std():.3f}", fontsize=12)

    for ax_row, (component, ratio, mask, ok) in enumerate([
        ("tx", tx_ratio, tx_mask, tx_ok),
        ("tz", tz_ratio, tz_mask, tz_ok),
    ]):
        orig_vals = txO if component == "tx" else tzO
        run_vals  = txR if component == "tx" else tzR
        prog_m    = prog[mask][ok]

        # scatterplot: orig vs run
        ax = axes[ax_row, 0]
        ax.scatter(orig_vals[mask][ok], run_vals[mask][ok],
                   s=10, alpha=0.6, color=c)
        lims = [min(orig_vals[mask][ok].min(), run_vals[mask][ok].min()) - 0.5,
                max(orig_vals[mask][ok].max(), run_vals[mask][ok].max()) + 0.5]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label="1:1 line")
        # scale line: orig = mean_ratio * run
        x_line = np.array(lims)
        ax.plot(x_line, x_line / ratio.mean(), 'r--', lw=1, alpha=0.7,
                label=f"fitted scale (÷{ratio.mean():.3f})")
        ax.set_title(f"{component.upper()} orig vs run")
        ax.set_xlabel(f"{component.upper()} original")
        ax.set_ylabel(f"{component.upper()} run")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.axis('equal')

        # ratio vs progress
        ax = axes[ax_row, 1]
        ax.scatter(prog_m, ratio, s=10, alpha=0.6, color=c,
                   label=f"mean={ratio.mean():.3f}  std={ratio.std():.3f}")
        ax.axhline(ratio.mean(), color='red', lw=1.2, ls='-',
                   label=f"mean={ratio.mean():.3f}")
        ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.5, label="ratio=1 (perfect)")
        ax.fill_between([0, 1],
                        ratio.mean() - ratio.std(),
                        ratio.mean() + ratio.std(),
                        alpha=0.15, color='red', label="±1 std")
        ax.set_title(f"{component.upper()} ratio over trajectory progress")
        ax.set_xlabel("Trajectory progress (0→1)")
        ax.set_ylabel(f"{component.upper()} ratio (orig/run)")
        ax.set_ylim(-3, 5); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = f"{OUT_DIR}/ratio_scatter_{name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: ratio_scatter_{name}.png")