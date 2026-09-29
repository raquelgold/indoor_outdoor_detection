import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os
DFPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "direct_360_FPE")

ORIG_PATH   = DFPE_DIR + '/mp3d_fpe_dataset/huggingface/1LXtFkjw3qL/0/vo_final/cam_pose_estimated.csv'
FRESH_PATH  = DFPE_DIR + '/slam_output/run_fresh/keyframe_trajectory.txt'
OUTPUT_PATH = DFPE_DIR + '/slam_output/run_fresh/traj_compare.png'


def load_trajectory(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = list(map(float, line.split()))
            if len(vals) < 8:
                continue
            ts = vals[0]
            tx, ty, tz = vals[1], vals[2], vals[3]
            qx, qy, qz, qw = vals[4], vals[5], vals[6], vals[7]
            yaw = Rotation.from_quat([qx, qy, qz, qw]).as_euler('yxz', degrees=True)[0]
            rows.append([ts, tx, ty, tz, yaw])
    data = np.array(rows)
    progress = (data[:, 0] - data[0, 0]) / (data[-1, 0] - data[0, 0])
    return data, progress


orig,  orig_prog  = load_trajectory(ORIG_PATH)
fresh, fresh_prog = load_trajectory(FRESH_PATH)

tx_o, tz_o, yaw_o = orig[:,1],  orig[:,3],  orig[:,4]
tx_f, tz_f, yaw_f = fresh[:,1], fresh[:,3], fresh[:,4]

# Center both
tx_oc, tz_oc = tx_o - tx_o.mean(), tz_o - tz_o.mean()
tx_fc, tz_fc = tx_f - tx_f.mean(), tz_f - tz_f.mean()

# Translation magnitude
mag_o = np.sqrt(tx_o**2 + tz_o**2)
mag_f = np.sqrt(tx_f**2 + tz_f**2)

# Global shift (start-to-start offset)
shift_tx = tx_f[0] - tx_o[0]
shift_tz = tz_f[0] - tz_o[0]

# Scale ratio (extent comparison)
range_tx_o = tx_o.max() - tx_o.min()
range_tx_f = tx_f.max() - tx_f.min()
range_tz_o = tz_o.max() - tz_o.min()
range_tz_f = tz_f.max() - tz_f.min()

print(f"Original  : {len(orig)} kf  |  TX [{tx_o.min():.2f}, {tx_o.max():.2f}]  TZ [{tz_o.min():.2f}, {tz_o.max():.2f}]")
print(f"Fresh run : {len(fresh)} kf  |  TX [{tx_f.min():.2f}, {tx_f.max():.2f}]  TZ [{tz_f.min():.2f}, {tz_f.max():.2f}]")
print(f"Start offset : dTX={shift_tx:.4f}  dTZ={shift_tz:.4f}")
print(f"TX extent ratio (fresh/orig) : {range_tx_f/range_tx_o:.4f}")
print(f"TZ extent ratio (fresh/orig) : {range_tz_f/range_tz_o:.4f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"Trajectory comparison — HF original ({len(orig)} kf) vs Fresh run ({len(fresh)} kf)", fontsize=13)

# ── Row 0: spatial plots ──────────────────────────────────────────────────────

# Raw top-down
axes[0,0].plot(tx_o, tz_o, 'b-', lw=0.8, alpha=0.7, label='HF original')
axes[0,0].plot(tx_f, tz_f, 'r-', lw=0.8, alpha=0.7, label='Fresh run')
axes[0,0].set_title('Top-down (raw)')
axes[0,0].set_xlabel('TX'); axes[0,0].set_ylabel('TZ')
axes[0,0].axis('equal'); axes[0,0].grid(alpha=0.3); axes[0,0].legend()

# Centered top-down
axes[0,1].plot(tx_oc, tz_oc, 'b-', lw=0.8, alpha=0.7, label='HF original')
axes[0,1].plot(tx_fc, tz_fc, 'r-', lw=0.8, alpha=0.7, label='Fresh run')
axes[0,1].set_title('Top-down (centered)')
axes[0,1].set_xlabel('TX'); axes[0,1].set_ylabel('TZ')
axes[0,1].axis('equal'); axes[0,1].grid(alpha=0.3); axes[0,1].legend()

# Translation magnitude over progress
axes[0,2].plot(orig_prog,  mag_o, 'b-', lw=0.8, alpha=0.7, label='HF original')
axes[0,2].plot(fresh_prog, mag_f, 'r-', lw=0.8, alpha=0.7, label='Fresh run')
axes[0,2].set_title('Translation magnitude ||(TX,TZ)||')
axes[0,2].set_xlabel('Trajectory progress'); axes[0,2].set_ylabel('metres')
axes[0,2].grid(alpha=0.3); axes[0,2].legend()

# ── Row 1: per-axis over time ─────────────────────────────────────────────────

axes[1,0].plot(orig_prog,  tx_o, 'b-', lw=0.8, alpha=0.7, label='HF original')
axes[1,0].plot(fresh_prog, tx_f, 'r-', lw=0.8, alpha=0.7, label='Fresh run')
axes[1,0].set_title('TX over time')
axes[1,0].set_xlabel('Trajectory progress'); axes[1,0].set_ylabel('TX (m)')
axes[1,0].grid(alpha=0.3); axes[1,0].legend()

axes[1,1].plot(orig_prog,  tz_o, 'b-', lw=0.8, alpha=0.7, label='HF original')
axes[1,1].plot(fresh_prog, tz_f, 'r-', lw=0.8, alpha=0.7, label='Fresh run')
axes[1,1].set_title('TZ over time')
axes[1,1].set_xlabel('Trajectory progress'); axes[1,1].set_ylabel('TZ (m)')
axes[1,1].grid(alpha=0.3); axes[1,1].legend()

axes[1,2].plot(orig_prog,  yaw_o, 'b-', lw=0.8, alpha=0.7, label='HF original')
axes[1,2].plot(fresh_prog, yaw_f, 'r-', lw=0.8, alpha=0.7, label='Fresh run')
axes[1,2].set_title('Yaw over time')
axes[1,2].set_xlabel('Trajectory progress'); axes[1,2].set_ylabel('Yaw (°)')
axes[1,2].grid(alpha=0.3); axes[1,2].legend()

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f'\nSaved → {OUTPUT_PATH}')
