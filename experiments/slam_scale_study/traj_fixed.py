import numpy as np
import matplotlib.pyplot as plt
import os
DFPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "direct_360_FPE")

ORIG_PATH   = DFPE_DIR + '/mp3d_fpe_dataset/huggingface/1LXtFkjw3qL/0/vo_final/cam_pose_estimated.csv'
RUN_PATH    = DFPE_DIR + '/slam_output/run_4/keyframe_trajectory.txt'
OUTPUT_PATH = DFPE_DIR + '/slam_output/run_4/traj_scaled.png'

# Mean ratios from ratio_scatter.py results for Run4
TX_SCALE = 1.1644
TZ_SCALE = 1.2500

def load_trajectory(path):
    poses = []
    with open(path) as f:
        for line in f:
            vals = list(map(float, line.split()))
            poses.append(vals[1:4])  # tx, ty, tz
    return np.array(poses)

orig = load_trajectory(ORIG_PATH)
run  = load_trajectory(RUN_PATH)

# Apply scale correction: multiply TX and TZ by their respective ratios
run_scaled = run.copy()
run_scaled[:, 0] *= TX_SCALE   # TX
run_scaled[:, 2] *= TZ_SCALE   # TZ

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Scale correction: TX×{TX_SCALE}  TZ×{TZ_SCALE}', fontsize=12)

# Left: raw unscaled
axes[0].plot(orig[:,0], orig[:,2], 'b-', alpha=0.6, label=f'Original ({len(orig)} kf)')
axes[0].plot(run[:,0],  run[:,2],  'r-', alpha=0.6, label=f'Run4 raw ({len(run)} kf)')
axes[0].set_title('Raw (no correction)')
axes[0].set_xlabel('TX'); axes[0].set_ylabel('TZ')
axes[0].legend(); axes[0].axis('equal'); axes[0].grid(alpha=0.3)

# Middle: scaled
axes[1].plot(orig[:,0],       orig[:,2],       'b-', alpha=0.6, label=f'Original ({len(orig)} kf)')
axes[1].plot(run_scaled[:,0], run_scaled[:,2], 'r-', alpha=0.6, label=f'Run4 scaled ({len(run)} kf)')
axes[1].set_title('After scale correction')
axes[1].set_xlabel('TX'); axes[1].set_ylabel('TZ')
axes[1].legend(); axes[1].axis('equal'); axes[1].grid(alpha=0.3)



plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f'Saved to {OUTPUT_PATH}')