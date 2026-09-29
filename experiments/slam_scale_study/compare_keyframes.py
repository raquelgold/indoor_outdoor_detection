import numpy as np
import matplotlib.pyplot as plt
import os
DFPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "direct_360_FPE")

RUN1 = DFPE_DIR + '/slam_output/run_1/keyframe_trajectory.txt'
RUN2 = DFPE_DIR + '/slam_output/run_2/keyframe_trajectory.txt'

def load_keyframes(path):
    frames = {}
    with open(path) as f:
        for line in f:
            vals = list(map(float, line.split()))
            fid = int(vals[0])
            tx, ty, tz = vals[1], vals[2], vals[3]
            frames[fid] = (tx, ty, tz)
    return frames

kf1 = load_keyframes(RUN1)
kf2 = load_keyframes(RUN2)

ids1 = set(kf1.keys())
ids2 = set(kf2.keys())

in_both   = ids1 & ids2
only_run1 = ids1 - ids2
only_run2 = ids2 - ids1

print(f"Run 1 keyframes:       {len(ids1)}")
print(f"Run 2 keyframes:       {len(ids2)}")
print(f"In both runs:          {len(in_both)}")
print(f"Only in run 1:         {len(only_run1)}  {sorted(only_run1)}")
print(f"Only in run 2:         {len(only_run2)}  {sorted(only_run2)}")

# For frames in both runs, compare their poses
print("\n--- Pose differences for shared keyframes (tx, tz) ---")
diffs = []
for fid in sorted(in_both):
    tx1, ty1, tz1 = kf1[fid]
    tx2, ty2, tz2 = kf2[fid]
    d = np.sqrt((tx1-tx2)**2 + (tz1-tz2)**2)
    diffs.append((fid, d))

diffs_arr = np.array([d for _, d in diffs])
print(f"Mean pose diff (tx,tz): {diffs_arr.mean():.6f}")
print(f"Max pose diff (tx,tz):  {diffs_arr.max():.6f}  at frame {diffs[np.argmax(diffs_arr)][0]}")
print(f"Frames with diff > 0.01: {sum(diffs_arr > 0.01)}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: which frame IDs were selected
axes[0].scatter(sorted(ids1), [1]*len(ids1), s=2, alpha=0.5, label=f'Run 1 ({len(ids1)} kf)', color='blue')
axes[0].scatter(sorted(ids2), [2]*len(ids2), s=2, alpha=0.5, label=f'Run 2 ({len(ids2)} kf)', color='red')
axes[0].scatter(sorted(only_run1), [1]*len(only_run1), s=10, color='blue', marker='x', label=f'Only run 1 ({len(only_run1)})')
axes[0].scatter(sorted(only_run2), [2]*len(only_run2), s=10, color='red', marker='x', label=f'Only run 2 ({len(only_run2)})')
axes[0].set_xlabel('Frame ID')
axes[0].set_yticks([1, 2])
axes[0].set_yticklabels(['Run 1', 'Run 2'])
axes[0].set_title('Keyframe selection per run')
axes[0].legend(markerscale=3)

# Right: pose difference for shared keyframes
fids_shared = [fid for fid, _ in diffs]
axes[1].plot(fids_shared, diffs_arr, 'k-', linewidth=0.7)
axes[1].set_xlabel('Frame ID')
axes[1].set_ylabel('Euclidean distance (tx, tz)')
axes[1].set_title('Pose difference for shared keyframes')
axes[1].axhline(0.01, color='r', linestyle='--', linewidth=0.8, label='0.01 threshold')
axes[1].legend()

plt.tight_layout()
out = DFPE_DIR + '/slam_output/keyframe_comparison.png'
plt.savefig(out, dpi=150)
print(f'\nSaved to {out}')