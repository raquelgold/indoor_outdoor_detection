import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os
DFPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "direct_360_FPE")

df = pd.read_csv(DFPE_DIR + '/mp3d_fpe_dataset/my_run_fps30/1LXtFkjw3qL/0/vo_final/cam_pose_estimated.csv', sep=' ', header=None,
                 names=['t','tx','ty','tz','qx','qy','qz','qw'])

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('cam_pose_estimated.csv', fontsize=14)

# 1. XY top-down path
ax = axes[0,0]
sc = ax.scatter(df['tx'], df['tz'], c=df['t'], cmap='plasma', s=20)
ax.plot(df['tx'], df['tz'], alpha=0.3, color='gray')
plt.colorbar(sc, ax=ax, label='time (s)')
ax.set_title('Camera Path (top-down)')
ax.set_xlabel('TX (m)'); ax.set_ylabel('TZ (m)')

# 2. Position vs time
ax = axes[0,1]
for col, color in zip(['tx','ty','tz'], ['r','g','b']):
    ax.plot(df['t'], df[col], label=col, color=color)
ax.set_title('Position vs Time')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Position (m)')
ax.legend()

# 3. Quaternion vs time
ax = axes[1,0]
for col, color in zip(['qx','qy','qz','qw'], ['r','g','b','orange']):
    ax.plot(df['t'], df[col], label=col, color=color)
ax.set_title('Quaternion vs Time')
ax.set_xlabel('Time (s)')
ax.legend()

# 4. Yaw vs time
ax = axes[1,1]
r = Rotation.from_quat(df[['qx','qy','qz','qw']].values)
yaw = r.as_euler('yxz', degrees=True)[:, 0]
ax.plot(df['t'], yaw, color='purple')
ax.set_title('Yaw vs Time')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Yaw (degrees)')

plt.tight_layout()
plt.savefig(DFPE_DIR + '/cam_pose_plot.png', dpi=150)
plt.show()