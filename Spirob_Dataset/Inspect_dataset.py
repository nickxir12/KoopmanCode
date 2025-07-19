import numpy as np

import os
import sys

train_data = np.load("./Spirob_Dataset_25steps_right_gt_left/Spirob_Ktrain_v1.npy")
test_data = np.load("./Spirob_Dataset_25steps_right_gt_left/Spirob_Ktest_v1.npy")

print("Train shape:", train_data.shape)  # (steps+1, trajs, dim)
print("Test shape:", test_data.shape)

steps_plus_1, n_traj, dim = train_data.shape
print(
    f"Each trajectory has {steps_plus_1} steps, {n_traj} trajectories, dimension {dim}"
)

# Print one trajectory
i = 0  # change index to inspect different samples
print(f"\nTrajectory {i} input/state vectors:\n")
for t in range(steps_plus_1):
    print(f"t={t}: {train_data[t, i]}")
