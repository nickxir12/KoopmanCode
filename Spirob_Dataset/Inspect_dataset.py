import numpy as np
import os

# Load dataset
train_data = np.load("./Spirob_Dataset_32steps_right_gt_left/Spirob_Ktrain_v1.npy")
test_data = np.load("./Spirob_Dataset_32steps_right_gt_left/Spirob_Ktest_v1.npy")

print("Train shape:", train_data.shape)  # (steps+1, trajs, dim)
print("Test shape:", test_data.shape)

steps_plus_1, n_traj, dim = train_data.shape
u_dim = 2
state_dim = dim - u_dim
assert state_dim == 42, "Expected 21 angles + 21 speeds"

print(
    f"\nEach trajectory has {steps_plus_1} steps, {n_traj} trajectories, dimension {dim}"
)
print(f"Control dim: {u_dim}, State dim: {state_dim} (21 angles + 21 speeds)")

# Choose trajectory to inspect
traj_idx = 0


print(f"\n🧪 Inspecting trajectory {traj_idx} step-by-step (angles in degrees):\n")
for t in range(steps_plus_1):
    sample = train_data[t, traj_idx]  # shape = (u_dim + 42,)
    u = sample[:u_dim]
    q_rad = sample[u_dim : u_dim + 21]
    q_deg = np.rad2deg(q_rad)
    qdot = sample[u_dim + 21 :]

    assert np.all(q_deg >= -30.1) and np.all(q_deg <= 30.1), "🚨 Angle out of bounds!"

    print(f"--- t = {t} ---")
    print("Controls (u)         :", np.round(u, 4))
    print("Joint angles (deg)   :", np.round(q_deg, 2))
    print("Joint speeds (rad/s) :", np.round(qdot, 4))
    print()
