import matplotlib.pyplot as plt
from spirob_env import SpirobEnv
from Utility import data_collecter
from train.Learn_Koopman_with_KlinearEig import Network  # or import your Network class
import torch
import numpy as np
import time
import mujoco

# --- Load Koopman model ---
model_path = "path/to/best_model.pth"  # UPDATE this path
checkpoint = torch.load(model_path)
A = checkpoint["A"]
B = checkpoint["B"]

# Create network and load weights
u_dim = B.shape[1]
Nkoopman = A.shape[0]
net = Network(checkpoint["layer"], Nkoopman=Nkoopman, u_dim=u_dim)
net.load_state_dict(checkpoint["model"])
net.double()
net.eval()

# --- Setup simulation ---
env = SpirobEnv()
env.render = True
s_t = env.reset()
s_dim = 42  # joint angles + joint velocities (21+21), customize if needed

real_states = []
pred_states = []

# Initial Koopman-encoded state
X_koop = net.encode(torch.DoubleTensor(s_t[:s_dim]).unsqueeze(0))

# Rollout
for t in range(100):
    u_t = np.random.uniform(env.umin, env.umax)
    u_tensor = torch.DoubleTensor(u_t).unsqueeze(0)

    # --- Real simulation ---
    s_t1, _, _, _ = env.step(u_t)
    real_states.append(s_t1[:s_dim])
    env.render()
    time.sleep(0.01)

    # --- Koopman prediction ---
    with torch.no_grad():
        X_koop = net.forward(X_koop, u_tensor)
        pred_q = X_koop[0, :s_dim].detach().cpu().numpy()
        pred_states.append(pred_q)

# --- Convert to numpy arrays ---
real_states = np.array(real_states)
pred_states = np.array(pred_states)

# --- Playback: true vs predicted in simulation ---
print("Visualizing: True (green) → Predicted (red)...")
for q_true, q_pred in zip(real_states, pred_states):
    env.data.qpos[:] = q_true[: env.model.nq]
    mujoco.mj_forward(env.model, env.data)
    time.sleep(0.02)

    env.data.qpos[:] = q_pred[: env.model.nq]
    mujoco.mj_forward(env.model, env.data)
    time.sleep(0.02)

# --- Plotting for first joint ---
plt.figure()
plt.plot(real_states[:, 0], label="True Joint[0]", color="green")
plt.plot(pred_states[:, 0], label="Koopman Joint[0]", color="red", linestyle="--")
plt.xlabel("Time Step")
plt.ylabel("Joint Angle")
plt.title("Koopman Prediction vs Ground Truth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
