#!/usr/bin/env python3
"""
debug_simulator.py

Run a short simulation of your SpirobEnv, print joint angles per step,
plot them and save the figure to a PNG file for headless environments.
"""
import sys
import numpy as np
import matplotlib
import time

# Use a non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("../utility/")
from Utility import DataCollector  # Ensure correct import path

from Utility import SpirobEnv

# === Configuration ===
xml_path = "../Spirob/2Dspiralrobot/2Dtendon10deg.xml"  # MJCF file path
steps = 20  # number of control steps
sim_steps_per_control = 1  # fewer substeps → finer sampling
output_fig = "joint_angles.png"  # saved plot file
render = True  # toggle MuJoCo window
slowdown = 200.0  # factor to slow visualization

# === Initialize environment ===
env = SpirobEnv(xml_path, sim_steps_per_control=sim_steps_per_control)
# real simulation dt per control step = base timestep * substeps
sim_dt = env.model.opt.timestep * env.sim_steps
# scaled sleep time for visualization
dt_vis = sim_dt * slowdown
print(
    f"Simulation dt per control (substeps={sim_steps_per_control}): {sim_dt:.4f}s, visualizing at dt={dt_vis:.4f}s"
)

# small constant control input
u = np.array([0.05, 0.10])  # safe within [umin, umax]
print(f"Using constant control u={u}")

# reset environment
env.reset()

# === Optional MuJoCo passive viewer ===
viewer = None
if render:
    try:
        from mujoco.viewer import launch_passive

        viewer = launch_passive(env.model, env.data)
        print("Passive viewer launched")
    except ImportError:
        print("Passive viewer not available; continuing headless mode")
        render = False

# === Run simulation ===
q_history = []
for i in range(steps):
    state, _, _, _ = env.step(u)
    q = state[: env.state_dim // 2]
    q_history.append(q)
    print(f"step={i+1:02d}: q = {np.round(q, 3)}")

    if render and viewer is not None:
        viewer.sync(state_only=False)
        time.sleep(dt_vis)

# === Plot joint angles ===
q_array = np.stack(q_history, axis=0)  # shape (steps, 21)
plt.figure(figsize=(8, 5))
for j in range(min(5, q_array.shape[1])):
    plt.plot(q_array[:, j], label=f"J{j+1}")
plt.xlabel("Step")
plt.ylabel("Angle (rad)")
plt.title("Joint Angles Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_fig)
print(f"Plot saved to {output_fig}")
