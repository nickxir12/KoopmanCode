# Visualise real vs. Koopman-predicted SpiRob motion
# ==================================================
# Left  window : full MuJoCo   (env_true)
# Right window: Koopman model (env_pred)

import os, sys, time
import numpy as np
import torch
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

sys.path += ["../utility/", "../Spirob_train/"]
from Utility import SpirobEnv
from Spirob_Learn_Koopman_with_KlinearEig import Network

# Load Koopman model
ckpt_path = "../Spirob_checkpoints/best_model.pth"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(ckpt_path)
ckpt = torch.load(ckpt_path, map_location="cpu")
net = Network(
    ckpt["layer"], Nkoopman=ckpt["A"].shape[0], u_dim=ckpt["B"].shape[1]
).double()
net.load_state_dict(ckpt["model"])
net.eval()

# Create environments
env_true = SpirobEnv()
env_pred = SpirobEnv()
STEPS_PER_CTRL = env_true.sim_steps_per_control

n_free_q, n_free_v = 7, 6
n_act_q = env_true.model.nq - n_free_q
n_act_v = env_true.model.nv - n_free_v
s_dim = n_act_q + n_act_v


def zero_reset(env):
    env.reset()
    env.data.qpos[:] = 0.0
    env.data.qpos[3] = 1.0
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


zero_reset(env_true)
zero_reset(env_pred)

# Launch viewer windows
v_true = viewer.launch_passive(env_true.model, env_true.data)
v_pred = viewer.launch_passive(env_pred.model, env_pred.data)
print("Viewers open — holding zero pose for 3s …", flush=True)
time.sleep(3.0)

# Run 5 control segments
SEGMENTS = 5
STEPS_PER_SEG = 20
VIEW_DT = 0.15
CTRL_AMPLITUDES = np.linspace(0, 1, SEGMENTS)

true_log, pred_log = [], []
step_counter = 0

try:
    for seg_idx, amp in enumerate(CTRL_AMPLITUDES, start=1):
        ctrl = np.zeros(env_true.model.nu)
        ctrl[0] = amp
        ctrl[1] = 0.25 * amp
        print(
            f"\n--- Segment {seg_idx}/{SEGMENTS}, ctrl = [l:{ctrl[0]:.2f}, r:{ctrl[1]:.2f}]",
            flush=True,
        )

        zero_reset(env_true)
        zero_reset(env_pred)

        init_state = np.hstack(
            [env_true.data.qpos[n_free_q:], env_true.data.qvel[n_free_v:]]
        )
        Xk = net.encode(torch.tensor(init_state, dtype=torch.double)[None])
        true_log.append(init_state.copy())
        pred_log.append(init_state.copy())

        for _ in range(STEPS_PER_SEG):
            env_true.data.ctrl[:] = ctrl
            for _ in range(STEPS_PER_CTRL):
                mujoco.mj_step(env_true.model, env_true.data)

            cur_true = np.hstack(
                [env_true.data.qpos[n_free_q:], env_true.data.qvel[n_free_v:]]
            )
            true_log.append(cur_true)

            with torch.no_grad():
                Xk = net.forward(Xk, torch.tensor(ctrl, dtype=torch.double)[None])
            cur_pred = Xk[0, :s_dim].cpu().numpy()
            pred_log.append(cur_pred)

            # sync base and prediction into env_pred
            env_pred.data.qpos[:n_free_q] = env_true.data.qpos[:n_free_q]
            env_pred.data.qvel[:n_free_v] = env_true.data.qvel[:n_free_v]
            env_pred.data.qpos[n_free_q:] = cur_pred[:n_act_q]
            env_pred.data.qvel[n_free_v:] = cur_pred[n_act_q:]
            mujoco.mj_forward(env_pred.model, env_pred.data)

            # ---- Informative print
            j0_true, j0_pred = cur_true[0], cur_pred[0]
            j10_true, j10_pred = cur_true[10], cur_pred[10]
            j20_true, j20_pred = cur_true[20], cur_pred[20]
            err0 = abs(j0_true - j0_pred)
            err10 = abs(j10_true - j10_pred)
            err20 = abs(j20_true - j20_pred)

            print(
                f"step={step_counter:03d}  "
                f"q0: {j0_true:+.3f}/{j0_pred:+.3f} err={err0:.3f} |  "
                f"q10: {j10_true:+.3f}/{j10_pred:+.3f} err={err10:.3f} |  "
                f"q20: {j20_true:+.3f}/{j20_pred:+.3f} err={err20:.3f}",
                flush=True,
            )
            step_counter += 1

            v_true.sync()
            v_pred.sync()
            time.sleep(VIEW_DT)

finally:
    v_true.close()
    v_pred.close()
    print("\nViewer windows closed.")

# Summary & Plots
true_log = np.array(true_log)
pred_log = np.array(pred_log)
mae = np.mean(np.abs(true_log - pred_log), axis=0)

print("\n====== Mean-Abs Errors per Joint ======")
for j in [0, 10, 20]:
    print(f"Joint-{j:<2d}   MAE = {mae[j]:.4f} rad")
print("=======================================\n")

plt.figure(figsize=(12, 4))
for i, j in enumerate([0, 10, 20]):
    plt.subplot(1, 3, i + 1)
    plt.plot(true_log[:, j], label=f"true q{j}", c="green")
    plt.plot(pred_log[:, j], label=f"pred q{j}", c="red", ls="--")
    plt.title(f"Joint-{j}")
    plt.xlabel("step")
    plt.ylabel("angle (rad)")
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()
