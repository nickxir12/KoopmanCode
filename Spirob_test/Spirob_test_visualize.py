"""
Visualise real vs. Koopman-predicted SpiRob motion
==================================================
• Left  window : full MuJoCo   (env_true)
• Right window : Koopman model (env_pred)

Both environments are forcibly reset to **all–zero joint states** before
the demo and again before each input-segment.
"""

import os, sys, time, numpy as np, torch, mujoco
from mujoco import viewer

# ----------------------------------------------------------------------
#  local imports
# ----------------------------------------------------------------------
sys.path += ["../utility/", "../Spirob_train/"]

from Utility import SpirobEnv
from Spirob_Learn_Koopman_with_KlinearEig import Network

# ----------------------------------------------------------------------
#  (1) load Koopman checkpoint
# ----------------------------------------------------------------------
ckpt_path = "../Spirob_checkpoints/best_model.pth"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(ckpt_path)

ckpt = torch.load(ckpt_path, map_location="cpu")
net = Network(
    ckpt["layer"], Nkoopman=ckpt["A"].shape[0], u_dim=ckpt["B"].shape[1]
).double()
net.load_state_dict(ckpt["model"])
net.eval()

# ----------------------------------------------------------------------
#  (2) build two identical environments
# ----------------------------------------------------------------------
env_true, env_pred = SpirobEnv(), SpirobEnv()


# ------------ helper: put BOTH envs in an all-zero state -------------
def reset_both_to_zero():
    """Set every *joint* angle/vel to zero; base @ (0,0,0) w/ unit quat."""
    for env in (env_true, env_pred):
        env.reset()  # make sure mj_reset is called
        env.data.qpos[:] = 0.0  # xyz=0, quat=(1,0,0,0), joints=0
        env.data.qpos[3] = 1.0  # ensure quaternion w=1
        env.data.qvel[:] = 0.0  # all velocities zero
        mujoco.mj_forward(env.model, env.data)


# first global reset
reset_both_to_zero()

# ---------------- dimensions we’ll need repeatedly --------------------
n_free_q, n_free_v = 7, 6
n_act_q = env_true.model.nq - n_free_q  # 21
n_act_v = env_true.model.nv - n_free_v  # 21
s_dim = n_act_q + n_act_v  # 42  (angles + vels)

# ----------------------------------------------------------------------
#  (3) launch (passive) viewers
# ----------------------------------------------------------------------
v_true = viewer.launch_passive(env_true.model, env_true.data)
v_pred = viewer.launch_passive(env_pred.model, env_pred.data)

print("Viewers open – will start moving after 5 s …", flush=True)
time.sleep(5.0)

# ----------------------------------------------------------------------
#  (4) run several input-segments with different magnitudes
# ----------------------------------------------------------------------
SEGMENTS = 5
STEPS_PER_SEG = 20
DT_VIEW = 0.25  # seconds between frames  (~4 Hz)

ctrl_levels = np.linspace(0.05, 0.25, SEGMENTS)  # 0.05 … 0.25
true_log, pred_log = [], []  # for plotting
step_global = 0

try:
    for seg, amp in enumerate(ctrl_levels, 1):
        print(f"\n► Segment {seg}/{SEGMENTS}  ctrl=[{amp:.2f}, {amp*0.25:.2f}]")
        ctrl = np.array([amp, amp * 0.25], dtype=np.float64)

        # -------------- RESET BOTH ENVS TO ZERO ----------------------
        reset_both_to_zero()

        # ---- encode fresh Koopman state (strictly joint q+v only) ---
        state0 = np.hstack(
            [
                env_true.data.qpos[n_free_q:],  # 21 joint angles
                env_true.data.qvel[n_free_v:],  # 21 joint vels
            ]
        )
        Xk = net.encode(torch.tensor(state0, dtype=torch.double)[None])

        for _ in range(STEPS_PER_SEG):
            # -------- TRUE physics -------------
            env_true.data.ctrl[:] = ctrl
            mujoco.mj_step(env_true.model, env_true.data)

            cur_true = np.hstack(
                [env_true.data.qpos[n_free_q:], env_true.data.qvel[n_free_v:]]
            )
            true_log.append(cur_true)

            # -------- KOOPMAN one-step --------
            with torch.no_grad():
                Xk = net.forward(Xk, torch.tensor(ctrl, dtype=torch.double)[None])
            pred_state = Xk[0, :s_dim].cpu().numpy()
            pred_log.append(pred_state)

            # push into second sim & forward kinematics
            env_pred.data.qpos[n_free_q:] = pred_state[:n_act_q]
            env_pred.data.qvel[n_free_v:] = pred_state[n_act_q:]
            mujoco.mj_forward(env_pred.model, env_pred.data)

            # console debug
            err = abs(cur_true[0] - pred_state[0])
            print(
                f"step={step_global:03d}  "
                f"q0 true={cur_true[0]:+.4f}  pred={pred_state[0]:+.4f}  "
                f"|err|={err:.4f}",
                flush=True,
            )
            step_global += 1

            # refresh viewers
            v_true.sync()
            v_pred.sync()
            time.sleep(DT_VIEW)

finally:
    v_true.close()
    v_pred.close()
    print("\nViewer windows closed.")

# ----------------------------------------------------------------------
#  (5) error summary & plot
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt

true_log = np.asarray(true_log)
pred_log = np.asarray(pred_log)
mae = np.mean(np.abs(true_log - pred_log), axis=0)
print(f"\nMean-abs-error joint-0 over {step_global} steps : {mae[0]:.4f} rad")

plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.plot(true_log[:, 0], label="true q0", c="green")
plt.plot(pred_log[:, 0], label="pred q0", c="red", ls="--")
plt.title("Joint-0   true vs. Koopman")
plt.xlabel("step")
plt.ylabel("angle (rad)")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.abs(true_log[:, 0] - pred_log[:, 0]), c="purple")
plt.title("|error| joint-0")
plt.xlabel("step")
plt.grid()
plt.tight_layout()
plt.show()
