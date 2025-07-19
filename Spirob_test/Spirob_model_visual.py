import os, sys, time
import numpy as np
import torch
import matplotlib.pyplot as plt
import mujoco
from mujoco import viewer

# Add your code paths
sys.path += ["../utility/", "../Spirob_train/"]
from Utility import SpirobEnv
from Spirob_Learn_Koopman_with_KlinearEig import Network

# --- CONFIGURATION ---
SEGMENTS = 5
STEPS_PER_SEG = 20
JOINTS = [0, 5, 10, 15, 20]  # which joints to plot
XML_PATH = "../Spirob/2Dspiralrobot/2Dtendon10deg.xml"


# --- Load trained Koopman model ---
def load_koopman(path="../Spirob_checkpoints/best_model_v1.pth"):
    ckpt = torch.load(path, map_location="cpu")
    net = Network(
        ckpt["layer"], Nkoopman=ckpt["A"].shape[0], u_dim=ckpt["B"].shape[1]
    ).double()
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net


# --- Input policy u0 < u1 in [0, 0.6] ---
def right_greater_left_policy():
    u0 = np.random.uniform(0.0, 0.55)
    diffs = [0.02, 0.05, 0.1, 0.2, 0.4]
    ps = [0.3, 0.3, 0.2, 0.15, 0.05]
    d = np.random.choice(diffs, p=ps)
    u1 = np.clip(u0 + d, u0 + 0.01, 0.6)
    return np.array([u0, u1])


# --- Simulate both actual and predicted from the same start ---
def run_comparison(net):
    env = SpirobEnv(XML_PATH)
    n_free_q = 7
    n_free_v = 6

    true_q, pred_q = [], []
    all_inputs = []

    for seg in range(SEGMENTS):
        ctrl = right_greater_left_policy()
        print(f"Segment {seg+1}/{SEGMENTS} | ctrl = {ctrl}")
        all_inputs.append(ctrl)

        # Reset env and record initial state
        env.reset()
        env.data.qpos[:n_free_q] = 0
        env.data.qvel[:n_free_v] = 0
        x0 = env.get_state()

        # --- SIMULATE TRUE ---
        q_true = [x0.copy()]
        env.reset()
        env.data.qpos[:n_free_q] = 0
        env.data.qvel[:n_free_v] = 0
        for _ in range(STEPS_PER_SEG):
            env.data.ctrl[:] = ctrl
            mujoco.mj_step(env.model, env.data)
            q_true.append(env.get_state().copy())
        true_q.extend(q_true)

        # --- SIMULATE KOOPMAN ---
        with torch.no_grad():
            Xk = net.encode(torch.tensor(x0, dtype=torch.double)[None])
        q_pred = [x0.copy()]
        for _ in range(STEPS_PER_SEG):
            with torch.no_grad():
                Xk = net.forward(Xk, torch.tensor(ctrl, dtype=torch.double)[None])
            q = Xk[0, : len(x0)].cpu().numpy()
            q_pred.append(q.copy())
        pred_q.extend(q_pred)

    return np.array(true_q), np.array(pred_q), np.array(all_inputs)


# --- Plotting ---
def plot_results(true_q, pred_q):
    T = len(true_q)
    t = np.arange(T)

    plt.figure(figsize=(12, 8))
    for i, joint in enumerate(JOINTS):
        plt.subplot(len(JOINTS), 1, i + 1)
        plt.plot(t, true_q[:, joint], "b-", label="True")
        plt.plot(t, pred_q[:, joint], "r--", label="Predicted")
        for s in range(1, SEGMENTS):
            plt.axvline(s * (STEPS_PER_SEG + 1), color="gray", linestyle="--")
        plt.ylabel(f"q[{joint}]")
        plt.legend()

    plt.suptitle("True vs Predicted Joint Angles per Segment")
    plt.xlabel("Timestep")
    plt.tight_layout()
    plt.show()


# --- Optional: Show predicted motion in viewer ---
def visualize_true_motion(true_q, delay=0.2):
    """Visualize the true robot trajectory in the MuJoCo simulator viewer."""
    env = SpirobEnv(XML_PATH)
    viewer_handle = viewer.launch_passive(env.model, env.data)

    n_free_q = 7
    n_free_v = 6

    try:
        for q in true_q:
            env.data.qpos[n_free_q:] = q[: env.model.nq - n_free_q]
            env.data.qvel[n_free_v:] = q[env.model.nq - n_free_q :]
            mujoco.mj_forward(env.model, env.data)
            viewer_handle.sync()
            time.sleep(delay)
    except KeyboardInterrupt:
        viewer_handle.close()


def visualize_koopman_motion(pred_q, delay=0.2):
    env = SpirobEnv(XML_PATH)
    viewer_handle = viewer.launch_passive(env.model, env.data)

    n_free_q = 7
    n_free_v = 6

    try:
        for q in pred_q:
            env.data.qpos[n_free_q:] = q[: env.model.nq - n_free_q]
            env.data.qvel[n_free_v:] = q[env.model.nq - n_free_q :]
            mujoco.mj_forward(env.model, env.data)
            viewer_handle.sync()
            time.sleep(delay)  # Adjust delay for slower playback
    except KeyboardInterrupt:
        viewer_handle.close()


# --- MAIN ---
if __name__ == "__main__":
    model = load_koopman()
    true_q, pred_q, inputs = run_comparison(model)
    plot_results(true_q, pred_q)
    print("\nVisualizing TRUE robot movement...")
    visualize_true_motion(true_q, delay=0.25)

    print("\nVisualizing KOOPMAN-predicted movement...")
    visualize_koopman_motion(pred_q, delay=0.25)
