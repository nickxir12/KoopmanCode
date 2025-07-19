import os
import sys
import time
import numpy as np
import torch
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

# Add paths
sys.path += ["../utility/", "../Spirob_train/"]
from Utility import SpirobEnv
from Spirob_Learn_Koopman_with_KlinearEig import Network

SEGMENTS = 5
STEPS_PER_SEG = 20
VIEW_DT = 0.1
JOINTS_TO_PLOT = [0, 5, 10, 15, 20]


def load_model():
    ckpt_path = "../Spirob_checkpoints/best_model.pth"
    assert os.path.exists(ckpt_path), f"Model not found at {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    net = Network(
        ckpt["layer"], Nkoopman=ckpt["A"].shape[0], u_dim=ckpt["B"].shape[1]
    ).double()
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net


def right_greater_left_policy():
    u0 = np.random.uniform(0.0, 0.55)
    diff_choices = [0.02, 0.05, 0.1, 0.2, 0.4]
    diff_probs = [0.3, 0.3, 0.2, 0.15, 0.05]
    diff = np.random.choice(diff_choices, p=diff_probs)
    u1 = np.clip(u0 + diff, u0 + 0.01, 0.6)
    return np.array([u0, u1])


def visualize_koopman_predictions(net):
    true_env = SpirobEnv("../Spirob/2Dspiralrobot/2Dtendon10deg.xml")
    koopman_env = SpirobEnv("../Spirob/2Dspiralrobot/2Dtendon10deg.xml")

    n_free_q, n_free_v = 7, 6
    viewer_true = viewer.launch_passive(true_env.model, true_env.data)
    viewer_koop = viewer.launch_passive(koopman_env.model, koopman_env.data)

    true_states, pred_states = [], []

    for seg_idx in range(1, SEGMENTS + 1):
        ctrl = right_greater_left_policy()
        print(f"Segment {seg_idx}/{SEGMENTS}, ctrl={ctrl}")

        # Reset both environments
        true_env.reset()
        koopman_env.reset()

        true_env.data.qpos[:n_free_q] = 0
        true_env.data.qvel[:n_free_v] = 0
        koopman_env.data.qpos[:n_free_q] = 0
        koopman_env.data.qvel[:n_free_v] = 0

        # Initial state encoding
        state0 = true_env.get_state()
        with torch.no_grad():
            Xk = net.encode(torch.tensor(state0, dtype=torch.double)[None])

        true_states.append(state0.copy())
        pred_states.append(state0.copy())

        for step in range(STEPS_PER_SEG):
            # True environment stepping
            true_env.data.ctrl[:] = ctrl
            mujoco.mj_step(true_env.model, true_env.data)
            current_true = true_env.get_state()
            true_states.append(current_true.copy())

            # Koopman prediction
            with torch.no_grad():
                Xk = net.forward(Xk, torch.tensor(ctrl, dtype=torch.double)[None])
            current_pred = Xk[0, : len(state0)].cpu().numpy()
            pred_states.append(current_pred.copy())

            # Update Koopman env state for visualization
            koopman_env.data.qpos[n_free_q:] = current_pred[
                : koopman_env.model.nq - n_free_q
            ]
            koopman_env.data.qvel[n_free_v:] = current_pred[
                koopman_env.model.nq - n_free_q :
            ]
            mujoco.mj_forward(koopman_env.model, koopman_env.data)

            # Sync visualizations
            viewer_true.sync()
            viewer_koop.sync()
            time.sleep(VIEW_DT)

            print(
                f"Step {step}: True={current_true[0]:+.3f}, Pred={current_pred[0]:+.3f}"
            )

    viewer_true.close()
    viewer_koop.close()

    return np.array(true_states), np.array(pred_states)


def plot_results(true_states, pred_states):
    plt.figure(figsize=(12, 8))
    for i, joint in enumerate(JOINTS_TO_PLOT):
        plt.subplot(len(JOINTS_TO_PLOT), 1, i + 1)
        plt.plot(true_states[:, joint], "b-", label="True")
        plt.plot(pred_states[:, joint], "r--", label="Predicted")
        plt.ylabel(f"Joint {joint}")
        plt.legend()
    plt.suptitle("True vs Predicted Joint Positions")
    plt.xlabel("Steps")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    net = load_model()
    true_log, pred_log = visualize_koopman_predictions(net)
    plot_results(true_log, pred_log)
