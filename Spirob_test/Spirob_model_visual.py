import os
import sys
import time
import numpy as np
import torch
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

# Add paths to your utility modules
sys.path += ["../utility/", "../Spirob_train/"]
from Utility import SpirobEnv
from Spirob_Learn_Koopman_with_KlinearEig import Network

# Configuration
SEGMENTS = 5  # Number of control segments
STEPS_PER_SEG = 20  # Steps per segment
VIEW_DT = 0.1  # Visualization timestep (slower = easier to see)
CTRL_AMPLITUDES = np.linspace(0.4, 1.0, SEGMENTS)  # Control amplitudes in [0,1]
JOINTS_TO_PLOT = [0, 5, 10, 15, 20]  # Joint indices to plot (0-20)


def load_model():
    """Load trained Koopman model from checkpoint"""
    ckpt_path = "../Spirob_checkpoints/best_model.pth"
    assert os.path.exists(ckpt_path), f"Model not found at {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    net = Network(
        ckpt["layer"], Nkoopman=ckpt["A"].shape[0], u_dim=ckpt["B"].shape[1]
    ).double()
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net


def visualize_koopman_predictions(net):
    """Run and visualize predictions for controlled joints only"""
    # Initialize environment
    env = SpirobEnv("../Spirob/2Dspiralrobot/2Dtendon10deg.xml")
    n_free_q = 7  # Number of free joints (qpos[0:7] are root)
    n_free_v = 6  # Number of free velocities (qvel[0:6] are root)

    # Data logging
    true_states, pred_states = [], []

    try:
        # Launch MuJoCo viewer
        viewer_handle = viewer.launch_passive(env.model, env.data)

        for seg_idx, amp in enumerate(CTRL_AMPLITUDES, start=1):
            # Set control signal (maintain 1:0.25 ratio)
            ctrl = np.zeros(env.model.nu)
            ctrl[0] = amp  # Primary control
            ctrl[1] = 0.25 * amp  # Secondary control

            print(
                f"\n--- Segment {seg_idx}/{SEGMENTS}, ctrl = [l:{ctrl[0]:.2f}, r:{ctrl[1]:.2f}]"
            )

            # Reset and lock root position
            env.reset()
            env.data.qpos[:n_free_q] = 0  # Fix root at origin
            env.data.qvel[:n_free_v] = 0  # Zero root velocity

            # Initialize Koopman state
            init_state = env.get_state()
            with torch.no_grad():
                Xk = net.encode(torch.tensor(init_state, dtype=torch.double)[None])

            true_states.append(init_state.copy())
            pred_states.append(init_state.copy())

            for step in range(STEPS_PER_SEG):
                # Step the true simulation
                env.data.ctrl[:] = ctrl
                mujoco.mj_step(env.model, env.data)

                # Get true state (excluding root)
                current_true = env.get_state()
                true_states.append(current_true.copy())

                # Predict with Koopman model
                with torch.no_grad():
                    Xk = net.forward(Xk, torch.tensor(ctrl, dtype=torch.double)[None])
                current_pred = Xk[0, : len(init_state)].cpu().numpy()
                pred_states.append(current_pred.copy())

                # Update visualization (only controlled joints)
                env.data.qpos[n_free_q:] = current_pred[
                    : env.model.nq - n_free_q
                ]  # Positions
                env.data.qvel[n_free_v:] = current_pred[
                    env.model.nq - n_free_q :
                ]  # Velocities
                mujoco.mj_forward(env.model, env.data)

                # Print diagnostics
                step_idx = len(true_states) - 1
                print(
                    f"Step {step_idx:03d} - Joint 0: True={current_true[0]:+.3f}, Pred={current_pred[0]:+.3f}"
                )

                viewer_handle.sync()
                time.sleep(VIEW_DT)

        viewer_handle.close()

    except KeyboardInterrupt:
        viewer_handle.close()

    return np.array(true_states), np.array(pred_states)


def plot_results(true_states, pred_states):
    """Plot comparison between true and predicted joint states"""
    plt.figure(figsize=(12, 8))

    for i, joint in enumerate(JOINTS_TO_PLOT):
        plt.subplot(len(JOINTS_TO_PLOT), 1, i + 1)
        plt.plot(true_states[:, joint], "b-", label="True")
        plt.plot(pred_states[:, joint], "r--", label="Predicted")
        plt.ylabel(f"Joint {joint} Position")
        plt.legend()

    plt.suptitle("True vs Predicted Joint Positions (Root Motion Locked)")
    plt.xlabel("Time Steps")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load trained model
    koopman_net = load_model()

    # Run visualization
    true_log, pred_log = visualize_koopman_predictions(koopman_net)

    # Plot results
    plot_results(true_log, pred_log)
