import numpy as np
import os
import sys

sys.path.append("../utility/")
from Utility import DataCollector  # Ensure correct import path


def small_constant_policy(_):
    """
    Ignores its input and always returns [u0, u1] = [0.05, 0.1].
    Both are ≤ 0.6 and u1 > u0.
    """
    return np.array([0.05, 0.10])


def right_greater_left_policy(_):
    """
    Returns a constant input [u0, u1] for an entire trajectory,
    where u1 > u0 and both are in [0, 0.6].
    The difference (u1 - u0) is sampled probabilistically with variation.
    """
    # Sample base u0 in [0, 0.2] to allow some headroom
    u0 = np.random.uniform(0.0, 0.2)

    # Sample the difference with bias: small diff is more likely, but allow larger diffs
    diff_choices = [0.02, 0.05, 0.1, 0.2, 0.4]
    diff_probs = [0.3, 0.3, 0.2, 0.15, 0.05]  # More likely to get small u1-u0

    diff = np.random.choice(diff_choices, p=diff_probs)

    u1 = np.clip(u0 + diff, u0 + 0.01, 0.6)  # ensure u1 > u0 and ≤ 0.6

    return np.array([u0, u1])


def save_koopman_dataset(
    xml_path,
    save_dir,
    Ktrain,
    Ktest,
    Ksteps,
    sim_steps_per_control=1,
    input_policy=None,
):
    os.makedirs(save_dir, exist_ok=True)
    collector = DataCollector(xml_path, sim_steps_per_control)
    # collect 200 train trajs, 20 steps each, using the small constant policy
    train = collector.collect_koopman_data(
        traj_num=200,
        steps=Ksteps,  # gives you 20 time‐points (0…19)
        mode="train",
        input_policy=input_policy,
    )

    test = collector.collect_koopman_data(
        Ktest, Ksteps, mode="eval", input_policy=input_policy
    )

    os.makedirs(os.path.join(save_dir, "Dataset"), exist_ok=True)

    np.save(f"{save_dir}/Dataset/Spirob_Ktrain.npy", train)
    np.save(f"{save_dir}/Dataset/Spirob_Ktest.npy", test)
    print(f"Datasets saved to {save_dir}")


if __name__ == "__main__":
    xml = "../Spirob/2Dspiralrobot/2Dtendon10deg.xml"
    save_koopman_dataset(
        xml,
        "./Spirob_Dataset_consistent",
        Ktrain=200,
        Ktest=40,
        Ksteps=25,
        sim_steps_per_control=1,
        input_policy=right_greater_left_policy,
    )
