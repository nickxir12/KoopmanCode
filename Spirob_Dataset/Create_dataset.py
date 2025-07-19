import numpy as np
import os
import sys

sys.path.append("../utility/")
from Utility import data_collecter  # Ensure correct import path


def right_greater_left_policy(_):
    """
    Returns a constant input [u0, u1] for an entire trajectory,
    where u1 > u0 and both are in [0, 0.6].
    The difference (u1 - u0) is sampled probabilistically with variation.
    """
    # Sample base u0 in [0, 0.55] to allow some headroom
    u0 = np.random.uniform(0.0, 0.55)

    # Sample the difference with bias: small diff is more likely, but allow larger diffs
    diff_choices = [0.02, 0.05, 0.1, 0.2, 0.4]
    diff_probs = [0.3, 0.3, 0.2, 0.15, 0.05]  # More likely to get small u1-u0

    diff = np.random.choice(diff_choices, p=diff_probs)

    u1 = np.clip(u0 + diff, u0 + 0.01, 0.6)  # ensure u1 > u0 and ≤ 0.6

    return np.array([u0, u1])


def save_koopman_dataset(
    env_name,
    Ktrain_samples=2000,
    Ktest_samples=400,
    Ksteps=25,
    save_dir="./Spirob_Dataset",
    sim_steps_per_control=10,
    input_policy=None,
):
    os.makedirs(save_dir, exist_ok=True)

    collector = data_collecter(env_name, sim_steps_per_control=sim_steps_per_control)
    train_data = collector.collect_koopman_data(
        Ktrain_samples, Ksteps, mode="train", input_policy=input_policy
    )
    test_data = collector.collect_koopman_data(
        Ktest_samples, Ksteps, mode="eval", input_policy=input_policy
    )

    np.save(os.path.join(save_dir, f"{env_name}_Ktrain_v1.npy"), train_data)
    np.save(os.path.join(save_dir, f"{env_name}_Ktest_v1.npy"), test_data)
    print(f"Datasets saved to {save_dir}")


if __name__ == "__main__":
    save_koopman_dataset(
        env_name="Spirob",
        Ktrain_samples=2000,
        Ktest_samples=400,
        Ksteps=25,
        save_dir="./Spirob_Dataset_25steps_right_gt_left",
        sim_steps_per_control=10,
        input_policy=right_greater_left_policy,
    )
