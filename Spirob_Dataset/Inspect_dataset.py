import os
import sys
import numpy as np

# tolerated joint range (from your XML)
JOINT_MIN = -0.523
JOINT_MAX = 0.523


def inspect_file(path, action_dim, joint_count):

    data = np.load(path)  # shape: (time_steps, trajs, D)
    T, N, D = data.shape
    state_dim = D - action_dim

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"  shape         : time={T}, trajs={N}, dims={D}")
    print(f"  action_dim    : {action_dim}")
    print(f"  state_dim     : {state_dim}")
    print(f"  joint_count   : {joint_count}")
    print()

    # split U / S
    U = data[..., :action_dim]
    S = data[..., action_dim:]
    # within state: [q0…q20, qd0…qd20]
    Q = S[..., :joint_count]
    Qd = S[..., joint_count : joint_count * 2]

    # overall min/max
    q_min, q_max = Q.min(), Q.max()
    print(f"  joint angles  : min={q_min:.3f}, max={q_max:.3f}")
    print(f"  joint velo    : min={Qd.min():.3f}, max={Qd.max():.3f}")
    print(f"  control inputs: min={U.min():.3f}, max={U.max():.3f}")

    # find violations
    bad_lo = np.where(Q < JOINT_MIN)
    bad_hi = np.where(Q > JOINT_MAX)
    n_lo = len(bad_lo[0])
    n_hi = len(bad_hi[0])
    print(f"  violations    : below min = {n_lo}, above max = {n_hi}")

    if n_lo + n_hi > 0:
        # show up to 5 examples
        print("\n  Examples of OOB angles (time, traj, joint, value):")
        examples = []
        # stack lo & hi
        for arr, tag in [(bad_lo, "<"), (bad_hi, ">")]:
            t_idx, n_idx, j_idx = arr
            for t, n, j in zip(t_idx, n_idx, j_idx):
                examples.append((t, n, j, Q[t, n, j], tag))
                if len(examples) >= 5:
                    break
            if len(examples) >= 5:
                break
        for t, n, j, val, tag in examples:
            print(f"    step={t}, traj={n}, joint={j}, value={val:.3f} rad ({tag})")
    print("-" * 40)


def main(data_dir):
    ACTION_DIM = 2
    JOINT_COUNT = 21

    for fname in ("Spirob_Ktrain.npy", "Spirob_Ktest.npy"):
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            print(f"  ❌ missing {path}")
            continue
        inspect_file(path, ACTION_DIM, JOINT_COUNT)


if __name__ == "__main__":
    # 👇 Set the dataset path here instead of using command-line args
    dataset_dir = "./Spirob_Dataset/Spirob_Dataset_consistent"
    main(dataset_dir)
