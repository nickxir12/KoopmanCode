import numpy as np
import mujoco

# Replace this with the correct relative path to your XML
XML_PATH = "../Spirob/2Dspiralrobot/2Dtendon10deg.xml_fixed.xml"


def main():
    # 1) Load model & data
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 2) Find the 21 J-joint indices
    joint_idxs = sorted(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for i in range(model.njnt)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
        and name.startswith("J")
    )
    joint_qpos_addrs = [model.jnt_qposadr[j] for j in joint_idxs]
    joint_qvel_addrs = [model.jnt_dofadr[j] for j in joint_idxs]

    # 3) Zero‐action test
    reset_state = lambda: (mujoco.mj_forward(model, data), None)[1]
    sim_steps = 10
    zero_ctrl = np.zeros(model.nu)

    # Reset
    reset_state()
    q0 = np.array([data.qpos[a] for a in joint_qpos_addrs])
    print("After reset (deg):", np.round(np.rad2deg(q0), 3))

    # Do a few steps
    for step in range(1, 6):
        data.ctrl[:] = zero_ctrl
        for _ in range(sim_steps):
            mujoco.mj_step(model, data)
        qa = np.array([data.qpos[a] for a in joint_qpos_addrs])
        print(f"Step {step} (deg):", np.round(np.rad2deg(qa), 3))

    # 4) Try a constant nonzero control
    u = np.array([0.3, 0.1])  # tune to your ctrl_dim
    print("\nApplying constant control:", u)
    reset_state()
    for step in range(1, 6):
        data.ctrl[:] = u
        for _ in range(sim_steps):
            mujoco.mj_step(model, data)
        qa = np.array([data.qpos[a] for a in joint_qpos_addrs])
        print(f"Step {step} (deg):", np.round(np.rad2deg(qa), 3))


if __name__ == "__main__":
    main()
