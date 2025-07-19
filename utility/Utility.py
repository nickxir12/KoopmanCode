import numpy as np
import gym
import random
from scipy.integrate import odeint
import scipy.linalg
from copy import copy
from rbf import rbf
from gym import spaces
import sys
import time

sys.path.append("../franka")
# data collect

import numpy as np
import mujoco
from mujoco import MjModel, MjData
import os


import mujoco
import numpy as np


# --- Environment Wrapper ---


class SpirobEnv:
    def __init__(self, xml_path, sim_steps_per_control=1):
        # Load model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.sim_steps = sim_steps_per_control

        # List the 21 hinge joints by name
        self.joint_names = [f"J{i}" for i in range(1, 22)]
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]

        # Addresses for qpos and qvel
        self.qpos_addrs = [self.model.jnt_qposadr[j] for j in self.joint_ids]
        self.qvel_addrs = [self.model.jnt_dofadr[j] for j in self.joint_ids]

        # Actuator limits
        cr = self.model.actuator_ctrlrange
        self.umin, self.umax = cr[:, 0].copy(), cr[:, 1].copy()

        # Dimensions
        self.action_dim = self.model.nu
        self.state_dim = len(self.qpos_addrs) + len(self.qvel_addrs)  # 21 + 21

    def reset(self):
        # Reset simulation state
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        return self._get_state()

    def step(self, u):
        self.data.ctrl[:] = np.clip(u, self.umin, self.umax)
        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        time.sleep(self.model.opt.timestep * 10)  # Slow down for visibility
        return self._get_state(), 0.0, False, {}

    def _get_state(self):
        # Extract hinge positions and velocities
        q = self.data.qpos[self.qpos_addrs].copy()
        qd = self.data.qvel[self.qvel_addrs].copy()
        return np.concatenate([q, qd])


# Sample policy: constant u0<u1 for entire trajectory
def right_greater_left_policy(_):
    u0 = np.random.uniform(0.0, 0.2)
    diff_choices = [0.02, 0.05, 0.1, 0.2, 0.4]
    diff_probs = [0.3, 0.3, 0.2, 0.15, 0.05]
    diff = np.random.choice(diff_choices, p=diff_probs)
    u1 = np.clip(u0 + diff, u0 + 0.01, 0.6)
    return np.array([u0, u1])


class DataCollector:
    def __init__(self, xml_path, sim_steps_per_control=1, seed=2022):
        random.seed(seed)
        np.random.seed(seed)
        self.env = SpirobEnv(xml_path, sim_steps_per_control)
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        self.umin, self.umax = self.env.umin, self.env.umax

    def collect_koopman_data(
        self, traj_num, steps, mode="train", input_policy=None, max_tries=1
    ):
        D = self.action_dim + self.state_dim
        data = np.zeros((steps + 1, traj_num, D), dtype=np.float32)

        for t in range(traj_num):
            s = self.env.reset()
            u = input_policy(None) if input_policy else np.array([0.05, 0.1])
            u = np.clip(u, self.umin, self.umax)
            traj = np.zeros((steps + 1, D), dtype=np.float32)
            traj[0] = np.hstack([u, s])
            if t % 100 == 0:
                print(f"\n--- Trajectory {t} ---")
                print(f"Step 0 | u = {u} | q = {s[:self.state_dim//2]}")

            for i in range(1, steps + 1):
                if (t % 5 == 0) and (
                    i % 1 == 0
                ):  # show robot on every 100th trajectory
                    # Step & show visual slowly
                    s, _, _, _ = self.env.step(u)
                    time.sleep(self.env.model.opt.timestep * 10)  # slow visual
                else:
                    self.env.data.ctrl[:] = np.clip(u, self.umin, self.umax)
                    for _ in range(self.env.sim_steps):
                        mujoco.mj_step(self.env.model, self.env.data)
                    mujoco.mj_forward(self.env.model, self.env.data)
                    s = self.env._get_state()

                q = s[: self.state_dim // 2]
                # Clip joint angles to [-0.523, 0.523] (30 degrees)
                q = np.clip(q, -0.523, 0.523)

                traj[i] = np.hstack([u, s])
                # print(f"Step {i} | u = {u} | q = {q}")

            data[:, t, :] = traj

        return data
