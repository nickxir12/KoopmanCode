import numpy as np
import gym
import random
from scipy.integrate import odeint
import scipy.linalg
from copy import copy
from rbf import rbf
from gym import spaces
import sys

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
    def __init__(self, xml_path, sim_steps_per_control=10):
        # Load model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.sim_steps_per_control = sim_steps_per_control

        # Locate hinge joints J1-J21
        self.joint_idxs = sorted(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                for i in range(self.model.njnt)
            )
            if name and name.startswith("J")
        )
        assert (
            len(self.joint_idxs) == 21
        ), f"Expected 21 joints, got {len(self.joint_idxs)}"

        # Map to addresses
        self.joint_qpos_addrs = [self.model.jnt_qposadr[j] for j in self.joint_idxs]
        self.joint_qvel_addrs = [self.model.jnt_dofadr[j] for j in self.joint_idxs]

        # Control dims
        self.action_dim = self.model.nu
        cr = self.model.actuator_ctrlrange
        self.umin = cr[:, 0]
        self.umax = cr[:, 1]

        # State dim
        self.state_dim = 42

    def reset(self):
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        print("[DEBUG] Joint limits:", self.model.jnt_range[self.joint_idxs])

        return self.get_state()

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, self.umin, self.umax)
        for _ in range(self.sim_steps_per_control):
            mujoco.mj_step1(self.model, self.data)
            mujoco.mj_step2(self.model, self.data)
        return self.get_state(), 0.0, False, {}

    def get_state(self):
        q = np.array([self.data.qpos[a] for a in self.joint_qpos_addrs])
        qd = np.array([self.data.qvel[a] for a in self.joint_qvel_addrs])
        state = np.concatenate([q, qd])
        if not np.all(np.isfinite(state)):
            raise RuntimeError("Non-finite state")
        return state


# --- Data Collector ---
class DataCollector:
    def __init__(self, xml_path, sim_steps_per_control=10, seed=2022):
        np.random.seed(seed)
        random.seed(seed)
        self.env = SpirobEnv(xml_path, sim_steps_per_control)
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        self.umin = self.env.umin
        self.umax = self.env.umax

    def collect_koopman_data(self, traj_num, steps, mode="train", input_policy=None):
        D = self.action_dim + self.state_dim
        data = np.zeros((steps + 1, traj_num, D), dtype=np.float32)
        for t in range(traj_num):
            s = self.env.reset()
            assert np.all(np.abs(s[:21]) <= 0.54 + 1e-6)
            a = np.random.rand() if mode == "train" else 0.0
            u = input_policy(a) if input_policy else np.array([a, 0.25 * a])
            data[0, t] = np.hstack([u, s])
            for i in range(1, steps + 1):
                s, _, _, _ = self.env.step(u)
                if not np.all(np.abs(s[:21]) <= 0.54 + 1e-6):
                    raise ValueError(f"OOB at t={t} i={i} deg={np.rad2deg(s[:21])}")
                if mode == "train" and np.random.rand() < 0.1:
                    a = np.random.rand()
                elif mode != "train" and i < steps:
                    a = i / (steps - 1)
                else:
                    a = None
                if a is not None:
                    u = input_policy(a) if input_policy else np.array([a, 0.25 * a])
                data[i, t] = np.hstack([u, s])
        return data

    # def random_state(self):
    #     if self.env_name.startswith("DampingPendulum"):
    #         th0 = random.uniform(-2 * np.pi, 2 * np.pi)
    #         dth0 = random.uniform(-8, 8)
    #         s0 = np.array([th0, dth0])
    #     elif self.env_name.startswith("Pendulum"):
    #         th0 = random.uniform(-2 * np.pi, 2 * np.pi)
    #         dth0 = random.uniform(-8, 8)
    #         s0 = [th0, dth0]
    #     elif self.env_name.startswith("CartPole"):
    #         x0 = random.uniform(-4, 4)
    #         dx0 = random.uniform(-8, 8)
    #         th0 = random.uniform(-0.418, 0.418)
    #         dth0 = random.uniform(-8, 8)
    #         s0 = [x0, dx0, th0, dth0]
    #     elif self.env_name.startswith("MountainCarContinuous"):
    #         x0 = random.uniform(-0.1, 0.1)
    #         th0 = random.uniform(-0.5, 0.5)
    #         s0 = [x0, th0]
    #     elif self.env_name.startswith("InvertedDoublePendulum"):
    #         x0 = random.uniform(-0.1, 0.1)
    #         th0 = random.uniform(-0.3, 0.3)
    #         th1 = random.uniform(-0.3, 0.3)
    #         dx0 = random.uniform(-1, 1)
    #         dth0 = random.uniform(-6, 6)
    #         dth1 = random.uniform(-6, 6)
    #         s0 = np.array([x0, th0, th1, dx0, dth0, dth1])
    #     return np.array(s0)

    # def collect_detivative_data(self, traj_num, steps):
    #     train_data = np.empty((steps + 1, traj_num, self.Nstates + self.udim))
    #     for traj_i in range(traj_num):
    #         # s0 = self.env.reset()
    #         s0 = self.random_state()
    #         u10 = np.random.uniform(self.umin, self.umax)
    #         self.env.reset_state(s0)
    #         # print(s0,np.array(u10))
    #         # print(s0,u10)
    #         train_data[0, traj_i, :] = np.concatenate(
    #             [u10.reshape(-1), s0.reshape(-1)], axis=0
    #         ).reshape(-1)
    #         for i in range(1, steps + 1):
    #             s0, r, done, _ = self.env.step(u10)
    #             u10 = np.random.uniform(self.umin, self.umax)
    #             train_data[i, traj_i, :] = np.concatenate(
    #                 [u10.reshape(-1), s0.reshape(-1)], axis=0
    #             ).reshape(-1)
    #     return train_data
