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


class SpirobEnv:
    def __init__(
        self,
        xml_path="../Spirob/2Dspiralrobot/2Dtendon10deg.xml",
        sim_steps_per_control=10,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep  # e.g., 0.002
        self.sim_steps_per_control = sim_steps_per_control  # you can change this
        self.action_dim = self.model.nu  # number of controls
        self.state_dim = 21 + 21  # qpos + qvel
        self.umin = np.full(self.action_dim, self.model.actuator_ctrlrange[:, 0])
        self.umax = np.full(self.action_dim, self.model.actuator_ctrlrange[:, 1])

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.umin, high=self.umax, shape=(self.action_dim,), dtype=np.float32
        )

    def reset(self):
        self.data = mujoco.MjData(self.model)  # full reset
        return self.get_state()

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, self.umin, self.umax)
        for _ in range(self.sim_steps_per_control):
            mujoco.mj_step(self.model, self.data)
        return self.get_state(), 0.0, False, {}

    def get_state(self):
        # Remove first 7 qpos (free root), and first 6 qvel (root velocities)
        qpos = self.data.qpos[7:]  # 21 joint positions
        qvel = self.data.qvel[6:]  # 21 joint velocities
        return np.concatenate([qpos, qvel])


class data_collecter:
    def __init__(self, env_name, sim_steps_per_control=10):
        self.env_name = env_name
        np.random.seed(2022)
        random.seed(2022)
        if self.env_name.startswith("Spirob"):
            self.env = SpirobEnv(sim_steps_per_control=sim_steps_per_control)
            self.Nstates = self.env.state_dim
            self.udim = self.env.action_dim
            self.umin = self.env.umin
            self.umax = self.env.umax

    def collect_koopman_data(self, traj_num, steps, mode="train", input_policy=None):
        data = np.empty((steps + 1, traj_num, self.Nstates + self.udim))

        for traj_i in range(traj_num):
            s0 = self.env.reset()

            amp_seq = (
                np.random.uniform(0, 1) if mode == "train" else np.linspace(0, 1, steps)
            )
            amp = amp_seq if isinstance(amp_seq, np.ndarray) else [amp_seq] * steps

            # Initial input
            u_t = (
                input_policy(amp[0])
                if input_policy
                else np.array([amp[0], 0.25 * amp[0]])
            )
            data[0, traj_i, :] = np.concatenate([u_t, s0])

            for i in range(1, steps + 1):
                s1, _, _, _ = self.env.step(u_t)

                if mode == "train":
                    if np.random.rand() < 0.1:
                        new_amp = np.clip(np.random.uniform(0, 1), 0, 1)
                        u_t = (
                            input_policy(new_amp)
                            if input_policy
                            else np.array([new_amp, 0.25 * new_amp])
                        )
                else:
                    if i < steps:
                        a = amp[i]
                        u_t = (
                            input_policy(a) if input_policy else np.array([a, 0.25 * a])
                        )

                data[i, traj_i, :] = np.concatenate([u_t, s1])

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
