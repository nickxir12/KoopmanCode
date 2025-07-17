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
    def __init__(self, xml_path="../Spirob/2Dspiralrobot/2Dtendon10deg.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep  # e.g., 0.002
        self.sim_steps_per_control = 10  # you can change this
        self.action_dim = self.model.nu  # number of controls
        self.state_dim = self.model.nq + self.model.nv  # qpos + qvel
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
        return np.concatenate([self.data.qpos, self.data.qvel])


class data_collecter:
    def __init__(self, env_name) -> None:
        self.env_name = env_name
        np.random.seed(2022)
        random.seed(2022)
        if self.env_name.startswith("Spirob"):  # MINE
            # from spirob_env import SpirobEnv  # You define this

            self.env = SpirobEnv()
            self.Nstates = self.env.state_dim  # You define this
            self.udim = self.env.action_dim  # You define this
            self.umin = self.env.umin  # Optionally set
            self.umax = self.env.umax
        else:
            self.env = gym.make(env_name)
            self.env.seed(2022)
            self.udim = self.env.action_space.shape[0]
            self.Nstates = self.env.observation_space.shape[0]
            self.umin = self.env.action_space.low
            self.umax = self.env.action_space.high
        if not self.env_name.endswith("Snake"):
            self.observation_space = self.env.observation_space
            self.env.reset()
            self.dt = self.env.dt

    def random_state(self):
        if self.env_name.startswith("DampingPendulum"):
            th0 = random.uniform(-2 * np.pi, 2 * np.pi)
            dth0 = random.uniform(-8, 8)
            s0 = np.array([th0, dth0])
        elif self.env_name.startswith("Pendulum"):
            th0 = random.uniform(-2 * np.pi, 2 * np.pi)
            dth0 = random.uniform(-8, 8)
            s0 = [th0, dth0]
        elif self.env_name.startswith("CartPole"):
            x0 = random.uniform(-4, 4)
            dx0 = random.uniform(-8, 8)
            th0 = random.uniform(-0.418, 0.418)
            dth0 = random.uniform(-8, 8)
            s0 = [x0, dx0, th0, dth0]
        elif self.env_name.startswith("MountainCarContinuous"):
            x0 = random.uniform(-0.1, 0.1)
            th0 = random.uniform(-0.5, 0.5)
            s0 = [x0, th0]
        elif self.env_name.startswith("InvertedDoublePendulum"):
            x0 = random.uniform(-0.1, 0.1)
            th0 = random.uniform(-0.3, 0.3)
            th1 = random.uniform(-0.3, 0.3)
            dx0 = random.uniform(-1, 1)
            dth0 = random.uniform(-6, 6)
            dth1 = random.uniform(-6, 6)
            s0 = np.array([x0, th0, th1, dx0, dth0, dth1])
        return np.array(s0)

    def collect_koopman_data(self, traj_num, steps, mode="train"):
        train_data = np.empty((steps + 1, traj_num, self.Nstates + self.udim))
        if self.env_name.startswith("Spirob"):
            for traj_i in range(traj_num):
                s0 = self.env.reset()
                u_t = np.random.uniform(self.umin, self.umax, size=self.udim)
                train_data[0, traj_i, :] = np.concatenate([u_t, s0])
                for i in range(1, steps + 1):
                    s1, _, done, _ = self.env.step(u_t)
                    u_t = np.random.uniform(self.umin, self.umax, size=self.udim)
                    train_data[i, traj_i, :] = np.concatenate([u_t, s1])
        else:
            for traj_i in range(traj_num):
                s0 = self.env.reset()
                # s0 = self.random_state()
                u10 = np.random.uniform(self.umin, self.umax)
                # self.env.reset_state(s0)
                train_data[0, traj_i, :] = np.concatenate(
                    [u10.reshape(-1), s0.reshape(-1)], axis=0
                ).reshape(-1)
                for i in range(1, steps + 1):
                    s0, r, done, _ = self.env.step(u10)
                    u10 = np.random.uniform(self.umin, self.umax)
                    train_data[i, traj_i, :] = np.concatenate(
                        [u10.reshape(-1), s0.reshape(-1)], axis=0
                    ).reshape(-1)
        return train_data

    def collect_detivative_data(self, traj_num, steps):
        train_data = np.empty((steps + 1, traj_num, self.Nstates + self.udim))
        for traj_i in range(traj_num):
            # s0 = self.env.reset()
            s0 = self.random_state()
            u10 = np.random.uniform(self.umin, self.umax)
            self.env.reset_state(s0)
            # print(s0,np.array(u10))
            # print(s0,u10)
            train_data[0, traj_i, :] = np.concatenate(
                [u10.reshape(-1), s0.reshape(-1)], axis=0
            ).reshape(-1)
            for i in range(1, steps + 1):
                s0, r, done, _ = self.env.step(u10)
                u10 = np.random.uniform(self.umin, self.umax)
                train_data[i, traj_i, :] = np.concatenate(
                    [u10.reshape(-1), s0.reshape(-1)], axis=0
                ).reshape(-1)
        return train_data
