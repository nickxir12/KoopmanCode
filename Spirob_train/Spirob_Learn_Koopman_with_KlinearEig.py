import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from copy import copy
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import sys
from pathlib import Path

sys.path.append("../utility/")
from scipy.integrate import odeint
from Utility import data_collecter
import time


import sys

PRINT_EVERY = 100  # ← print every N gradient steps


# define network
def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(
        torch.Tensor([0]), torch.Tensor([std / n_units])
    )
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


class Network(nn.Module):
    def __init__(self, encode_layers, Lifted_dim, u_dim):
        super(Network, self).__init__()
        # DEFINE ENCODING NETWORK BELOW
        Layers = OrderedDict()
        for layer_i in range(len(encode_layers) - 1):
            Layers["linear_{}".format(layer_i)] = nn.Linear(
                encode_layers[layer_i], encode_layers[layer_i + 1]
            )
            if layer_i != len(encode_layers) - 2:
                Layers["relu_{}".format(layer_i)] = nn.ReLU()
        self.encode_net = nn.Sequential(Layers)
        self.Lifted_dim = Lifted_dim
        self.u_dim = u_dim
        self.lA = nn.Linear(Lifted_dim, Lifted_dim, bias=False)
        self.lA.weight.data = gaussian_init_(Lifted_dim, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        self.lB = nn.Linear(u_dim, Lifted_dim, bias=False)

    def encode_only(self, x):
        return self.encode_net(x)

    def encode(self, x):
        return torch.cat([x, self.encode_net(x)], axis=-1)

    def forward(self, x, u):
        return self.lA(x) + self.lB(u)


# def K_loss(data, net, u_dim=1, Nstate=4):
#     steps, train_traj_num, Nstates = data.shape
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     data = torch.DoubleTensor(data).to(device)
#     X_current = net.encode(data[0, :, u_dim:])
#     max_loss_list = []
#     mean_loss_list = []
#     for i in range(steps - 1):
#         X_current = net.forward(X_current, data[i, :, :u_dim])
#         Y = data[i + 1, :, u_dim:]
#         Err = X_current[:, :Nstate] - Y
#         max_loss_list.append(
#             torch.mean(torch.max(torch.abs(Err), axis=0).values).detach().cpu().numpy()
#         )
#         mean_loss_list.append(
#             torch.mean(torch.mean(torch.abs(Err), axis=0)).detach().cpu().numpy()
#         )
#     return np.array(max_loss_list), np.array(mean_loss_list)


def angle_limit_penalty(angles_pred, low=-0.523, high=0.523):
    # amount by which each prediction violates the box
    low_violation = torch.relu(low - angles_pred)  # (< low)  → positive
    high_violation = torch.relu(angles_pred - high)  # (> high) → positive
    return torch.mean((low_violation**2 + high_violation**2))


# loss function
def Klinear_loss(data, net, mse_loss, u_dim=1, gamma=0.99, Nstate=4, all_loss=0):
    steps, train_traj_num, Lifted_dim = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    X_current = net.encode(data[0, :, u_dim:])
    beta = 1.0
    beta_sum = 0.0
    loss = torch.zeros(1, dtype=torch.float64).to(device)
    Augloss = torch.zeros(1, dtype=torch.float64).to(device)
    for i in range(steps - 1):
        X_current = net.forward(X_current, data[i, :, :u_dim])
        beta_sum += beta
        if not all_loss:
            loss += beta * mse_loss(X_current[:, :Nstate], data[i + 1, :, u_dim:])
        else:
            Y = net.encode(data[i + 1, :, u_dim:])
            loss += beta * mse_loss(X_current, Y)
        X_current_encoded = net.encode(X_current[:, :Nstate])
        Augloss += mse_loss(X_current_encoded, X_current)

        # ------------------------------------------------------------------
        # ❸ **angle-limit penalty**
        # ------------------------------------------------------------------
        angles_pred = X[:, :21]  # first 21 dims = joint angles
        angle_pen = angle_limit_penalty(angles_pred)
        loss += beta * lam_bound * angle_pen

        beta *= gamma
    loss = loss / beta_sum
    Augloss = Augloss / beta_sum
    return loss + 0.5 * Augloss


# def Stable_loss(net, Nstate):
#     x_ref = np.zeros(Nstate)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x_ref_lift = net.encode_only(torch.DoubleTensor(x_ref).to(device))
#     loss = torch.norm(x_ref_lift)
#     return loss


def Eig_loss(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = net.lA.weight
    c = torch.linalg.eigvals(A).abs() - torch.ones(1, dtype=torch.float64).to(device)
    mask = c > 0
    loss = c[mask].sum()
    return loss


def load_or_collect_dataset(env_name, use_saved, Ktrain_samples, Ksteps):
    if use_saved:
        # resolve dataset folder relative to this script
        ds = (
            Path(__file__).resolve().parent.parent
            / "Spirob_Dataset"
            / "Spirob_Dataset_25steps_right_gt_left"
        )
        train_path = ds / f"{env_name}_Ktrain_v1.npy"
        test_path = ds / f"{env_name}_Ktest_v1.npy"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Cannot find dataset in {ds!r}")

        Ktrain = np.load(train_path)
        Ktest = np.load(test_path)
    else:
        coll = data_collecter(env_name)
        Ktrain = coll.collect_koopman_data(Ktrain_samples, Ksteps, mode="train")
        Ktest = coll.collect_koopman_data(200, Ksteps, mode="eval")

    # infer dims
    coll = data_collecter(env_name)
    u_dim = coll.udim
    in_dim = Ktrain.shape[-1] - u_dim

    return Ktrain, Ktest, u_dim, in_dim


def train(
    env_name="Spirob",
    train_steps=1000,
    batch_size=100,
    learning_rate=5e-4,
    layer_width=128,
    layer_depth=3,
    encode_dim=12,
    all_loss=0,
    e_loss=1,
    gamma=0.5,
    Ktrain_samples=500,
    Ksteps=25,
    use_saved_dataset=True,
    dataset_dir="./Spirob_Dataset/Spirob_Dataset_25steps_right_gt_left",  # Selecte dataset directory
    suffix="",
):

    print("\n=== Training Configuration ===")
    for k, v in locals().items():
        print(f"{k:>20}: {v}")
    print("==============================\n")

    ASCII_LOG_ROOT = r"C:\Users\nikolas\Desktop\tensorboard_logs"
    os.makedirs(ASCII_LOG_ROOT, exist_ok=True)

    Ksteps = 15
    Kbatch_size = batch_size
    res = 1
    normal = 1

    Ktrain_data, Ktest_data, u_dim, in_dim = load_or_collect_dataset(
        env_name, use_saved_dataset, Ktrain_samples, Ksteps
    )

    print("in_dim =", in_dim)
    print("u_dim =", u_dim)
    print("Ktrain_data shape =", Ktrain_data.shape)
    print("5 Sample Inputs:", Ktrain_data[:1, :5, :2])  # Print first input
    print("Sample input+state vector:", Ktrain_data[0, 0])

    layer_width = layer_width
    layers = [in_dim] + [layer_width] * layer_depth + [encode_dim]
    Lifted_dim = in_dim + encode_dim
    Nstate = in_dim

    print("Lifted_dim =", Lifted_dim)
    print("layers:", layers)
    net = Network(layers, Lifted_dim, u_dim)

    if torch.cuda.is_available():
        net.cuda()
    net.double()

    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

    for name, param in net.named_parameters():
        print("model:", name, param.requires_grad)

    best_loss = float("inf")

    best_state_dict = {}
    eval_step = 1000

    logdir = os.path.join(
        ASCII_LOG_ROOT,
        "KoopmanU_{}_layer{}_edim{}_eloss{}_gamma{}_aloss{}_samples{}".format(
            env_name, layer_depth, encode_dim, e_loss, gamma, all_loss, Ktrain_samples
        ),
    )
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir)

    start_time = time.process_time()

    for i in range(train_steps):
        # K loss — sample from the real data you loaded
        num_trajs = Ktrain_data.shape[1]
        Kindex = list(range(num_trajs))
        random.shuffle(Kindex)
        X = Ktrain_data[:, Kindex[:batch_size], :]
        Kloss = Klinear_loss(X, net, mse_loss, u_dim, gamma, Nstate, all_loss)
        Eloss = Eig_loss(net)
        loss = Kloss + Eloss if e_loss else Kloss

        # loss = Kloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("Train/Kloss", Kloss, i)
        writer.add_scalar("Train/Eloss", Eloss, i)
        writer.add_scalar("Train/loss", loss, i)

        # ─── *** PRINT TO TERMINAL *** ─────────────────────
        if i % PRINT_EVERY == 0:
            # use .item() to get Python scalars
            print(
                f"Step {i:>7}/{train_steps}  "
                f"loss: {loss.item():.4e}  "
                f"Kloss: {Kloss.item():.4e}  "
                f"Eloss: {Eloss.item():.4e}"
            )

        if (i + 1) % eval_step == 0:
            # K loss
            with torch.no_grad():
                Kloss = Klinear_loss(
                    Ktest_data, net, mse_loss, u_dim, gamma, Nstate, all_loss=0
                )
                Eloss = Eig_loss(net)
                loss = Kloss
                Kloss = Kloss.detach().cpu().numpy()
                Eloss = Eloss.detach().cpu().numpy()
                loss = loss.detach().cpu().numpy()
                writer.add_scalar("Eval/Kloss", Kloss, i)
                writer.add_scalar("Eval/Eloss", Eloss, i)
                writer.add_scalar("Eval/best_loss", best_loss, i)
                writer.add_scalar("Eval/loss", loss, i)
                if loss < best_loss:
                    best_loss = copy(Kloss)
                    best_state_dict = copy(net.state_dict())

                    # Extract Koopman operators A and B (as numpy arrays)
                    A = net.lA.weight.detach().cpu().numpy()
                    B = net.lB.weight.detach().cpu().numpy()

                    # Save everything into the checkpoint
                    Saved_dict = {
                        "model": best_state_dict,
                        "layer": layers,
                        "A": A,
                        "B": B,
                    }
                    torch.save(Saved_dict, ("../Spirob_checkpoints/best_model_v1.pth"))

                print("Step:{} Eval-loss{} K-loss:{} ".format(i, loss, Kloss))
            # print("-------------END-------------")
        writer.add_scalar("Eval/best_loss", best_loss, i)
        # if (time.process_time()-start_time)>=210*3600:
        #     print("time out!:{}".format(time.clock()-start_time))
        #     break
    print("END-best_loss{}".format(best_loss))


# ---------------- main -----------------
if __name__ == "__main__":
    # Define all training arguments here in Python
    training_args = {
        "env_name": "Spirob",
        "dataset_dir": "./Spirob_Dataset/Spirob_Dataset_25steps_right_gt_left",
        "use_saved_dataset": True,  # <- NOW SET HERE
        "train_steps": 50000,
        "batch_size": 100,
        "learning_rate": 5e-5,
        "layer_depth": 3,
        "layer_width": 128,
        "encode_dim": 44,
        "gamma": 0.5,
        "all_loss": 0,
        "e_loss": 1,
        "Ktrain_samples": 5000,
        "Ksteps": 25,
        "suffix": "",
    }

    # Call training with these arguments
    train(**training_args)
