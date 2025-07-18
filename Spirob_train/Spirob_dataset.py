import os
import numpy as np

Ktrain_samples = 2000
Ktest_samples = 200
Ksteps = 15
Kbatch_size = 100
res = 1
normal = 1

# File paths
data_dir = "../Spirob_data"
os.makedirs(data_dir, exist_ok=True)
train_file = os.path.join(data_dir, "Ktrain_data.npy")
test_file = os.path.join(data_dir, "Ktest_data.npy")

# Data collection
data_collect = data_collecter(env_name)
u_dim = data_collect.udim

# ----- Load or Generate Test Data -----
if os.path.exists(test_file):
    Ktest_data = np.load(test_file)
    print("Loaded test data from file:", test_file)
else:
    Ktest_data = data_collect.collect_koopman_data(Ktest_samples, Ksteps, mode="eval")
    np.save(test_file, Ktest_data)
    print("Generated and saved test data:", Ktest_data.shape)

# ----- Load or Generate Train Data -----
if os.path.exists(train_file):
    Ktrain_data = np.load(train_file)
    print("Loaded train data from file:", train_file)
else:
    Ktrain_data = data_collect.collect_koopman_data(
        Ktrain_samples, Ksteps, mode="train"
    )
    np.save(train_file, Ktrain_data)
    print("Generated and saved train data:", Ktrain_data.shape)

# Update sample counts
Ktrain_samples = Ktrain_data.shape[1]
Ktest_samples = Ktest_data.shape[1]

# Input dimension
in_dim = Ktest_data.shape[-1] - u_dim
Nstate = in_dim
