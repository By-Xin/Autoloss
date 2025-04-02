# run_grid_search.py
import itertools
import subprocess
import torch

distributions = ["laplace", "normal"]
loss_types = ["mse", "mae"]
L_values = [0, 1]
H_values = [2, 3]
total_sizes = [200, 300]
feat_dims = [5, 10]
train_ratio = 0.75
val_ratio = 0.1
global_updates_list = [3, 5]
hyper_iters_list = [4, 8]

# 任意固定参数
lambda_reg = 0.1
lr = 1e-2
scale = 1.0
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for dist, loss_type, L_val, H_val, tot_size, feat, n_global, n_hyper \
    in itertools.product(
        distributions,
        loss_types,
        L_values,
        H_values,
        total_sizes,
        feat_dims,
        global_updates_list,
        hyper_iters_list
    ):
    
    cmd = [
        "python", "main.py",
        "--distribution", dist,
        "--loss_type", loss_type,
        "--optimizer_choice", "adam",
        "--L", str(L_val),
        "--H", str(H_val),
        "--total_sample_size", str(tot_size),
        "--feature_dimension", str(feat),
        "--train_ratio", str(train_ratio),
        "--val_ratio", str(val_ratio),
        "--num_global_updates", str(n_global),
        "--num_hyperparam_iterations", str(n_hyper),
        "--lambda_reg", str(lambda_reg),
        "--lr", str(lr),
        "--scale", str(scale),
        "--seed", str(seed),
        "--device", device.type,
        "--verbose"
        # 如果想可视化，也可以加上 "--visualize"
    ]
    
    print("Running: ", " ".join(cmd))
    subprocess.run(cmd, check=True)
