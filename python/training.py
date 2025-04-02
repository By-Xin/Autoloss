# training.py

import torch
from tqdm import trange

from model_utils import solve_inner_qpth

def compute_outer_loss(X_train, y_train,
                       X_val, y_val,
                       U, V, S, T, tau,
                       lambda_reg,
                       loss_type="mse"):
    """
    先 solve_inner_qpth 得到 beta_opt，再在验证集计算外层损失 (MSE or MAE)。
    """
    beta_opt = solve_inner_qpth(U, V, S, T, tau, X_train, y_train, lambda_reg)
    n_val = X_val.shape[0]
    y_val_pred = X_val @ beta_opt

    if loss_type == "mse":
        loss_outer = (1.0 / n_val) * (y_val - y_val_pred).pow(2).sum()
    elif loss_type == "mae":
        loss_outer = (1.0 / n_val) * (y_val - y_val_pred).abs().sum()
    else:
        raise ValueError(f"Unknown loss type '{loss_type}'")

    return loss_outer, beta_opt


def train_hyperparams(X_train, y_train,
                      X_val,   y_val,
                      U, V, S, T, tau,
                      lambda_reg,
                      lr=1e-2,
                      num_hyperparam_iterations=50,
                      loss_type="mse",
                      optimizer_choice="adam"):
    """
    用 PyTorch 优化器更新 U, V, S, T。可指定 optimizer_choice='adam','sgd','adamw'。
    """
    if optimizer_choice == "adam":
        optimizer = torch.optim.Adam([U, V, S, T], lr=lr)
    elif optimizer_choice == "sgd":
        optimizer = torch.optim.SGD([U, V, S, T], lr=lr)
    elif optimizer_choice == "adamw":
        optimizer = torch.optim.AdamW([U, V, S, T], lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer choice: {optimizer_choice}.")

    loss_outer_history = []
    progress_bar = trange(num_hyperparam_iterations, desc='Hyperparam Updates', leave=True)
    
    for step in progress_bar:
        optimizer.zero_grad()
        loss_outer, beta_opt = compute_outer_loss(X_train, y_train,
                                           X_val,   y_val,
                                           U, V, S, T, tau,
                                           lambda_reg,
                                           loss_type)
        loss_outer.backward()
        optimizer.step()

        loss_val = loss_outer.item()
        loss_outer_history.append(loss_val)
        progress_bar.set_postfix(val_loss=f"{loss_val:.6f}")

    return U, V, S, T, loss_outer_history, beta_opt
