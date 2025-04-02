# main.py

import argparse
import pickle
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os

from data_utils import generate_full_data, split_train_val
from model_utils import solve_inner_qpth
from training import train_hyperparams
from evaluation import evaluate_and_print, train_ols, train_mae  # <= import train_mae here
from theoretical_loss import plot_theoretical_autoloss
from save_utils import save_experiment_results

def main():
    parser = argparse.ArgumentParser(description="Run AutoLoss QP Training")
    # ... same arguments as before ...
    args = parser.parse_args()
    
    # 1) Device setup
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 2) Generate data
    X, y, beta_true = generate_full_data(
        n=args.total_sample_size,
        d=args.feature_dimension,
        distribution=args.distribution,
        scale=args.scale,
        seed=args.seed,
        device=device
    )

    # 3) Train/Val split
    X_train, y_train, X_val, y_val = split_train_val(X, y, args.num_training_samples)

    # 4) Initialize hyperparams for AutoLoss
    U = torch.randn(args.L, device=device, requires_grad=True)
    V = torch.randn(args.L, device=device, requires_grad=True)
    S = torch.randn(args.H, device=device, requires_grad=True)
    T = torch.randn(args.H, device=device, requires_grad=True)
    tau = torch.ones(args.H, device=device, requires_grad=False)  # usually not learned

    # Outer loop
    all_val_losses = []
    for it in range(args.num_global_updates):
        if args.verbose:
            print(f"\nGlobal iteration {it+1}/{args.num_global_updates} ...")

        U, V, S, T, val_loss_hist = train_hyperparams(
            X_train, y_train,
            X_val,   y_val,
            U, V, S, T, tau,
            lambda_reg=args.lambda_reg,
            lr=args.lr,
            num_hyperparam_iterations=args.num_hyperparam_iterations,
            loss_type=args.loss_type,
            optimizer_choice=args.optimizer_choice
        )
        beta_opt = solve_inner_qpth(U, V, S, T, tau, X_train, y_train, args.lambda_reg)
        all_val_losses.append(val_loss_hist)

        if args.verbose:
            diff = beta_true - beta_opt
            print(f"> Beta difference: {diff.detach().cpu().numpy()}")

    # Final AutoLoss beta
    beta_autoloss = solve_inner_qpth(U, V, S, T, tau, X_train, y_train, args.lambda_reg)

    # ------------------------------
    # Train MSE-min model (closed-form OLS)
    # ------------------------------
    beta_ols = train_ols(X_train, y_train)

    # ------------------------------
    # Train MAE-min model (gradient-based)
    # ------------------------------
    beta_mae = train_mae(X_train, y_train, lr=1e-2, max_iter=1000)

    # ------------------------------
    # Evaluate all three models
    # ------------------------------
    if args.verbose:
        print("\n=== Final Comparisons on Train and Val ===")

        print("\n--- [AutoLoss Model] ---")
        evaluate_and_print(X_train, y_train, beta_autoloss, beta_true, label="[AutoLoss Train]")
        evaluate_and_print(X_val,   y_val,   beta_autoloss, beta_true, label="[AutoLoss Val]")

        print("\n--- [MSE-Min OLS Model] ---")
        evaluate_and_print(X_train, y_train, beta_ols, beta_true, label="[OLS Train]")
        evaluate_and_print(X_val,   y_val,   beta_ols, beta_true, label="[OLS Val]")

        print("\n--- [MAE-Min Model] ---")
        evaluate_and_print(X_train, y_train, beta_mae, beta_true, label="[MAE Train]")
        evaluate_and_print(X_val,   y_val,   beta_mae, beta_true, label="[MAE Val]")

    # Pack up final results
    autoloss_result = {
        "U": U.detach().clone(),
        "V": V.detach().clone(),
        "S": S.detach().clone(),
        "T": T.detach().clone(),
        "beta_opt": beta_autoloss.detach().clone(),
        "beta_ols": beta_ols.detach().clone(),
        "beta_mae": beta_mae.detach().clone(),
        "tau": tau,
        "lambda_reg": args.lambda_reg,
        "data_distribution": args.distribution,
        "scale": args.scale,
        "metric": args.loss_type,
        "optimizer_choice": args.optimizer_choice
    }

    # Optional: visualize
    if args.visualize:
        val_losses_flat = [v for iteration in all_val_losses for v in iteration]
        plt.figure()
        plt.plot(val_losses_flat, label="Val Loss")
        plt.xlabel("Iteration")
        plt.ylabel(f"{args.loss_type.upper()} Loss")
        plt.title(f"Val Loss Curve ({args.distribution}, {args.loss_type}, Optim={args.optimizer_choice})")
        plt.grid(True)
        plt.legend()
        plt.show()

    # Saving results
    pkl_path, txt_path = save_experiment_results(
        autoloss_result, args,
        beta_autoloss,
        U, V, S, T, tau,
        beta_true,
        all_val_losses,
        X_train, y_train,
        X_val, y_val
    )

if __name__ == "__main__":
    main()
