# main.py

import argparse
import pickle
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 导入各模块
from data_utils import generate_full_data, split_train_val, generate_test_data
from model_utils import solve_inner_qpth
from training import train_hyperparams
from evaluation import evaluate_and_print, compute_test_Xbeta, train_reg_l1, train_reg_l2,calc_beta_metrics, calc_pred_metrics
from theoretical_loss import plot_theoretical_autoloss
from save_utils import save_experiment_results

def main():
    parser = argparse.ArgumentParser(description="Run AutoLoss QP Training")

    # 添加命令行参数
    parser.add_argument("--total_sample_size", type=int, default=200, help="Total data sample size")
    parser.add_argument("--feature_dimension", type=int, default=5, help="Number of features")
    parser.add_argument("--L", type=int, default=0, help="Size of U, V")
    parser.add_argument("--H", type=int, default=3, help="Size of S, T")
    parser.add_argument("--lambda_reg", type=float, default=0.1, help="Regularization for inner QP")
    parser.add_argument("--num_hyperparam_iterations", type=int, default=4, help="Iterations per hyperparam update loop")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--num_global_updates", type=int, default=5, help="Outer loops")
    parser.add_argument("--num_training_samples", type=int, default=150, help="Number of training samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--distribution", type=str, default='normal', choices=['normal','laplace'], help="Noise distribution")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale param for noise distribution")
    parser.add_argument("--loss_type", type=str, default="mae", choices=['mse','mae'], help="Outer loss type")
    parser.add_argument("--optimizer_choice", type=str, default="adam", choices=['adam','sgd','adamw'], help="Which optimizer to use")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--visualize", action='store_true', help="Whether to visualize final Val Loss curve")
    parser.add_argument("--verbose", action='store_true', help="Verbose logging")

    args = parser.parse_args()
    
    # 根据args处理device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 1) 生成数据
    X, y, beta_true = generate_full_data(args.total_sample_size,
                                         args.feature_dimension,
                                         distribution=args.distribution,
                                         scale=args.scale,
                                         seed=args.seed,
                                         device=device)

    # 2) 切分train/val
    X_train, y_train, X_val, y_val = split_train_val(X, y, args.num_training_samples)

    # 3) 初始化超参数
    U = torch.randn(args.L, device=device, requires_grad=True)
    V = torch.randn(args.L, device=device, requires_grad=True)
    S = torch.randn(args.H, device=device, requires_grad=True)
    T = torch.randn(args.H, device=device, requires_grad=True)
    tau = torch.ones(args.H, device=device, requires_grad=False)

    # 多轮外层更新
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
        beta_autoloss = solve_inner_qpth(U, V, S, T, tau, X_train, y_train, args.lambda_reg)
        all_val_losses.append(val_loss_hist)

        if args.verbose:
            diff = beta_true - beta_autoloss
            print(f"> Beta difference: {diff.detach().cpu().numpy()}")
    
    # 打包结果
    autoloss_result = {
        "U": U.detach().clone(),
        "V": V.detach().clone(),
        "S": S.detach().clone(),
        "T": T.detach().clone(),
        "beta_opt": beta_autoloss.detach().clone(),
        "tau": tau,
        "lambda_reg": args.lambda_reg,
        "data_distribution": args.distribution,
        "scale": args.scale,
        "metric": args.loss_type,
        "optimizer_choice": args.optimizer_choice
    }

    

    # 评估
    if args.verbose:
    
        # 2. Calculate beta comparison metrics
        beta_reg_l2 = train_reg_l2(X_train, y_train)
        beta_reg_l1 = train_reg_l1(X_train, y_train, lr=args.lr, max_iter=1000)
        
        # 3. Print beta comparison table
        print("\n----- Beta Comparison -----")
        print(f"{'Method':<12} {'Beta MSE':<12} {'Beta MAE':<12}")
        print("-" * 36)
        for name, beta in [
            ("AutoLoss", beta_autoloss),
            ("L2 Reg", beta_reg_l2),
            ("L1 Reg", beta_reg_l1)
        ]:
            beta_mse, beta_mae = calc_beta_metrics(beta, beta_true)
            print(f"{name:<12} {beta_mse:<12.6f} {beta_mae:<12.6f}")
        
        # 4. Define functions to calculate prediction metrics

        
        # 5. Evaluate predictions on train data
        print("\n----- Training Data Evaluation -----")
        print(f"{'Method':<12} {'MSE':<12} {'MAE':<12}")
        print("-" * 36)
        for name, beta in [
            ("AutoLoss", beta_autoloss),
            ("L2 Reg", beta_reg_l2),
            ("L1 Reg", beta_reg_l1)
        ]:
            mse, mae = calc_pred_metrics(X_train, y_train, beta)
            print(f"{name:<12} {mse:<12.6f} {mae:<12.6f}")
        
        # 6. Evaluate predictions on validation data
        print("\n----- Validation Data Evaluation -----")
        print(f"{'Method':<12} {'MSE':<12} {'MAE':<12}")
        print("-" * 36)
        for name, beta in [
            ("AutoLoss", beta_autoloss),
            ("L2 Reg", beta_reg_l2),
            ("L1 Reg", beta_reg_l1)
        ]:
            mse, mae = calc_pred_metrics(X_val, y_val, beta)
            print(f"{name:<12} {mse:<12.6f} {mae:<12.6f}")
        print("\n")
# 可视化 Val loss
    val_losses_flat = [v for iteration in all_val_losses for v in iteration]
    if args.visualize:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.plot(val_losses_flat, label="Val Loss")
        plt.xlabel("Iteration")
        plt.ylabel(f"{args.loss_type.upper()} Loss")
        plt.title(f"Val Loss Curve ({args.distribution}, {args.loss_type}, Optim={args.optimizer_choice})")
        plt.grid(True)
        plt.legend()
        plt.show()

        pkl_path, txt_path = save_experiment_results(
            autoloss_result, args, beta_autoloss, 
            U, V, S, T, tau, beta_true, 
            all_val_losses, X_train, y_train, X_val, y_val
        )

if __name__ == "__main__":
    main()
