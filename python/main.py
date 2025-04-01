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
from evaluation import evaluate_and_print, train_ols, compute_test_Xbeta
from theoretical_loss import plot_theoretical_autoloss

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
            print(f"Global iteration {it+1}/{args.num_global_updates} ...")

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

    # 最终beta
    beta_opt = solve_inner_qpth(U, V, S, T, tau, X_train, y_train, args.lambda_reg)
    
    # 打包结果
    autoloss_result = {
        "U": U.detach().clone(),
        "V": V.detach().clone(),
        "S": S.detach().clone(),
        "T": T.detach().clone(),
        "beta_opt": beta_opt.detach().clone(),
        "tau": tau,
        "lambda_reg": args.lambda_reg,
        "data_distribution": args.distribution,
        "scale": args.scale,
        "metric": args.loss_type,
        "optimizer_choice": args.optimizer_choice
    }

    # 评估
    if args.verbose:
        print("[*] Final Beta:", beta_opt.detach().cpu().numpy())
        evaluate_and_print(X_train, y_train, beta_opt, beta_true, label="[Train]")
        evaluate_and_print(X_val,   y_val,   beta_opt, beta_true, label="[Val]")

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

    # 保存结果
    results_pkl_dir = os.path.join(os.path.dirname(__file__), 'results_pkl')
    results_txt_dir = os.path.join(os.path.dirname(__file__), 'results_txt')
    os.makedirs(results_pkl_dir, exist_ok=True)
    os.makedirs(results_txt_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%m%d%H%M') 
    base_filename = (
        f'D{args.distribution[0]}' 
        f'M{args.loss_type}'
        f'_L{args.L}'
        f'H{args.H}'
        f'N{args.total_sample_size}'
        f'F{args.feature_dimension}'
        f'T{args.num_training_samples}'
        f'G{args.num_global_updates}'
        f'H{args.num_hyperparam_iterations}'
        f'_{timestamp}'
    )
    
    pkl_path = os.path.join(results_pkl_dir, f'{base_filename}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(autoloss_result, f)
    
    # 保存 txt 文件
    txt_path = os.path.join(results_txt_dir, f'{base_filename}.txt')
    with open(txt_path, 'w') as f:
        f.write(f"AutoLoss Experiment Results\n")
        f.write(f"================{timestamp}================\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"- Distribution: {args.distribution}\n")
        f.write(f"- Loss Type: {args.loss_type}\n")
        f.write(f"- Optimizer: {args.optimizer_choice}\n")
        f.write(f"- Parameters: L={args.L}, H={args.H}\n")
        f.write(f"- Samples: {args.total_sample_size} (train={args.num_training_samples})\n")
        f.write(f"- Features: {args.feature_dimension}\n")
        f.write(f"- Updates: {args.num_global_updates} global, {args.num_hyperparam_iterations} hyper\n\n")
        
        f.write(f"Results:\n")
        f.write(f"- Final Beta: {beta_opt.detach().cpu().numpy()}\n")
        f.write(f"- U: {U.detach().cpu().numpy()}\n")
        f.write(f"- V: {V.detach().cpu().numpy()}\n")
        f.write(f"- S: {S.detach().cpu().numpy()}\n")
        f.write(f"- T: {T.detach().cpu().numpy()}\n")
        f.write(f"- tau: {tau.cpu().numpy()}\n")
        
    print(f"[*] Result saved to results")

    # 后续如有测试/可视化理论 AutoLoss，也可在此调用
    # 例如:
    # from theoretical_loss import plot_theoretical_autoloss
    # plot_theoretical_autoloss(autoloss_result, r_min=-10, r_max=10)

if __name__ == "__main__":
    main()
