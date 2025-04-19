# main.py

import argparse
import pickle
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 导入各模块 - 移除不存在的split_train_val函数
from data_utils import generate_full_data, generate_test_data, train_val_sample
from model_utils import solve_inner_qpth
from training import train_hyperparams
from evaluation import evaluate_and_print, compute_test_Xbeta, train_reg_l1, train_reg_l2, calc_beta_metrics, calc_pred_metrics
from theoretical_loss import plot_theoretical_autoloss, plot_combined_visualization
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
    parser.add_argument("--train_ratio", type=float, default=0.75, help="Proportion of data used for training")
    parser.add_argument("--val_ratio", type=float, default=0.25, help="Proportion of data used for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--distribution", type=str, default='normal', choices=['normal','laplace','t'], help="Noise distribution")
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

    # 3) 初始化超参数
    
    U = torch.full((args.L,), -1.0, device=device, requires_grad=True)
    V = torch.full((args.L,), 10.0, device=device, requires_grad=True)
    S = torch.full((args.H,), -1.0, device=device, requires_grad=True)
    T = torch.full((args.H,), 20.0, device=device, requires_grad=True)

    tau = torch.ones(args.H, device=device, requires_grad=False) 

    # 多轮外层更新
    all_val_losses = []

    # 初始化优化器 - 移到外循环
    if args.optimizer_choice == "adam":
        optimizer = torch.optim.Adam([U, V, S, T], lr=args.lr)
    elif args.optimizer_choice == "sgd":
        optimizer = torch.optim.SGD([U, V, S, T], lr=args.lr)
    elif args.optimizer_choice == "adamw":
        optimizer = torch.optim.AdamW([U, V, S, T], lr=args.lr)
    else:
        raise ValueError(f"不支持的优化器选择: {args.optimizer_choice}.")
    
    # 创建每次运行的唯一输出目录（单一目录，不按全局迭代分隔）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), 'theory_loss_plots', f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算训练集和验证集的实际样本数量
    train_size = int(args.total_sample_size * args.train_ratio)
    val_size = int(args.total_sample_size * args.val_ratio)
                
    # 确保样本总数不超过总体样本量
    if train_size + val_size > args.total_sample_size:
        val_size = args.total_sample_size - train_size

    _,_, X_val2, y_val2 = train_val_sample(
        X=X, y=y,
        train_size=0,
        val_size=500,
        seed=args.seed-10,
        device=device
    )

    for it in range(args.num_global_updates):
        if args.verbose:
            print(f"\nGlobal iteration {it+1}/{args.num_global_updates} ...")
        
        # 1) 生成训练和验证数据
        X_train, y_train, X_val, y_val = train_val_sample(  
            X=X, y=y,
            train_size=train_size,
            val_size=val_size,
            seed=args.seed,
            device=device
        )

        U, V, S, T, val_loss_hist, beta_autoloss = train_hyperparams(
            X_train, y_train,
            X_val, y_val,
            X_val2, y_val2,
            U, V, S, T, tau,
            lambda_reg=args.lambda_reg,
            optimizer=optimizer,  # 传递优化器而不是创建新的
            num_hyperparam_iterations=args.num_hyperparam_iterations,
            loss_type=args.loss_type,
            output_dir=output_dir,  # 使用单一目录
            global_iter=it
        )
          
        all_val_losses.append(val_loss_hist)
        
        # 在全局迭代结束时生成总结可视化
        params = {
            "U": U.detach().clone(),
            "V": V.detach().clone(),
            "S": S.detach().clone(),
            "T": T.detach().clone(),
            "tau": tau
        }
        
        # 绘制全局迭代总结图
        plot_combined_visualization(
            params, 
            r_min=-10, 
            r_max=10, 
            num_points=200,
            global_iter=it,
            hyper_iter=0,  
            output_dir=output_dir
        )

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
        beta_reg_l2 = train_reg_l2(X, y)
        beta_reg_l1 = train_reg_l1(X, y, lr=args.lr, max_iter=1000, tol=1e-4, weight_decay=0.0)

        # 3. Print beta comparison table
        print("\n----- Beta Comparison -----")
        print(f"{'Method':<12} {'Beta MSE':<12} {'Beta MAE':<12}")
        print("-" * 36)
        
        # 创建用于保存的指标字典
        beta_metrics_dict = {}
        train_metrics_dict = {}
        val_metrics_dict = {}
        test_metrics_dict = {}
        
        for name, beta in [
            ("AutoLoss", beta_autoloss),
            ("MSE Regression", beta_reg_l2),
            ("MAE Regression", beta_reg_l1)
        ]:
            # 计算并打印beta评估指标
            beta_mse, beta_mae = calc_beta_metrics(beta, beta_true)
            print(f"{name:<12} {beta_mse:<12.6f} {beta_mae:<12.6f}")
            
            # 保存到字典
            beta_metrics_dict[name] = {'mse': beta_mse, 'mae': beta_mae}
        
        # 5. Evaluate predictions on train data
        print("\n----- Training Data Evaluation -----")
        print(f"{'Method':<12} {'MSE':<12} {'MAE':<12}")
        print("-" * 36)
        for name, beta in [
            ("AutoLoss", beta_autoloss),
            ("MSE Regression", beta_reg_l2),
            ("MAE Regression", beta_reg_l1)
        ]:
            # 计算并打印训练集评估指标
            mse, mae = calc_pred_metrics(X_train, y_train, beta)
            print(f"{name:<12} {mse:<12.6f} {mae:<12.6f}")
            
            # 保存到字典
            train_metrics_dict[name] = {'mse': mse, 'mae': mae}
        
        # 6. Evaluate predictions on validation data
        print("\n----- Validation Data Evaluation -----")
        print(f"{'Method':<12} {'MSE':<12} {'MAE':<12}")
        print("-" * 36)
        for name, beta in [
            ("AutoLoss", beta_autoloss),
            ("MSE Regression", beta_reg_l2),
            ("MAE Regression", beta_reg_l1)
        ]:
            # 计算并打印验证集评估指标
            mse, mae = calc_pred_metrics(X_val2, y_val2, beta)
            print(f"{name:<12} {mse:<12.6f} {mae:<12.6f}")
            
            # 保存到字典
            val_metrics_dict[name] = {'mse': mse, 'mae': mae}

        # 7. Evaluate predictions on test data
        print("\n----- Test Data Evaluation -----")
        print(f"{'Method':<12} {'MSE':<12} {'MAE':<12}")
        print("-" * 36)
        # 生成测试数据
        X_test, y_test= generate_test_data(
            num_test_sample=1000,
            feature_dimension=args.feature_dimension,
            beta_true=beta_true,
            distribution=args.distribution,
            scale=args.scale,
            seed=args.seed,
            device=device
        )
        for name, beta in [
            ("AutoLoss", beta_autoloss),
            ("MSE Regression", beta_reg_l2),
            ("MAE Regression", beta_reg_l1)
        ]:
            # 计算并打印验证集评估指标
            mse, mae = calc_pred_metrics(X_test, y_test, beta)
            print(f"{name:<12} {mse:<12.6f} {mae:<12.6f}")
            
            # 保存到字典
            test_metrics_dict[name] = {'mse': mse, 'mae': mae}
        print("\n")
    else:
        # 如果不是verbose模式，创建空字典
        beta_metrics_dict = None
        train_metrics_dict = None
        val_metrics_dict = None
        test_metrics_dict = None

    # 可视化 Val loss
    val_losses_flat = [v for iteration in all_val_losses for v in iteration]
    if args.visualize:
        plt.figure(figsize=(8,6))
        plt.plot(val_losses_flat, label="Val Loss")
        plt.xlabel("Iteration")
        plt.ylabel(f"{args.loss_type.upper()} Loss")
        plt.title(f"Val Loss Curve ({args.distribution}, {args.loss_type}, Optim={args.optimizer_choice})")
        plt.grid(True)
        plt.legend()
        
        # 创建保存目录
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建文件名，包含参数设置和时间戳
        timestamp = datetime.now().strftime("%m%d_%H%M")
        filename = f"valloss_{args.distribution}_{args.loss_type}_{args.optimizer_choice}_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        
        # 保存图表
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss curve saved to: {filepath}")

    
    
    pkl_path, txt_path = save_experiment_results(
        autoloss_result, args, beta_autoloss, 
        U, V, S, T, tau, beta_true, 
        all_val_losses, beta_metrics_dict, train_metrics_dict, val_metrics_dict, test_metrics_dict,
    )

    try:
        from theoretical_loss import create_autoloss_animation
        
        print("\n正在生成训练过程动画...")
        gif_path = create_autoloss_animation(
            output_dir=output_dir,
            num_global_updates=args.num_global_updates,
            duration=5,  # 每帧持续时间增加到1.5秒（更慢）
            loop=0  # 设置为无限循环播放
        )
        
        if gif_path:
            print(f"动画已成功生成: {gif_path}")
            print("已设置为慢速和无限循环播放模式")
        else:
            print("动画生成失败。请确保已安装必要的库: pip install imageio pillow numpy")
    except ImportError as e:
        print(f"无法导入GIF生成功能: {e}")
        print("请确保theoretical_loss.py中包含create_autoloss_animation函数")
    except Exception as e:
        print(f"生成GIF动画时发生错误: {e}")

if __name__ == "__main__":
    main()