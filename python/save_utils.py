import os
import pickle
import torch
from datetime import datetime

def save_experiment_results(autoloss_result, args, beta_opt, U, V, S, T, tau, beta_true, 
                           all_val_losses, beta_metrics=None, train_metrics=None, val_metrics=None,test_metrics=None):
    """
    保存实验结果到pkl和txt文件
    
    Args:
        autoloss_result (dict): 实验结果字典
        args: 命令行参数
        beta_opt, U, V, S, T, tau: 模型参数
        beta_true: 真实beta值
        all_val_losses: 验证损失历史
        beta_metrics: 各方法beta估计准确性指标
        train_metrics: 各方法在训练集上的预测指标 
        val_metrics: 各方法在验证集上的预测指标
        
    Returns:
        tuple: (pkl_path, txt_path) 保存的文件路径
    """
    # 导入评估函数 (放在这里避免循环导入)
    from evaluation import evaluate_and_print
    
    # 创建目录
    results_pkl_dir = os.path.join(os.path.dirname(__file__), 'results_pkl')
    results_txt_dir = os.path.join(os.path.dirname(__file__), 'results_txt')
    os.makedirs(results_pkl_dir, exist_ok=True)
    os.makedirs(results_txt_dir, exist_ok=True)

    # 生成时间戳和文件名
    timestamp = datetime.now().strftime('%m%d%H%M')
    
    # 安全获取参数，避免缺少参数导致的错误
    try:
        dist_char = args.distribution[0] if hasattr(args, 'distribution') and args.distribution else 'X'
    except (IndexError, TypeError):
        dist_char = 'X'
    
    base_filename = (
        f'D{dist_char}'
        f'{timestamp}'
        f'M{getattr(args, "loss_type", "X")}'
        f'L{getattr(args, "L", 0)}'
        f'H{getattr(args, "H", 0)}'
        f'N{getattr(args, "total_sample_size", 0)}'
        f'F{getattr(args, "feature_dimension", 0)}'
        f'TR{getattr(args, "train_ratio", 0)}'
        f'VR{getattr(args, "val_ratio", 0)}'
        f'G{getattr(args, "num_global_updates", 0)}'
        f'H{getattr(args, "num_hyperparam_iterations", 0)}'
        #f'{timestamp}'
    )
    
    # 更新结果字典
    autoloss_result.update({
        "all_val_losses": all_val_losses,
        "timestamp": timestamp
    })
    
    # 添加评估指标（如果提供）
    if beta_metrics is not None:
        autoloss_result["beta_metrics"] = beta_metrics
    if train_metrics is not None:
        autoloss_result["train_metrics"] = train_metrics
    if val_metrics is not None:
        autoloss_result["val_metrics"] = val_metrics
    if test_metrics is not None:
        autoloss_result["test_metrics"] = test_metrics
    
    # 保存 pkl 文件
    pkl_path = os.path.join(results_pkl_dir, f'{base_filename}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(autoloss_result, f)
    
    # 保存 txt 文件
    txt_path = os.path.join(results_txt_dir, f'{base_filename}.txt')
    with open(txt_path, 'w') as f:
        # 1. 标题和配置
        f.write(f"AutoLoss Experiment Results\n")
        f.write(f"================{timestamp}================\n\n")
        f.write(f"Configuration:\n")
        f.write(f"- Distribution: {getattr(args, 'distribution', 'N/A')}\n")
        f.write(f"- Loss Type: {getattr(args, 'loss_type', 'N/A')}\n")
        f.write(f"- Optimizer: {getattr(args, 'optimizer_choice', 'N/A')}\n")
        f.write(f"- Parameters: L={getattr(args, 'L', 'N/A')}, H={getattr(args, 'H', 'N/A')}\n")
        f.write(f"- Samples: {getattr(args, 'total_sample_size', 'N/A')} (train={getattr(args, 'num_training_samples', 'N/A')})\n")
        f.write(f"- Features: {getattr(args, 'feature_dimension', 'N/A')}\n")
        f.write(f"- Updates: {getattr(args, 'num_global_updates', 'N/A')} global, {getattr(args, 'num_hyperparam_iterations', 'N/A')} hyper\n\n")
        
        # 2. 模型参数
        f.write(f"Model Parameters:\n")
        f.write(f"----------------\n")
        f.write(f"- Final Beta: {beta_opt.detach().cpu().numpy()}\n")
        f.write(f"- U: {U.detach().cpu().numpy()}\n")
        f.write(f"- V: {V.detach().cpu().numpy()}\n")
        f.write(f"- S: {S.detach().cpu().numpy()}\n")
        f.write(f"- T: {T.detach().cpu().numpy()}\n")
        f.write(f"- tau: {tau.cpu().numpy()}\n\n")
        
        # 3. 评估指标 (如果提供)
        if beta_metrics is not None:
            f.write(f"Beta Comparison Metrics:\n")
            f.write(f"--------------------\n")
            f.write(f"{'Method':<12} {'Beta MSE':<12} {'Beta MAE':<12}\n")
            f.write("-" * 36 + "\n")
            for method, metrics in beta_metrics.items():
                f.write(f"{method:<12} {metrics['mse']:<12.6f} {metrics['mae']:<12.6f}\n")
            f.write("\n")
        
        if train_metrics is not None:
            f.write(f"Training Data Evaluation:\n")
            f.write(f"------------------------\n")
            f.write(f"{'Method':<12} {'MSE':<12} {'MAE':<12}\n")
            f.write("-" * 36 + "\n")
            for method, metrics in train_metrics.items():
                f.write(f"{method:<12} {metrics['mse']:<12.6f} {metrics['mae']:<12.6f}\n")
            f.write("\n")
        
        if val_metrics is not None:
            f.write(f"Validation Data Evaluation:\n")
            f.write(f"--------------------------\n")
            f.write(f"{'Method':<12} {'MSE':<12} {'MAE':<12}\n")
            f.write("-" * 36 + "\n")
            for method, metrics in val_metrics.items():
                f.write(f"{method:<12} {metrics['mse']:<12.6f} {metrics['mae']:<12.6f}\n")
            f.write("\n")
        
        if test_metrics is not None:
            f.write(f"Test Data Evaluation:\n")
            f.write(f"--------------------\n")
            f.write(f"{'Method':<12} {'MSE':<12} {'MAE':<12}\n")
            f.write("-" * 36 + "\n")
            for method, metrics in test_metrics.items():
                f.write(f"{method:<12} {metrics['mse']:<12.6f} {metrics['mae']:<12.6f}\n")
            f.write("\n")
        
        # 4. 训练历史
        f.write(f"Training History:\n")
        f.write(f"----------------\n")
        val_losses_flat = [v for iteration in all_val_losses for v in iteration]
        f.write(f"Validation Loss Overview (flattened): {len(val_losses_flat)} iterations\n")
        for i in range(0, len(val_losses_flat), 5):  # 每行显示5个
            batch = val_losses_flat[i:i+5]
            f.write("  ".join([f"{j+i+1}:{loss:.6f}" for j, loss in enumerate(batch)]) + "\n")
        
        f.write(f"\nDetailed Validation Loss by Global Iteration:\n")
        for global_iter, losses in enumerate(all_val_losses):
            f.write(f"\nGlobal Iteration {global_iter + 1}:\n")
            for hyper_iter, loss in enumerate(losses):
                f.write(f"  Hyper step {hyper_iter + 1}: {loss:.6f}\n")
    
    print(f"\n> Results saved to:")
    print(f"    - PKL: results_pkl/{base_filename}.pkl")
    print(f"    - TXT: results_txt/{base_filename}.txt")
    print(f"-"*100)
    print("\n")
    
    return pkl_path, txt_path