import os
import pickle
import torch
from datetime import datetime

def save_experiment_results(autoloss_result, args, beta_opt, U, V, S, T, tau, beta_true, 
                           all_val_losses, X_train, y_train, X_val, y_val):
    """
    保存实验结果到pkl和txt文件
    
    Args:
        autoloss_result (dict): 实验结果字典
        args: 命令行参数
        beta_opt, U, V, S, T, tau: 模型参数
        beta_true: 真实beta值
        all_val_losses: 验证损失历史
        X_train, y_train, X_val, y_val: 训练和验证数据
        
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
    base_filename = (
        f'D{args.distribution[0]}'
        f'M{args.loss_type}'
        f'L{args.L}'
        f'H{args.H}'
        f'N{args.total_sample_size}'
        f'F{args.feature_dimension}'
        f'T{args.num_training_samples}'
        f'G{args.num_global_updates}'
        f'H{args.num_hyperparam_iterations}'
        f'{timestamp}'
    )
    
    # 计算训练和验证指标
    with torch.no_grad():
        train_metrics = evaluate_and_print(X_train, y_train, beta_opt, beta_true, label="", return_metrics=True)
        val_metrics = evaluate_and_print(X_val, y_val, beta_opt, beta_true, label="", return_metrics=True)
    
    # 更新结果字典
    autoloss_result.update({
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "all_val_losses": all_val_losses,
        "timestamp": timestamp
    })
    
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
        f.write(f"- Distribution: {args.distribution}\n")
        f.write(f"- Loss Type: {args.loss_type}\n")
        f.write(f"- Optimizer: {args.optimizer_choice}\n")
        f.write(f"- Parameters: L={args.L}, H={args.H}\n")
        f.write(f"- Samples: {args.total_sample_size} (train={args.num_training_samples})\n")
        f.write(f"- Features: {args.feature_dimension}\n")
        f.write(f"- Updates: {args.num_global_updates} global, {args.num_hyperparam_iterations} hyper\n\n")
        
        # 2. 模型参数
        f.write(f"Model Parameters:\n")
        f.write(f"----------------\n")
        f.write(f"- Final Beta: {beta_opt.detach().cpu().numpy()}\n")
        f.write(f"- U: {U.detach().cpu().numpy()}\n")
        f.write(f"- V: {V.detach().cpu().numpy()}\n")
        f.write(f"- S: {S.detach().cpu().numpy()}\n")
        f.write(f"- T: {T.detach().cpu().numpy()}\n")
        f.write(f"- tau: {tau.cpu().numpy()}\n\n")
        
        # 3. 评估指标
        f.write(f"Evaluation Metrics:\n")
        f.write(f"-------------------\n")
        f.write(f"Training Set:\n")
        for metric, value in train_metrics.items():
            f.write(f"  {metric}: {value:.6f}\n")
        
        f.write(f"\nValidation Set:\n")
        for metric, value in val_metrics.items():
            f.write(f"  {metric}: {value:.6f}\n\n")
        
        # 4. 训练历史 (放在最后)
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