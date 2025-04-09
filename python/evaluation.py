# evaluation.py

import torch
import numpy as np
import cvxpy as cp

def calc_beta_metrics(beta, beta_true):
            beta_mse = ((beta - beta_true)**2).mean().item()
            beta_mae = (beta - beta_true).abs().mean().item()
            return beta_mse, beta_mae

def calc_pred_metrics(X, y, beta):
    y_pred = X @ beta
    mse = ((y_pred - y)**2).mean().item()
    mae = (y_pred - y).abs().mean().item()
    return mse, mae

def evaluate_and_print(X_test, y_test, beta_est, beta_true, label="", return_metrics=False):
    """
    评估预测结果并打印/返回指标
    ...
    """
    with torch.no_grad():
        y_pred = X_test @ beta_est
        mse = ((y_pred - y_test)**2).mean().item()
        mae = (y_pred - y_test).abs().mean().item()
        
        metrics = {
            "y_mse": mse,
            "y_mae": mae
        }
        
        if beta_true is not None:
            beta_mse = ((beta_est - beta_true)**2).mean().item()
            beta_mae = (beta_est - beta_true).abs().mean().item()
            metrics.update({
                "beta_mse": beta_mse,
                "beta_mae": beta_mae
            })
        
        if label:
            print(f"{label} MSE(y): {mse:.6f}, MAE(y): {mae:.6f}")
            if beta_true is not None:
                print(f"{label} MSE(beta): {beta_mse:.6f}, MAE(beta): {beta_mae:.6f}")
        
        if return_metrics:
            return metrics


def train_reg_l2(X, y):
    """Train L2-regularized linear regression (analytical solution)."""
    X_t = X.t()
    beta = torch.inverse(X_t @ X) @ X_t @ y
    return beta


def train_reg_l1(X, y, lr=0.01, max_iter=1000, tol=1e-4, weight_decay=0.0):
    """
    通过Adam优化器训练L1损失的线性回归
    
    Args:
        X (torch.Tensor): 特征矩阵，形状为(n_samples, n_features)
        y (torch.Tensor): 目标向量，形状为(n_samples,)
        lr (float): 学习率，默认0.01
        max_iter (int): 最大迭代次数，默认1000
        tol (float): 收敛容差，默认1e-4
        weight_decay (float): L2正则化强度，默认0.0
        
    Returns:
        torch.Tensor: 学习到的回归系数
    """
    # 确保输入在CUDA上并转换为float类型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device).float()
    y = y.to(device).float()
    
    # 初始化回归系数
    beta = torch.zeros(X.shape[1], device=device, requires_grad=True)
    
    # 定义优化器
    optimizer = torch.optim.Adam([beta], lr=lr, weight_decay=weight_decay)
    
    # 训练循环
    for _ in range(max_iter):
        # 保存上一次的beta用于收敛检查
        beta_old = beta.clone().detach()
        
        # 前向传播计算预测值
        y_pred = X @ beta
        
        # 计算L1损失（MAE）
        loss = torch.mean(torch.abs(y - y_pred))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 检查收敛
        if torch.max(torch.abs(beta - beta_old)) < tol:
            break
    
    return beta.detach()  # 返回不需要梯度的tensor

def train_reg_l1_cvxpy(X, y):
    """
    通过Adam优化器训练L1损失的线性回归
    
    Args:
        X (torch.Tensor): 特征矩阵，形状为(n_samples, n_features)
        y (torch.Tensor): 目标向量，形状为(n_samples,)
        lr (float): 学习率
        max_iter (int): 最大迭代次数
        tol (float): 收敛容差
        weight_decay (float): L2正则化强度（如需L1正则化可在损失函数中添加）
        
    Returns:
        torch.Tensor: 学习到的回归系数
    """
    beta = cp.Variable(X.shape[1])
    objective = cp.Minimize(cp.norm1(y - X @ beta))
    prob = cp.Problem(objective)
    prob.solve()
    beta = torch.tensor(beta.value).float()
    return beta


def calc_beta_metrics(beta, beta_true):
    """Calculate MSE and MAE between two beta vectors."""
    beta_mse = ((beta - beta_true)**2).mean().item()
    beta_mae = (beta - beta_true).abs().mean().item()
    return beta_mse, beta_mae


def calc_pred_metrics(X, y, beta):
    """Calculate prediction metrics (MSE, MAE) using a given beta."""
    y_pred = X @ beta
    mse = ((y_pred - y)**2).mean().item()
    mae = (y_pred - y).abs().mean().item()
    return mse, mae


def print_beta_comparison(betas_dict, beta_true):
    """Print a comparison table of different beta coefficients against the true beta."""
    print("\n----- Beta Comparison -----")
    print(f"{'Method':<12} {'Beta MSE':<12} {'Beta MAE':<12}")
    print("-" * 36)
    
    for name, beta in betas_dict.items():
        beta_mse, beta_mae = calc_beta_metrics(beta, beta_true)
        print(f"{name:<12} {beta_mse:<12.6f} {beta_mae:<12.6f}")


def print_prediction_evaluation(X, y, betas_dict, dataset_name=""):
    """Print prediction evaluation metrics for different beta coefficients."""
    print(f"\n----- {dataset_name} Data Evaluation -----")
    print(f"{'Method':<12} {'MSE':<12} {'MAE':<12}")
    print("-" * 36)
    
    for name, beta in betas_dict.items():
        mse, mae = calc_pred_metrics(X, y, beta)
        print(f"{name:<12} {mse:<12.6f} {mae:<12.6f}")


def evaluate_models(X_train, y_train, X_val, y_val, betas_dict, beta_true):
    """Evaluate and print comparisons for various models."""
    # Compare beta coefficients
    print_beta_comparison(betas_dict, beta_true)
    
    # Evaluate on training data
    print_prediction_evaluation(X_train, y_train, betas_dict, "Training")
    
    # Evaluate on validation data
    print_prediction_evaluation(X_val, y_val, betas_dict, "Validation")


# This function can be deprecated as it's replaced by the above functions
def evaluate_and_print(X, y, beta, beta_true, label=None):
    """Legacy function - maintained for backward compatibility."""
    y_pred = X @ beta
    mse_loss = ((y_pred - y)**2).mean().item()
    mae_loss = (y_pred - y).abs().mean().item()
    
    if label:
        print(f"{label} MSE: {mse_loss:.6f}, MAE: {mae_loss:.6f}")
    
    return mse_loss, mae_loss


# This can be deprecated if not used elsewhere
def compute_test_Xbeta(X, beta):
    return X @ beta
