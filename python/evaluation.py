# evaluation.py

import torch

def evaluate_and_print(X_test, y_test, beta_est, beta_true, label="", return_metrics=False):
    """
    评估预测结果并打印/返回指标
    
    Args:
        X_test, y_test: 测试数据
        beta_est: 估计的beta系数
        beta_true: 真实的beta系数
        label: 打印前缀
        return_metrics: 是否返回指标字典
    
    Returns:
        dict 或 None: 如果return_metrics=True，返回包含评估指标的字典
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


def train_ols(X_train, y_train):
    """
    简单最小二乘回归: beta = (X^T X)^(-1) X^T y
    返回beta估计
    """
    beta_ols = torch.linalg.lstsq(X_train, y_train).solution
    return beta_ols


def compute_test_Xbeta(X_test, y_test, beta_est, beta_true=None):
    """
    在测试集(X_test, y_test)上计算预测误差以及与beta_true之间的误差.
    """
    with torch.no_grad():
        y_pred = X_test @ beta_est
        y_mse = ((y_pred - y_test)**2).mean().item()
        y_mae = (y_pred - y_test).abs().mean().item()
        beta_mse = beta_mae = None
        if beta_true is not None:
            beta_mse = ((beta_est - beta_true)**2).mean().item()
            beta_mae = (beta_est - beta_true).abs().mean().item()

    return y_mse, y_mae, beta_mse, beta_mae
