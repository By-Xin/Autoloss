# evaluation.py

import torch

def evaluate_and_print(X, y, beta_est, beta_true=None, label=""):
    """
    计算并打印(y_pred对y)的MSE/MAE，以及如果给了beta_true，也打印beta的MSE/MAE。
    """
    with torch.no_grad():
        y_pred = X @ beta_est
        mse = ((y_pred - y)**2).mean().item()
        mae = (y_pred - y).abs().mean().item()
        print(f"{label} MSE(y): {mse:.6f}, MAE(y): {mae:.6f}")

        if beta_true is not None:
            beta_mse = ((beta_est - beta_true)**2).mean().item()
            beta_mae = (beta_est - beta_true).abs().mean().item()
            print(f"{label} MSE(beta): {beta_mse:.6f}, MAE(beta): {beta_mae:.6f}")


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
