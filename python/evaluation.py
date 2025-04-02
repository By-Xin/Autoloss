# evaluation.py

import torch

# evaluation.py
import torch

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


def train_ols(X_train, y_train):
    """
    简单最小二乘回归: beta = (X^T X)^(-1) X^T y
    返回beta估计
    """
    beta_ols = torch.linalg.lstsq(X_train, y_train).solution
    return beta_ols


def train_mae(X_train, y_train, lr=1e-2, max_iter=1000):
    """
    用梯度下降(Adam)来最小化MAE, 返回训练得到的beta.
    由于MAE没有简单的闭形式解, 只能数值优化.
    """
    d = X_train.shape[1]
    # 初始化beta
    beta = torch.zeros(d, device=X_train.device, requires_grad=True)
    optimizer = torch.optim.Adam([beta], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()
        y_pred = X_train @ beta
        loss = (y_pred - y_train).abs().mean()  # MAE
        loss.backward()
        optimizer.step()

    return beta.detach()



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
