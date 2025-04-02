# evaluation.py

import torch
import numpy as np

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


def train_reg_l1(X, y, lr=0.01, max_iter=1000, tol=1e-6):
    """Train L1-regularized linear regression using proximal gradient descent."""
    n, p = X.shape
    beta = torch.zeros(p, device=X.device)
    
    for i in range(max_iter):
        grad = -2 * X.t() @ (y - X @ beta) / n
        beta_new = beta - lr * grad
        
        # Apply soft thresholding (proximal operator for L1)
        beta_new = torch.sign(beta_new) * torch.clamp(torch.abs(beta_new) - lr, min=0)
        
        if torch.norm(beta_new - beta) < tol:
            break
        
        beta = beta_new
    
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
