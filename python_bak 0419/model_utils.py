# model_utils.py

import torch
from qpth.qp import QPFunction

def build_qp_matrices(U, V, S, T, tau, X_train, y_train, lambda_reg):
    """
    构建Q, p, G, h，用于内层QP：
      minimize 0.5 * [beta^T diag(lambda_reg) beta + theta^T theta + sigma^T sigma]
               + tau^T sigma
      subject to pi_li >= ...
    """
    n, d = X_train.shape
    L = U.shape[0]
    H = S.shape[0]

    total_vars = d + L*n + 2*H*n  # [beta, pi, theta, sigma]

    # 对角线 Q
    Q_diag = torch.zeros(total_vars, dtype=X_train.dtype, device=X_train.device)
    Q_diag[:d] = lambda_reg
    Q_diag[d + L*n : d + L*n + H*n] = 1.0
    # Q_diag[d + L*n + H*n:] = 1.0  # sigma

    Q = torch.diag(Q_diag).unsqueeze(0)

    # p 向量
    p = torch.zeros(total_vars, dtype=X_train.dtype, device=X_train.device)
    p[d : d + L*n] = 1.0
    p[d + L*n + H*n:] = tau.repeat(n)  # sigma
    p = p.unsqueeze(0)

    # 不等式约束 Gz <= h
    G_rows = 2*L*n + 2*H*n + d
    G = torch.zeros(G_rows, total_vars, dtype=X_train.dtype, device=X_train.device)
    h_val = torch.zeros(G_rows, dtype=X_train.dtype, device=X_train.device)

    row_idx = 0

    # pi_li >= U_l*(y_i - x_i^T beta) + V_l
    for i in range(n):
        for l in range(L):
            G[row_idx, :d] = -U[l]*X_train[i]
            G[row_idx, d + l*n + i] = -1.0
            h_val[row_idx] = -U[l]*y_train[i] - V[l]
            row_idx += 1

    # pi_li >= 0
    for i in range(n):
        for l in range(L):
            G[row_idx, d + l*n + i] = -1.0
            h_val[row_idx] = 0.0
            row_idx += 1

    # theta_hi + sigma_hi >= S_h*(y_i - x_i^T beta) + T_h
    for i in range(n):
        for h_ in range(H):
            G[row_idx, :d] = -S[h_]*X_train[i]
            G[row_idx, d + L*n + h_*n + i] = -1.0
            G[row_idx, d + L*n + H*n + h_*n + i] = -1.0
            h_val[row_idx] = -S[h_]*y_train[i] - T[h_]
            row_idx += 1

    # sigma_hi >= 0
    for i in range(n):
        for h_ in range(H):
            G[row_idx, d + L*n + H*n + h_*n + i] = -1.0
            h_val[row_idx] = 0.0
            row_idx += 1
    
    # beta_j >= 0
    for j in range(d):
        G[row_idx, j] = -1.0
        h_val[row_idx] = 10000.0
        row_idx += 1
    

    G = G.unsqueeze(0)
    h = h_val.unsqueeze(0)

    # 数值扰动
    eps = 1e-4
    Q = Q + eps * torch.eye(total_vars, dtype=X_train.dtype, device=X_train.device).unsqueeze(0)

    return Q, p, G, h


def solve_inner_qpth(U, V, S, T, tau, X_train, y_train, lambda_reg):
    """
    调用 qpth 求解给定超参数下的内层 QP 问题, 得到最优 beta。
    """
    Q, p, G, h = build_qp_matrices(U, V, S, T, tau, X_train, y_train, lambda_reg)
    z = QPFunction(verbose=False)(
        Q, p, G, h,
        torch.empty(0, device=X_train.device),
        torch.empty(0, device=X_train.device)
    )
    d = X_train.shape[1]
    beta_opt = z[:, :d].squeeze(0)
    return beta_opt
