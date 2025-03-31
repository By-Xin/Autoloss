# data_utils.py

import torch
from torch.distributions import Laplace, Normal

def generate_full_data(n, d, distribution='laplace', scale=1.0, seed=42, device="cpu"):
    """
    生成模拟数据 (X, y, beta_true)，其中:
      - beta_true ~ Uniform(0, 10)
      - X ~ N(0,1)
      - eps 来自指定分布 (laplace或normal)
      - y = X beta_true + eps
    """
    torch.manual_seed(seed)
    beta_true = torch.rand(d, device=device) * 10.0
    X = torch.randn(n, d, device=device)

    if distribution.lower() == 'laplace':
        dist = Laplace(0.0, scale)
    elif distribution.lower() == 'normal':
        dist = Normal(0.0, scale)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}, must be 'laplace' or 'normal'")

    eps = dist.sample((n,)).to(device)
    y = X @ beta_true + eps
    return X, y, beta_true


def split_train_val(X, y, n_train):
    """
    简单切分前 n_train 为训练, 后面的为验证
    """
    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:], y[n_train:]
    return X_train, y_train, X_val, y_val


def generate_test_data(num_test_sample, feature_dimension, beta_true,
                       distribution='laplace', scale=1.0, seed=42, device="cpu"):
    """
    生成测试集 (X_test, y_test)，与训练数据同样的分布设定。
    """
    torch.manual_seed(seed)
    if distribution.lower() == 'laplace':
        dist = Laplace(0.0, scale)
    elif distribution.lower() == 'normal':
        dist = Normal(0.0, scale)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}, must be 'laplace' or 'normal'")

    X_test = torch.randn(num_test_sample, feature_dimension, device=device)
    eps = dist.sample((num_test_sample,)).to(device)
    y_test = X_test @ beta_true + eps

    return X_test, y_test
