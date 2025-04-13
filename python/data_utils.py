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
    beta_true = torch.randn(d, device=device, dtype=torch.float64) 
    #beta_true = torch.rand(d, device=device, dtype=torch.float64) * 10.0
    X = torch.randn(n, d, device=device, dtype=torch.float64) 

    if distribution.lower() == 'laplace':
        #dist = Laplace(0.0, scale)
        dist = Laplace(torch.tensor(0.0, dtype=torch.float64), torch.tensor(scale, dtype=torch.float64))
    elif distribution.lower() == 'normal':
        #dist = Normal(0.0, scale)
        dist = Normal(torch.tensor(0.0, dtype=torch.float64), torch.tensor(scale, dtype=torch.float64))
    else:
        raise ValueError(f"Unsupported distribution: {distribution}, must be 'laplace' or 'normal'")

    eps = dist.sample((n,)).to(device).double()
    y = X @ beta_true + eps
    return X, y, beta_true

def train_val_sample(train_size, val_size, X, y, seed=42, device="cpu"):
    """
    无放回抽样，先抽取 train_portion 比例的训练样本，再从剩余样本中抽取 val_portion 比例的验证样本
    
    参数:
    train_portion - 训练集占总数据的比例
    val_portion - 验证集占总数据的比例
    X - 特征数据
    y - 标签数据
    seed - 随机种子
    device - 计算设备
    
    返回:
    X_train, y_train - 训练数据
    X_val, y_val - 验证数据
    """
    torch.manual_seed(seed)
    n = X.shape[0]

    # 确保 train_size + val_size <= n
    if train_size + val_size > n:
        raise ValueError("train_size + val_size must be less than or equal to total number of samples.")
    

    # 生成随机排列的索引
    indices = torch.randperm(n)
    
    # 抽取训练集
    train_indices = indices[:train_size]
    X_train = X[train_indices].clone().detach().to(device).double() 
    y_train = y[train_indices].clone().detach().to(device).double() 
    
    # 从剩余数据中抽取验证集
    val_indices = indices[train_size:train_size + val_size]
    X_val = X[val_indices].clone().detach().to(device).double()
    y_val = y[val_indices].clone().detach().to(device).double()
    
    return X_train, y_train, X_val, y_val


# def split_train_val(X, y, n_train):
#     """
#     简单切分前 n_train 为训练, 后面的为验证
#     """
#     X_train, y_train = X[:n_train], y[:n_train]
#     X_val,   y_val   = X[n_train:], y[n_train:]
#     return X_train, y_train, X_val, y_val


def generate_test_data(num_test_sample, feature_dimension, beta_true,
                       distribution='laplace', scale=1.0, seed=42, device="cpu"):
    """
    生成测试集 (X_test, y_test)，与训练数据同样的分布设定。
    """
    torch.manual_seed(seed)
    if distribution.lower() == 'laplace':
        #dist = Laplace(0.0, scale)
        dist = Laplace(torch.tensor(0.0, dtype=torch.float64), torch.tensor(scale, dtype=torch.float64))
    elif distribution.lower() == 'normal':
        #dist = Normal(0.0, scale)
        dist = Normal(torch.tensor(0.0, dtype=torch.float64), torch.tensor(scale, dtype=torch.float64))
    else:
        raise ValueError(f"Unsupported distribution: {distribution}, must be 'laplace' or 'normal'")

    X_test = torch.randn(num_test_sample, feature_dimension, device=device, dtype=torch.float64)
    eps = dist.sample((num_test_sample,)).to(device).double()
    y_test = X_test @ beta_true + eps

    return X_test, y_test
