# theoretical_loss.py

import torch
import matplotlib.pyplot as plt
import datetime  # Add this import for timestamp



def reHU_piecewise(x, gamma):
    """
    ReHU_gamma(x):
     = 0                  if x <= 0
     = 0.5*x^2            if 0 < x < gamma
     = x^2/4 + gamma*x/2 - gamma^2/4   if x >= gamma
    """
    cond0 = (x <= 0)
    cond2 = (x >= gamma)

    val0 = torch.zeros_like(x)
    val1 = 0.5 * x**2
    val2 = (x**2)/4 + 0.5*gamma*x - (gamma**2)/4

    out = torch.where(cond0, val0, val1)
    out = torch.where(cond2, val2, out)
    return out


def single_autoloss(r, U, V, S, T, tau):
    """
    对单个标量残差 r 计算:
      sum_{l=1}^L ReLU(U_l*r + V_l)
      + sum_{h=1}^H ReHU_{tau_h}(S_h*r + T_h)
    """
    if not isinstance(r, torch.Tensor):
        r = torch.tensor(r, dtype=U.dtype, device=U.device)
    if r.dim() == 0:
        r = r.view(1)

    # ReLU 部分
    uv = U.unsqueeze(1)*r + V.unsqueeze(1)  # [L,N]
    partL = torch.relu(uv).sum(dim=0)       # sum over L

    # ReHU 部分
    st = S.unsqueeze(1)*r + T.unsqueeze(1)  # [H,N]
    tau_expand = tau.unsqueeze(1).expand_as(st)
    partH = reHU_piecewise(st, tau_expand).sum(dim=0)

    return partL + partH


def plot_hyperparams_heatmap(U, V, S, T, global_iter=None, hyper_iter=None, output_dir="theory_loss_plots"):
    """
    将超参数 U, V, S, T 绘制为热图
    
    Args:
        U, V, S, T: 超参数向量
        global_iter: 外层迭代次数
        hyper_iter: 内层超参数迭代次数
        output_dir: 输出目录
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    
    # 创建子图布局
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Hyperparameters Visualization", fontsize=16)
    
    # 如果有迭代信息，添加到标题
    if global_iter is not None and hyper_iter is not None:
        fig.suptitle(f"Hyperparameters (Global Iter: {global_iter}, Hyper Iter: {hyper_iter})", fontsize=16)
    
    # 获取所有参数的最大和最小值，用于一致的颜色映射
    all_values = np.concatenate([
        U.cpu().detach().numpy(), 
        V.cpu().detach().numpy(),
        S.cpu().detach().numpy(), 
        T.cpu().detach().numpy()
    ])
    vmin, vmax = np.min(all_values), np.max(all_values)
    
    # 绘制U
    u_data = U.cpu().detach().numpy().reshape(-1, 1)
    im0 = axs[0, 0].imshow(u_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title(f"U (size={len(U)})")
    axs[0, 0].set_xlabel("Dimension")
    axs[0, 0].set_ylabel("Parameter Index")
    
    # 绘制V
    v_data = V.cpu().detach().numpy().reshape(-1, 1)
    im1 = axs[0, 1].imshow(v_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title(f"V (size={len(V)})")
    axs[0, 1].set_xlabel("Dimension")
    
    # 绘制S
    s_data = S.cpu().detach().numpy().reshape(-1, 1)
    im2 = axs[1, 0].imshow(s_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title(f"S (size={len(S)})")
    axs[1, 0].set_xlabel("Dimension")
    axs[1, 0].set_ylabel("Parameter Index")
    
    # 绘制T
    t_data = T.cpu().detach().numpy().reshape(-1, 1)
    im3 = axs[1, 1].imshow(t_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axs[1, 1].set_title(f"T (size={len(T)})")
    axs[1, 1].set_xlabel("Dimension")
    
    # 添加颜色条
    cbar = fig.colorbar(im0, ax=axs.ravel().tolist())
    cbar.set_label('Parameter Value')
    
    plt.tight_layout()
    
    # 生成带有参数和迭代信息的文件名
    timestamp = datetime.now().strftime("%Y%m%d")
    
    iter_info = ""
    if global_iter is not None:
        iter_info += f"_G{global_iter}"
    if hyper_iter is not None:
        iter_info += f"_H{hyper_iter}"
    
    filename = f"HyperparamHeatmap{iter_info}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    # 保存图像
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    return filepath

def plot_theoretical_autoloss(params, r_min=-10, r_max=10, num_points=200, 
                             global_iter=None, hyper_iter=None, output_dir="theory_loss_plots"):
    """
    绘制单点 Autoloss 关于残差 r 的分段曲线并按迭代次数保存
    
    Args:
        params: 包含U, V, S, T, tau的字典
        r_min, r_max: 残差范围
        num_points: 绘图点数
        global_iter: 外层迭代次数
        hyper_iter: 内层超参数迭代次数
        output_dir: 输出目录
    """
    import os
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    U = params["U"]
    V = params["V"]
    S = params["S"]
    T = params["T"]
    tau = params["tau"]
    device = U.device
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    r_vals = torch.linspace(r_min, r_max, steps=num_points, device=device)
    L_vals = single_autoloss(r_vals, U, V, S, T, tau)

    plt.figure(figsize=(8,5))
    plt.plot(r_vals.cpu().numpy(), L_vals.cpu().numpy(), label="Single Autoloss")
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel("Residual r")
    plt.ylabel("Autoloss(r)")
    
    # 生成合适的标题
    if global_iter is not None and hyper_iter is not None:
        title = f"Theoretical Autoloss (Global Iter: {global_iter}, Hyper Iter: {hyper_iter})"
    else:
        title = "Theoretical Autoloss"
    plt.title(title)
    
    plt.grid(True)
    plt.legend()
    
    # 生成带有参数和迭代信息的文件名
    timestamp = datetime.now().strftime("%Y%m%d")
    
    iter_info = ""
    if global_iter is not None:
        iter_info += f"_G{global_iter}"
    if hyper_iter is not None:
        iter_info += f"_H{hyper_iter}"
    
    # 包含参数信息
    param_info = f"_L{len(U)}_H{len(S)}"
    filename = f"TheoryLoss{iter_info}{param_info}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    # 保存图像
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    return filepath


# def plot_theoretical_autoloss(params, r_min=-10, r_max=10, num_points=200):
#     """
#     绘制单点 Autoloss 关于残差 r 的分段曲线
#     """
#     U = params["U"]
#     V = params["V"]
#     S = params["S"]
#     T = params["T"]
#     tau = params["tau"]  # 需要在 params 中包含 tau
#     device = U.device

#     r_vals = torch.linspace(r_min, r_max, steps=num_points, device=device)
#     L_vals = single_autoloss(r_vals, U, V, S, T, tau)

#     plt.figure(figsize=(8,5))
#     plt.plot(r_vals.cpu().numpy(), L_vals.cpu().numpy(), label="Single Autoloss")
#     plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
#     plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
#     plt.xlabel("Residual r")
#     plt.ylabel("Autoloss(r)")
#     plt.title("Theoretical Autoloss")
#     plt.grid(True)
#     plt.legend()
#     # plt.show()
    
#     # Generate filename with parameters and timestamp
#     timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
#     param_info = f"L{U}_H{S}"
#     #filename = f"TheoryLoss_{param_info}_{timestamp}.png"
#     filename = f"TheoryLoss_{timestamp}.png"
    
#     # Save the figure
#     plt.savefig(filename)
#     print(f"Plot saved as: {filename}")

def plot_combined_visualization(params, r_min=-10, r_max=10, num_points=200, 
                               global_iter=None, hyper_iter=None, output_dir="theory_loss_plots"):
    """
    将超参数热图和理论损失曲线合并到一个图中
    
    Args:
        params: 包含U, V, S, T, tau的字典
        r_min, r_max: 残差范围
        num_points: 绘图点数
        global_iter: 外层迭代次数
        hyper_iter: 内层超参数迭代次数
        output_dir: 输出目录
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    
    U = params["U"]
    V = params["V"]
    S = params["S"]
    T = params["T"]
    tau = params["tau"]
    device = U.device
    
    # 创建输出目录(如果不存在)
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建一个大的图形，包含5个子图(1个用于损失曲线，4个用于超参数)
    fig = plt.figure(figsize=(15, 10))
    
    # 添加标题
    if global_iter is not None and hyper_iter is not None:
        title = f"AutoLoss Visualization (Global Iter: {global_iter}, Hyper Iter: {hyper_iter})"
    else:
        title = "AutoLoss Visualization"
    fig.suptitle(title, fontsize=16)
    
    # 1. 绘制理论损失曲线 (占据上方区域，跨越所有列)
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    r_vals = torch.linspace(r_min, r_max, steps=num_points, device=device)
    L_vals = single_autoloss(r_vals, U, V, S, T, tau)
    ax1.plot(r_vals.cpu().numpy(), L_vals.cpu().numpy(), 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel("Residual r")
    ax1.set_ylabel("Autoloss(r)")
    ax1.set_title("Theoretical AutoLoss Function")
    ax1.grid(True)
    
    # 获取所有参数的最大和最小值，用于一致的颜色映射
    all_values = np.concatenate([
        U.cpu().detach().numpy(), 
        V.cpu().detach().numpy(),
        S.cpu().detach().numpy(), 
        T.cpu().detach().numpy()
    ])
    vmin, vmax = np.min(all_values), np.max(all_values)
    
    # 2. 绘制U热图
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    u_data = U.cpu().detach().numpy().reshape(-1, 1)
    im2 = ax2.imshow(u_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    ax2.set_title(f"U Parameters (size={len(U)})")
    ax2.set_xlabel("Dimension")
    ax2.set_ylabel("Parameter Index")
    
    # 3. 绘制V热图
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    v_data = V.cpu().detach().numpy().reshape(-1, 1)
    im3 = ax3.imshow(v_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    ax3.set_title(f"V Parameters (size={len(V)})")
    ax3.set_xlabel("Dimension")
    
    # 添加S和T参数的文本描述（如果存在）
    if len(S) > 0:
        s_text = f"S Parameters: {S.cpu().detach().numpy()}"
        t_text = f"T Parameters: {T.cpu().detach().numpy()}"
        plt.figtext(0.5, 0.02, s_text + "\n" + t_text, ha="center", fontsize=10, 
                  bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    # 添加颜色条
    cbar = fig.colorbar(im2, ax=[ax2, ax3])
    cbar.set_label('Parameter Value')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以适应文本
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # 将全局迭代和超参数迭代信息添加到文件名
    iter_info = ""
    if global_iter is not None:
        iter_info += f"G{global_iter}"
    if hyper_iter is not None:
        iter_info += f"_H{hyper_iter}"
    
    # 包含参数大小信息
    param_info = f"_L{len(U)}_H{len(S)}"
    
    filename = f"AutoLoss_{iter_info}{param_info}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    # 保存图像
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath