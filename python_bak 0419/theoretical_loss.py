# theoretical_loss.py

import torch
import matplotlib.pyplot as plt
import datetime  # Add this import for timestamp
import os
import re
from datetime import datetime
import numpy as np

# 尝试导入imageio，如果不存在则提供安装提示
try:
    import imageio
    from PIL import Image
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("警告: imageio或PIL库未安装，无法生成GIF动画。")
    print("请安装所需库: pip install imageio pillow numpy")


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

# +
FIXED_Y_MIN = 0.0
FIXED_Y_MAX = 30.0


def plot_theoretical_autoloss(params, r_min=-10, r_max=10, num_points=200, 
                             global_iter=None, hyper_iter=None, output_dir="theory_loss_plots"):
    """
    绘制单点 Autoloss 关于残差 r 的分段曲线并按迭代次数保存，使用固定坐标轴
    
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
    
    # 使用固定的y轴范围
    plt.ylim(FIXED_Y_MIN, FIXED_Y_MAX)
    
    # 使用固定的x轴范围
    plt.xlim(r_min, r_max)
    
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


# -

def plot_combined_visualization(params, r_min=-10, r_max=10, num_points=200, 
                               global_iter=None, hyper_iter=None, output_dir="theory_loss_plots"):
    """
    将超参数热图和理论损失曲线合并到一个图中，使用固定的坐标轴范围
    
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
    
    # 创建画布，分为上下两部分
    fig = plt.figure(figsize=(12, 10))
    
    # 添加标题
    if global_iter is not None and hyper_iter is not None:
        title = f"AutoLoss Visualization (Global Iter: {global_iter}, Hyper Iter: {hyper_iter})"
    else:
        title = "AutoLoss Visualization"
    fig.suptitle(title, fontsize=16)
    
    # 1. 绘制理论损失曲线 (占据上方70%区域)
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
    r_vals = torch.linspace(r_min, r_max, steps=num_points, device=device)
    L_vals = single_autoloss(r_vals, U, V, S, T, tau)
    ax1.plot(r_vals.cpu().numpy(), L_vals.cpu().numpy(), 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel("Residual r")
    ax1.set_ylabel("Autoloss(r)")
    ax1.set_title("Theoretical AutoLoss Function")
    
    # 设置固定的y轴范围
    ax1.set_ylim(FIXED_Y_MIN, FIXED_Y_MAX)
    
    # 设置固定的x轴范围
    ax1.set_xlim(r_min, r_max)
    
    ax1.grid(True)
    
    # 2. 创建所有参数的GitHub风格热图 (占据下方30%区域)
    ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2)
    
    # 获取所有参数的最大和最小值，用于一致的颜色映射
    all_values = np.concatenate([
        U.cpu().detach().numpy(), 
        V.cpu().detach().numpy(),
        S.cpu().detach().numpy(), 
        T.cpu().detach().numpy()
    ])
    vmin, vmax = np.min(all_values), np.max(all_values)
    
    # 转换参数为numpy数组
    u_data = U.cpu().detach().numpy()
    v_data = V.cpu().detach().numpy()
    s_data = S.cpu().detach().numpy()
    t_data = T.cpu().detach().numpy()
    
    # 参数长度，用于确定热图宽度
    max_len = max(len(u_data), len(v_data), len(s_data), len(t_data))
    
    # 创建统一的二维数组，行代表不同的参数组，列代表参数索引
    # 对于长度不足的参数组，用NaN填充
    param_array = np.full((4, max_len), np.nan)
    param_array[0, :len(u_data)] = u_data
    param_array[1, :len(v_data)] = v_data
    param_array[2, :len(s_data)] = s_data
    param_array[3, :len(t_data)] = t_data
    
    # 创建热图
    im = ax2.imshow(param_array, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Parameter Value')
    
    # 设置y轴标签为参数组名称
    ax2.set_yticks(np.arange(4))
    ax2.set_yticklabels(['U', 'V', 'S', 'T'])
    
    # X轴标签（参数索引）
    if max_len <= 10:  # 当参数不多时显示所有索引
        ax2.set_xticks(np.arange(max_len))
        ax2.set_xticklabels(np.arange(max_len))
    else:  # 参数过多时只显示部分索引
        step = max(1, max_len // 10)
        ax2.set_xticks(np.arange(0, max_len, step))
        ax2.set_xticklabels(np.arange(0, max_len, step))
    
    ax2.set_xlabel("Parameter Index")
    ax2.set_title("Parameters Visualization (GitHub-style)")
    
    # 在每个单元格中绘制网格线
    for i in range(1, 4):
        ax2.axhline(y=i-0.5, color='white', linestyle='-', linewidth=0.5)
    
    for j in range(1, max_len):
        ax2.axvline(x=j-0.5, color='white', linestyle='-', linewidth=0.5)
    
    # 为NaN值区域绘制灰色背景
    for i in range(4):
        for j in range(max_len):
            if np.isnan(param_array[i, j]):
                ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='lightgrey'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局以适应标题
    timestamp = datetime.now().strftime("%Y%m%d")

    # 构建迭代信息部分
    iter_info = ""
    if global_iter is not None:
        iter_info += f"Global{global_iter}"
        
        # 对超参数迭代进行正确处理
        if hyper_iter is not None:
            iter_info += f"_Hyper{hyper_iter}"
        else:
            iter_info += "_Hyper0"  # 如果为None，使用Hyper0表示初始状态
            
    # 包含参数大小信息
    param_info = f"_L{len(U)}_H{len(S)}"

    # 构建完整文件名
    filename = f"AutoLoss_{iter_info}{param_info}_{timestamp}.png"

    filepath = os.path.join(output_dir, filename)
        
    # 保存图像
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def create_gif_from_pngs(input_dir='.', output_gif='AutoLoss_Animation.gif', duration=1.5, pattern="AutoLoss_Global", loop=0):
    """
    将指定目录下的PNG图片合成为GIF动画
    
    参数:
        input_dir: 包含PNG图片的目录
        output_gif: 输出GIF文件的路径
        duration: 每一帧的持续时间（秒），值越大动画越慢
        pattern: 匹配文件名的模式
        loop: 循环次数，0表示无限循环，1表示播放一次不循环，n表示循环n次
    
    返回:
        bool: 是否成功创建GIF
    """
    if not HAS_IMAGEIO:
        print("警告: imageio或PIL库未安装，无法生成GIF动画。")
        print("请安装所需库: pip install imageio pillow numpy")
        return False
    
    # 查找目录中所有符合pattern的PNG文件
    png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png') and pattern in f]
    
    if len(png_files) == 0:
        print(f"在 {input_dir} 目录中没有找到匹配 '{pattern}' 的PNG文件")
        return False
    
    # 自定义排序函数，按照Global和Hyper的数字顺序排序
    def extract_numbers(filename):
        # 从文件名中提取Global和Hyper的数字
        global_match = re.search(r'Global(\d+)', filename)
        hyper_match = re.search(r'Hyper(\d+)', filename)
        
        global_num = int(global_match.group(1)) if global_match else 0
        hyper_num = int(hyper_match.group(1)) if hyper_match else 0
        
        return (global_num, hyper_num)
    
    # 根据Global和Hyper的数字顺序排序文件
    png_files.sort(key=extract_numbers)
    
    # 打印排序后的文件列表（用于验证）
    print("将按以下顺序处理图片：")
    for i, f in enumerate(png_files):
        global_num, hyper_num = extract_numbers(f)
        print(f"{i+1}. {f} (Global: {global_num}, Hyper: {hyper_num})")
    
    # 读取所有图片
    images = []
    for png_file in png_files:
        file_path = os.path.join(input_dir, png_file)
        try:
            img = Image.open(file_path)
            # 确保所有图片都是RGB模式（而不是RGBA）
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            images.append(np.array(img))
        except Exception as e:
            print(f"无法处理文件 {png_file}: {e}")
    
    if len(images) == 0:
        print("没有有效的图片可以处理")
        return False
    
    # 创建GIF，设置循环参数和更长的帧持续时间
    print(f"创建GIF动画，共 {len(images)} 帧，帧持续时间: {duration}秒，循环模式: {'无限循环' if loop == 0 else f'循环{loop}次'}")
    imageio.mimsave(output_gif, images, duration=duration, loop=loop)
    print(f"GIF动画已保存到: {output_gif}")
    
    # 输出文件大小信息
    file_size = os.path.getsize(output_gif) / (1024 * 1024)  # 转换为MB
    print(f"文件大小: {file_size:.2f} MB")
    
    return True

def create_autoloss_animation(output_dir, num_global_updates=None, output_gif=None, duration=1.5, loop=0):
    """
    自动生成AutoLoss训练过程的GIF动画
    
    Args:
        output_dir: 包含PNG图片的目录
        num_global_updates: 全局更新次数（如果提供，用于生成更具体的文件名）
        output_gif: 输出GIF文件的名称（如果为None，将自动生成）
        duration: 每一帧的持续时间（秒），值越大动画越慢
        loop: 循环次数，0表示无限循环，1表示播放一次不循环，n表示循环n次
    
    Returns:
        str: 生成的GIF文件路径，或者None（如果失败）
    """
    if not HAS_IMAGEIO:
        print("警告: 缺少必要的库，无法生成GIF动画。")
        return None
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有指定输出文件名，创建一个包含时间戳的名称
    if output_gif is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if num_global_updates is not None:
            output_gif = f"AutoLoss_G{num_global_updates}_Animation_{timestamp}.gif"
        else:
            output_gif = f"AutoLoss_Animation_{timestamp}.gif"
    
    # 完整路径
    output_path = os.path.join(output_dir, output_gif)
    
    # 生成GIF动画
    success = create_gif_from_pngs(
        input_dir=output_dir,
        output_gif=output_path,
        duration=duration,
        pattern="AutoLoss_Global",
        loop=loop
    )
    
    if success:
        return output_path
    else:
        return None
