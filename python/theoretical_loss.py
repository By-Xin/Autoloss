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


def plot_theoretical_autoloss(params, r_min=-10, r_max=10, num_points=200):
    """
    绘制单点 Autoloss 关于残差 r 的分段曲线
    """
    U = params["U"]
    V = params["V"]
    S = params["S"]
    T = params["T"]
    tau = params["tau"]  # 需要在 params 中包含 tau
    device = U.device

    r_vals = torch.linspace(r_min, r_max, steps=num_points, device=device)
    L_vals = single_autoloss(r_vals, U, V, S, T, tau)

    plt.figure(figsize=(8,5))
    plt.plot(r_vals.cpu().numpy(), L_vals.cpu().numpy(), label="Single Autoloss")
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel("Residual r")
    plt.ylabel("Autoloss(r)")
    plt.title("Theoretical Autoloss")
    plt.grid(True)
    plt.legend()
    # plt.show()
    
    # Generate filename with parameters and timestamp
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    param_info = f"L{U}_H{S}"
    #filename = f"TheoryLoss_{param_info}_{timestamp}.png"
    filename = f"TheoryLoss_{timestamp}.png"
    
    # Save the figure
    plt.savefig(filename)
    print(f"Plot saved as: {filename}")

