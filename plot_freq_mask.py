import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 载入你训练好的权重
weights_path = "./checkpoints/freqmamba/full_model/freqmamba_best.pth"
state_dict = torch.load(weights_path, map_location='cpu')

# 2. 提取 Bottleneck 中最后一个 FSMB 的频域参数
# 假设 bottleneck.4 是最后一个 FSMB
cutoff = state_dict['bottleneck.4.cutoff'].squeeze().numpy()
sharpness = state_dict['bottleneck.4.sharpness'].squeeze().numpy()

# 提取所有通道的均值，作为可视化的代表
c_mean = np.mean(cutoff)
k_mean = np.mean(sharpness)

print(f"Learned Cutoff Mean: {c_mean:.4f}, Sharpness Mean: {k_mean:.4f}")

# 3. 构造 2D 物理频率网格 (假设展示尺寸为 128x128)
H, W = 128, 128
fy = torch.fft.fftfreq(H)
fx = torch.fft.rfftfreq(W)
fy_mesh, fx_mesh = torch.meshgrid(fy, fx, indexing='ij')
freq_radius = torch.sqrt(fy_mesh**2 + fx_mesh**2).numpy()

# 4. 计算 Butterworth 响应 W = sigmoid(k * (R - c))
W_mask = 1.0 / (1.0 + np.exp(-k_mean * (freq_radius - c_mean)))

# 5. 画图并保存
plt.figure(figsize=(6, 5))
plt.imshow(W_mask, cmap='inferno')
plt.colorbar(label='Attention Weight')
plt.title('Learned Butterworth High-Frequency Prior', fontsize=14, fontweight='bold')
plt.axis('off')

os.makedirs("./paper_figures", exist_ok=True)
plt.savefig("./paper_figures/Fig_Learned_Freq_Mask.pdf", bbox_inches='tight', dpi=300)
print("✅ 频域先验可视化图已保存至 ./paper_figures/Fig_Learned_Freq_Mask.pdf")