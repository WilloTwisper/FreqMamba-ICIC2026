import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================= Config =========================
VIS_DIR = "./visualizations"
OUT_DIR = "./paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ['DRIVE', 'HRF', 'CHASE']

# 注意这里的顺序：越来越强的模型，最后是 FreqMamba 和 GT
METHODS = ['in', 'clahe', 'msrcr', 'unet', 'nafnet', 'ffcnet', 'restormer', 'freq', 'gt']
METHOD_NAMES = [
    'Degraded Input', 'CLAHE', 'MSRCR', 'U-Net', 
    'NAFNet', 'FFCUNet', 'Restormer', 'FreqMamba', 'Ground Truth'
]

# ========================= Helper =========================
def load_img(path):
    if not os.path.exists(path):
        print(f"[Warning] Image not found: {path}")
        # 如果找不到，返回一张白底图占位，防止报错
        return np.ones((256, 384, 3), dtype=np.uint8) * 255
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ========================= Figure 1: Enhancement + Spectrum =========================
def compose_enhancement_spectrum_figure():
    print("Generating Figure 1: Enhancement & Spectrum Comparison...")
    
    # 3个数据集，每个数据集 2 行（RGB + 频谱），共 6 行。9列。
    fig = plt.figure(figsize=(24, 13))
    gs = fig.add_gridspec(6, 9, wspace=0.02, hspace=0.05)
    
    for row_idx, dataset in enumerate(DATASETS):
        base_row = row_idx * 2
        
        for col_idx, method in enumerate(METHODS):
            # 加载 RGB 图和 频谱图
            rgb_path = os.path.join(VIS_DIR, dataset, f"{method}_rgb.png")
            spec_path = os.path.join(VIS_DIR, dataset, f"{method}_spectrum.png")
            
            img_rgb = load_img(rgb_path)
            img_spec = load_img(spec_path)
            
            # =======================================================
            # 🌟 核心对齐操作：动态填充白边，让频谱图的宽度和RGB图完全一致
            # =======================================================
            h_rgb, w_rgb = img_rgb.shape[:2]
            h_spec, w_spec = img_spec.shape[:2]
            
            if w_spec < w_rgb:
                # 创建一个和 RGB 图一模一样大的纯白画布 (通道数跟随频谱图)
                padded_spec = np.ones((h_rgb, w_rgb, img_spec.shape[2]), dtype=img_spec.dtype) * 255
                # 把正方形的频谱图贴在最左边，右侧自然留白
                padded_spec[:h_spec, :w_spec] = img_spec
                img_spec = padded_spec
            # =======================================================
            
            # --- 绘制 RGB 行 ---
            ax_rgb = fig.add_subplot(gs[base_row, col_idx])
            ax_rgb.imshow(img_rgb)
            ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
            
            # 第一行加列标题 (方法名)
            if base_row == 0:
                ax_rgb.set_title(METHOD_NAMES[col_idx], fontsize=18, fontweight='bold', pad=10)
            
            # 第一列加行标题 (数据集 + RGB)
            if col_idx == 0:
                ax_rgb.set_ylabel(f"{dataset}\n(Spatial)", fontsize=16, fontweight='bold', labelpad=10)
                
            # 给图片加黑色细边框
            for spine in ax_rgb.spines.values():
                spine.set_edgecolor('black'); spine.set_linewidth(1.0)
                
            # --- 绘制 Spectrum 行 ---
            ax_spec = fig.add_subplot(gs[base_row + 1, col_idx])
            ax_spec.imshow(img_spec)
            ax_spec.set_xticks([]); ax_spec.set_yticks([])
            
            # 第一列加行标题 (数据集 + Spectrum)
            if col_idx == 0:
                ax_spec.set_ylabel(f"{dataset}\n(Spectrum)", fontsize=16, fontweight='bold', labelpad=10)
                
            for spine in ax_spec.spines.values():
                spine.set_edgecolor('black'); spine.set_linewidth(1.0)
                
    out_path = os.path.join(OUT_DIR, "All_Datasets_Enhancement_Comparison.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")

# ========================= Figure 2: Segmentation Overlays =========================
def compose_segmentation_figure():
    print("Generating Figure 2: Segmentation Overlay Comparison...")
    
    # 3个数据集 (3行) x 9个方法 (9列)
    fig = plt.figure(figsize=(24, 7))
    gs = fig.add_gridspec(3, 9, wspace=0.02, hspace=0.05)
    
    for row_idx, dataset in enumerate(DATASETS):
        for col_idx, method in enumerate(METHODS):
            
            # GT 没有 overlay，直接加载它的 RGB 图作为参照
            if method == 'gt':
                img_path = os.path.join(VIS_DIR, dataset, f"{method}_rgb.png")
            else:
                img_path = os.path.join(VIS_DIR, dataset, f"{method}_overlay.png")
                
            img = load_img(img_path)
            
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            
            # 第一行加列标题
            if row_idx == 0:
                ax.set_title(METHOD_NAMES[col_idx], fontsize=18, fontweight='bold', pad=10)
                
            # 第一列加行标题
            if col_idx == 0:
                ax.set_ylabel(f"{dataset}\n(Overlay)", fontsize=16, fontweight='bold', labelpad=10)
                
            for spine in ax.spines.values():
                spine.set_edgecolor('black'); spine.set_linewidth(1.0)
                
    out_path = os.path.join(OUT_DIR, "All_Datasets_Segmentation_Comparison.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    compose_enhancement_spectrum_figure()
    compose_segmentation_figure()