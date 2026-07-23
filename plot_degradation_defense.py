import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ========================= Config =========================
IMG_SIZE = 512
SAMPLE_IMG_PATH = "./data/test/drive_images/22_training.tif"
FOV_MASK_PATH = "./data/test/drive_fov_masks/22_training_mask.gif"

# ========================= Utils =========================
from src.utils import (
    add_illumination_shading, reduce_contrast, 
    add_scattering_haze, add_sensor_noise, 
    add_color_cast, degrade_image
)

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))

def load_fov_mask(path):
    """安全读取 GIF 格式的 FOV Mask，并扩展维度以便于矩阵乘法"""
    if not os.path.exists(path):
        print(f"[警告] 找不到 FOV Mask: {path}，将使用全白 Mask 代替。")
        mask = np.ones((IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
        return mask
    # 使用 PIL 读取 GIF 最安全
    mask = np.array(Image.open(path).convert('L'))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)
    return np.expand_dims(mask, axis=-1)  # 扩展为 (H, W, 1)

# ========================= Main Plotting =========================
def generate_degradation_defense_figure():
    print("Generating Degradation Reference Figure with FOV Masking...")

    gt_img = load_image(SAMPLE_IMG_PATH)
    fov_mask = load_fov_mask(FOV_MASK_PATH)
    
    # 强制固定种子
    np.random.seed(42)

    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(18, 8)) 
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.45)
    
    # ---------------------------------------------------------
    # Part A: Clinical Degradation Modeling 
    # ---------------------------------------------------------
    gs_a = gs[0].subgridspec(1, 6, wspace=0.1) 
    
    # 所有的退化操作后，乘以 fov_mask 消除边界外的伪影
    img_illu = add_illumination_shading(gt_img.copy(), strength=0.6) * fov_mask
    img_cont = reduce_contrast(gt_img.copy(), alpha=0.6) * fov_mask
    img_haze = add_scattering_haze(gt_img.copy(), strength=1.8) * fov_mask
    img_nois = add_sensor_noise(gt_img.copy(), sigma=15) * fov_mask
    img_colo = add_color_cast(gt_img.copy(), strength=0.15) * fov_mask
    gt_clean_masked = gt_img * fov_mask
    
    dict_a = {
        "Clean GT": gt_clean_masked,
        "Vignetting\n(Illumination)": img_illu,
        "Low Contrast\n(Opacity)": img_cont,
        "Cataract\n(Scattering Haze)": img_haze,
        "Sensor Defect\n(Noise)": img_nois,
        "Color Shift\n(Device Gap)": img_colo
    }
    
    for i, (title, img) in enumerate(dict_a.items()):
        ax = fig.add_subplot(gs_a[0, i])
        ax.imshow(img)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12) 
        ax.axis('off')
        
        if i == 0:
            ax.text(0, 1.35, "(a) Clinical Degradation Modeling", 
                    transform=ax.transAxes, fontsize=16, fontweight='bold', ha='left')

    # ---------------------------------------------------------
    # Part B: Reference for Degradation Severities
    # ---------------------------------------------------------
    gs_b = gs[1].subgridspec(1, 4, wspace=0.1) 
    
    np.random.seed(42) 
    img_sev1 = degrade_image(gt_img.copy(), severity=1) * fov_mask
    img_sev2 = degrade_image(gt_img.copy(), severity=2) * fov_mask
    img_sev3 = degrade_image(gt_img.copy(), severity=3) * fov_mask
    
    dict_b = {
        "Clean GT": gt_clean_masked,
        "Mild Degradation\n(Severity 1)": img_sev1,
        "Moderate Degradation\n(Severity 2)": img_sev2,
        "Severe Degradation\n(Severity 3)": img_sev3
    }
    
    for i, (title, img) in enumerate(dict_b.items()):
        ax = fig.add_subplot(gs_b[0, i])
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.axis('off')
        
        if i == 0:
            ax.text(0, 1.30, "(b) Simulated Degradation Severities", 
                    transform=ax.transAxes, fontsize=16, fontweight='bold', ha='left')

    os.makedirs("./paper_figures", exist_ok=True)
    out_path = "./paper_figures/Fig_Degradation_Defense.pdf"
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n🎉 完美退化展示图已生成: {out_path}")

if __name__ == "__main__":
    generate_degradation_defense_figure()