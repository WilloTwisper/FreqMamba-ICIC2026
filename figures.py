import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import cv2
import torch
import random
import hashlib
import numpy as np
import matplotlib.pyplot as plt

# ========================= Config =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATHS = {
    "unet": "./checkpoints/unet/full_model/unet_best.pth",
    "restormer": "./checkpoints/restormer/full_model/restormer_best.pth",
    "nafnet": "./checkpoints/nafnet/full_model/nafnet_best.pth",
    "ffcnet": "./checkpoints/ffcnet/full_model/ffcnet_best.pth",
    "freq": "./checkpoints/freqmamba/full_model/freqmamba_best.pth"
}

SEG_MODELS = {
    "DRIVE": "./segmentation/drive_seg_unet.pth",
    "HRF": "./segmentation/hrf_seg_unet.pth",
    "CHASE": "./segmentation/chase_seg_unet.pth"
}

VIS_DIR = "./visualizations"
os.makedirs(VIS_DIR, exist_ok=True)
IMG_SIZE = 512

# ========================= 🌟 核心放大框坐标配置 =========================
ZOOM_CONFIGS = {
    'DRIVE': {'crop1': (140, 160), 'crop2': (320, 360), 'size': 64},
    'HRF':   {'crop1': (200, 10), 'crop2': (300, 320), 'size': 64},
    'CHASE': {'crop1': (220, 200), 'crop2': (60, 300), 'size': 64}
}

# ========================= Utils =========================
from src.utils import degrade_image
# 注意：如果您的评估脚本叫 evaluate_segmentation.py，请将下面的 evaluate 改为 evaluate_segmentation
from evaluate import (
    load_model, apply_clahe, apply_msrcr, seg_preprocess, tensor_to_img
)

def create_overlay(img, pred, gt):
    overlay = img.copy()
    overlay[(pred==1) & (gt==1)] = [0, 255, 0]   # 绿: TP
    overlay[(pred==1) & (gt==0)] = [255, 0, 0]   # 红: FP
    overlay[(pred==0) & (gt==1)] = [0, 0, 255]   # 蓝: FN (断裂血管)
    return overlay

def save_fft_spectrum(img, save_path):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    plt.imsave(save_path, mag, cmap='inferno')

def load_fov_mask(dataset, name, gt):
    if dataset == "DRIVE": 
        path = os.path.join("./data/test/drive_fov_masks", name.replace(".tif", "_mask.gif"))
    elif dataset == "HRF": 
        path = os.path.join("./data/test/hrf_fov_masks", name.replace(".jpg", "_mask.tif"))
    else:
        gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        _, fov = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5,5), np.uint8)
        return cv2.morphologyEx(fov, cv2.MORPH_CLOSE, kernel)

    if not os.path.exists(path):
        return np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    fov = cv2.imread(path, 0)
    fov = cv2.resize(fov, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (fov > 0).astype(np.uint8)

# ========================= 核心排版函数（一左二右无缝拼接） =========================
def add_dual_magnified_crops_right(img, crop1_pos, crop2_pos, crop_size=60):
    h, w = img.shape[:2]
    canvas = np.ones((h, int(w * 1.5), 3), dtype=np.uint8) * 255
    canvas[:, :w] = img.copy()
    crop_size = min(crop_size, 80)

    # ===== 提取 crop1（红框）=====
    x1, y1 = crop1_pos
    x1 = max(0, min(x1, w - crop_size))
    y1 = max(0, min(y1, h - crop_size))
    crop1 = img[y1:y1+crop_size, x1:x1+crop_size]

    # ===== 提取 crop2（蓝框）=====
    x2, y2 = crop2_pos
    x2 = max(0, min(x2, w - crop_size))
    y2 = max(0, min(y2, h - crop_size))
    crop2 = img[y2:y2+crop_size, x2:x2+crop_size]

    # ===== 放大到目标尺寸 =====
    target_w = w // 2
    target_h = h // 2
    mag1 = cv2.resize(crop1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mag2 = cv2.resize(crop2, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # ===== 原图画框 =====
    cv2.rectangle(canvas[:, :w], (x1, y1), (x1+crop_size, y1+crop_size), (0, 0, 255), 4) # 红色粗框
    cv2.rectangle(canvas[:, :w], (x2, y2), (x2+crop_size, y2+crop_size), (255, 0, 0), 4) # 蓝色粗框

    # ===== 拼接右侧 =====
    right_x = w
    canvas[0:target_h, right_x:right_x + target_w] = mag1        # 右上角放红框内容
    canvas[target_h:h, right_x:right_x + target_w] = mag2        # 右下角放蓝框内容

        # ===== 放大图外边框 (向内缩进2个像素，防止被画布边缘裁切) =====
    t = 4  # 线条粗细
    offset = t // 2  # 缩进量
    
    # 右上角红框
    cv2.rectangle(canvas, 
                  (right_x + offset, offset), 
                  (right_x + target_w - offset, target_h - offset), 
                  (0, 0, 255), t)
                  
    # 右下角蓝框
    cv2.rectangle(canvas, 
                  (right_x + offset, target_h + offset), 
                  (right_x + target_w - offset, h - offset), 
                  (255, 0, 0), t)

    return canvas

# ========================= Main =========================
def generate_visualizations():

    TARGET_SAMPLES = {
        'DRIVE': '22_training.tif',
        'HRF': '01_h.jpg',
        'CHASE': 'Image_02L.jpg'
    }

    # 导入所有需要的模型架构
    from src.model import HybridMambaUNet, VanillaUNet, RestormerUNet, NAFUNet, FFCUNet
    from monai.networks.nets import UNet as SegUNet

    for dataset, target_name in TARGET_SAMPLES.items():
        print(f"\nGenerating visuals for {dataset} - {target_name}...")

        d_dir = f"./data/test/{dataset.lower()}_images"
        m_dir = f"./data/test/{dataset.lower()}_masks"

        img_path = os.path.join(d_dir, target_name)
        if dataset == "DRIVE": mask_name = f"{target_name.split('_')[0]}_manual1.gif"
        elif dataset == "CHASE": mask_name = target_name.replace(".jpg", "_1stHO.png")
        else: mask_name = target_name.replace(".jpg", ".tif")
        mask_path = os.path.join(m_dir, mask_name)

        if not os.path.exists(img_path):
            print(f"Skipping {dataset} (Image not found)")
            continue

        # 加载数据
        img_gt = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
        mask_gt = cv2.resize(cv2.imread(mask_path, 0), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask_gt = (mask_gt > 127).astype(np.uint8)
        fov = load_fov_mask(dataset, target_name, img_gt)

        # 固定随机退化
        seed_val = int(hashlib.md5(target_name.encode()).hexdigest(), 16) % (2**32)
        random.seed(seed_val); np.random.seed(seed_val)
        img_in = degrade_image(img_gt, severity=2)

        # ===== 🌟 加载所有深度学习模型 =====
        models_dl = {
            'unet': load_model(MODEL_PATHS['unet'], VanillaUNet(), False),
            'restormer': load_model(MODEL_PATHS['restormer'], RestormerUNet(), False),
            'nafnet': load_model(MODEL_PATHS['nafnet'], NAFUNet(), False),
            'ffcnet': load_model(MODEL_PATHS['ffcnet'], FFCUNet(), False),
            'freq': load_model(MODEL_PATHS['freq'], HybridMambaUNet(), False)
        }
        
        seg_model = load_model(SEG_MODELS[dataset], SegUNet(spatial_dims=2, in_channels=3, out_channels=1, channels=(16,32,64,128,256), strides=(2,2,2,2)), True)

        # ===== 收集所有生成结果 =====
        results = {
            'gt': img_gt, 
            'in': img_in,
            'clahe': apply_clahe(img_in), 
            'msrcr': apply_msrcr(img_in)
        }

        # 🌟 DL 模型推理 (包含 nafnet 和 ffcnet)
        t_in = (torch.from_numpy(img_in).permute(2,0,1).float()/255).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            for k in ['unet', 'restormer', 'nafnet', 'ffcnet', 'freq']:
                if models_dl.get(k) is not None: 
                    results[k] = tensor_to_img(models_dl[k](t_in))
                else:
                    print(f"Warning: Model weights for '{k}' not found. Skipping...")

        # ===== 保存所有带局部放大的 RGB 和 Overlay =====
        out_folder = os.path.join(VIS_DIR, dataset)
        os.makedirs(out_folder, exist_ok=True)

        c1 = ZOOM_CONFIGS[dataset]['crop1']
        c2 = ZOOM_CONFIGS[dataset]['crop2']
        csize = ZOOM_CONFIGS[dataset]['size']

        for k, img_res in results.items():
            # 1. 增强 RGB 图像 (带双放大框)
            final_rgb = add_dual_magnified_crops_right(img_res, c1, c2, csize)
            cv2.imwrite(f"{out_folder}/{k}_rgb.png", cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))

            # 2. 频谱图 (不需要放大框)
            save_fft_spectrum(img_res, f"{out_folder}/{k}_spectrum.png")

            # 3. 分割 Overlay 图 (也带双放大框！)
            if k != 'gt':
                with torch.no_grad():
                    prob = torch.sigmoid(seg_model(seg_preprocess(img_res))).cpu().numpy().squeeze()
                pred = (prob > 0.5).astype(np.uint8)
                
                # 先生成正常的 Overlay
                overlay = create_overlay(img_res, pred * fov, mask_gt * fov)
                # 给 Overlay 套上双放大框
                final_overlay = add_dual_magnified_crops_right(overlay, c1, c2, csize)
                
                cv2.imwrite(f"{out_folder}/{k}_overlay.png", cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR))

    print("\n🎉 Done! All visualizations with dual zoom-in boxes saved.")

if __name__ == "__main__":
    generate_visualizations()