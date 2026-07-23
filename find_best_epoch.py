import os
import glob
import torch
import numpy as np
import hashlib
import random
import argparse
import re
import lpips
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ======================================================================
# 严格对齐 evaluate.py 里的图像质量评测逻辑
# ======================================================================
from evaluate import (
    load_model, load_image, img_to_tensor, tensor_to_img, img_to_lpips
)
from src.utils import degrade_image
from src.model import HybridMambaUNet, RestormerUNet, VanillaUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def pre_load_dataset(dataset):
    """预加载并退化数据集，缓存 GT 用于计算 PSNR/SSIM，缓存 LPIPS 张量提速"""
    data_dir = f"./data/test/{dataset.lower()}_images"
    
    if dataset == "DRIVE": suffix_img = ".tif"
    elif dataset == "CHASE": suffix_img = ".jpg"
    elif dataset == "HRF": suffix_img = ".jpg"
    
    images = sorted([f for f in os.listdir(data_dir) if f.endswith(suffix_img)])
    
    dataset_cache = []
    for name in tqdm(images, desc=f"Caching {dataset}", leave=False):
        img_path = os.path.join(data_dir, name)
        gt = load_image(img_path)
            
        # 严格固定种子，保证退化效果和之前完全一致
        seed_val = int(hashlib.md5(name.encode('utf-8')).hexdigest(), 16) % (2**32)
        np.random.seed(seed_val)
        random.seed(seed_val)
        
        degraded = degrade_image(gt, severity=2)
        t_in = img_to_tensor(degraded)
        gt_lpips_tensor = img_to_lpips(gt).detach()
        
        dataset_cache.append({
            't_in': t_in,
            'gt': gt,                     # 用于 PSNR / SSIM
            'gt_lpips': gt_lpips_tensor   # 提前算好，LPIPS 评估时起飞
        })
        
    return dataset_cache


def evaluate_cache(model, lpips_metric, dataset_cache):
    """极速计算全套感知质量指标"""
    psnrs, ssims, lpips_vals = [], [], []
    model.eval()
    
    with torch.no_grad():
        for data in dataset_cache:
            # 模型推理
            img_res = tensor_to_img(model(data['t_in']))
            
            # 1. PSNR & SSIM (基于 numpy)
            psnrs.append(psnr(data['gt'], img_res, data_range=255))
            ssims.append(ssim(data['gt'], img_res, channel_axis=-1, data_range=255))
            
            # 2. LPIPS (基于 tensor)
            img_res_lpips = img_to_lpips(img_res).detach()
            lpips_val = lpips_metric(data['gt_lpips'], img_res_lpips).item()
            lpips_vals.append(lpips_val)
            
    return {
        'psnr': float(np.mean(psnrs)),
        'ssim': float(np.mean(ssims)),
        'lpips': float(np.mean(lpips_vals))
    }

def get_checkpoints(directory, prefix):
    """自动扫描目录下所有匹配的 pth，并按 Epoch 顺序排列"""
    pattern = os.path.join(directory, f"{prefix}*.pth")
    files = glob.glob(pattern)
    
    def sort_key(filepath):
        name = os.path.basename(filepath)
        if "best" in name: return 99999  # best 放最后
        match = re.search(r'epoch(\d+)', name)
        return int(match.group(1)) if match else -1
        
    return sorted(files, key=sort_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="freqmamba")
    parser.add_argument("--dir", type=str, default="./checkpoints/freqmamba/full_model")
    args = parser.parse_args()
    
    datasets = ["DRIVE", "HRF", "CHASE"]
    checkpoints = get_checkpoints(args.dir, args.model)
    
    if not checkpoints:
        print(f"❌ 找不到任何匹配的模型: {args.dir}/{args.model}*.pth")
        exit()
        
    print(f"🔍 找到 {len(checkpoints)} 个权重文件 (包含 _best)")
    
    # 提前加载 LPIPS 评估模型（AlexNet backbone），全局只加载一次！
    print("Loading LPIPS Metric (AlexNet)...")
    lpips_metric = lpips.LPIPS(net="alex").to(DEVICE)
    lpips_metric.eval()
    
    # 用来记录每个数据集的最优盲盒 (以 PSNR 决定胜负)
    best_records = {ds: {'psnr': 0, 'ckpt': '', 'metrics': {}} for ds in datasets}
    
    for dataset in datasets:
        print(f"\n" + "="*50)
        print(f"🎨 开始扫荡数据集: {dataset} (Image Quality)")
        print("="*50)
        
        # 1. 预加载数据
        dataset_cache = pre_load_dataset(dataset)
        
        print(f"\n{'-'*13} 评估结果 {'-'*13}")
        print(f"{'Checkpoint Name':<25} | {'PSNR ↑':<8} | {'SSIM ↑':<8} | {'LPIPS ↓':<8}")
        
        # 2. 遍历扫库
        for ckpt_path in checkpoints:
            ckpt_name = os.path.basename(ckpt_path)
            
            if args.model == "freqmamba": net = HybridMambaUNet()
            elif args.model == "restormer": net = RestormerUNet()
            else: net = VanillaUNet()
            
            model = load_model(ckpt_path, net, is_seg=False)
            res = evaluate_cache(model, lpips_metric, dataset_cache)
            
            print(f"{ckpt_name:<25} | {res['psnr']:.4f}   | {res['ssim']:.4f}   | {res['lpips']:.4f}")
            
            # 更新最佳记录 (以 PSNR 最高为准)
            if res['psnr'] > best_records[dataset]['psnr']:
                best_records[dataset]['psnr'] = res['psnr']
                best_records[dataset]['ckpt'] = ckpt_name
                best_records[dataset]['metrics'] = res
                
    # 打印最终锦集
    print("\n" + "🌟"*25)
    print("📸 全局最佳画质盲盒总结 (基于 PSNR 最高)")
    print("🌟"*25)
    for dataset in datasets:
        b = best_records[dataset]
        print(f"[{dataset}] 画质最强权重: {b['ckpt']}")
        print(f"       -> PSNR: {b['metrics']['psnr']:.4f} | SSIM: {b['metrics']['ssim']:.4f} | LPIPS: {b['metrics']['lpips']:.4f}\n")