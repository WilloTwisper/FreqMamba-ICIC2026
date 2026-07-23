import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import cv2
import time
import torch
import lpips
import random
import hashlib
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.filters import frangi, threshold_otsu

# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./data/test/drive_images"
MASK_DIR = "./data/test/drive_masks"

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

SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)
IMG_SIZE = 512

# =========================
# Reproducibility
# =========================
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================
# Utils
# =========================
from src.utils import degrade_image

def load_image(path):
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))

def load_mask(path):
    mask = cv2.imread(path, 0)
    if mask is None: raise FileNotFoundError(f"Mask not found: {path}")
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)

def load_fov_mask(path):
    if not os.path.exists(path): return np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    fov = cv2.imread(path, 0)
    fov = cv2.resize(fov, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (fov > 0).astype(np.uint8)

def generate_fov_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, fov = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(fov, cv2.MORPH_CLOSE, kernel)

def img_to_tensor(img):
    return (torch.from_numpy(img).permute(2, 0, 1).float() / 255).unsqueeze(0).to(DEVICE)

def img_to_lpips(img):
    return ((torch.from_numpy(img).permute(2, 0, 1).float() / 255) * 2 - 1).unsqueeze(0).to(DEVICE)

def seg_preprocess(img):
    img = img.astype(np.float32) / 255
    img = (img - 0.5) / 0.5
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

def tensor_to_img(t):
    return np.clip(t.squeeze().permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)

# =========================
# Metrics
# =========================
def fft_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    center = (h // 2, w // 2)
    low = mag[center[0]-20:center[0]+20, center[1]-20:center[1]+20].sum()
    high = mag.sum() - low
    return low, high

def dice_score(pred, gt):
    inter = np.sum(pred.flatten() * gt.flatten())
    return (2 * inter + 1e-8) / (pred.sum() + gt.sum() + 1e-8)

def sensitivity(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fn = np.sum((pred == 0) & (gt == 1))
    return tp / (tp + fn + 1e-8)

def specificity(pred, gt):
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    return tn / (tn + fp + 1e-8)

def calculate_hd95(pred, gt):
    pred_pts = np.argwhere(pred == 1)
    gt_pts = np.argwhere(gt == 1)
    if len(pred_pts) == 0 or len(gt_pts) == 0: return np.nan
    dist = cdist(pred_pts, gt_pts)
    return max(np.percentile(np.min(dist, axis=1), 95), np.percentile(np.min(dist, axis=0), 95))

# =========================
# Traditional Baselines
# =========================
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def apply_msrcr(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32) + 1.0
    scales = [15, 81, 251]
    retinex = np.zeros_like(v)
    for s in scales:
        blur = cv2.GaussianBlur(v, (s, s), 0)
        retinex += np.log10(v) - np.log10(blur)
    retinex = retinex / len(scales)
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex) + 1e-8) * 255.0
    hsv[:, :, 2] = np.clip(retinex, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def extract_vessels_frangi(img_rgb):
    gray = img_rgb[:, :, 1]
    vesselness = frangi(gray, sigmas=range(1, 5, 1), black_ridges=True)
    vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)
    thresh = threshold_otsu(vesselness)
    return vesselness, (vesselness > thresh).astype(np.uint8)

# =========================
# Model Loaders & Speed
# =========================
def load_model(path, model, is_seg=False):
    if not os.path.exists(path):
        print(f"[Warning] Weights not found at {path}, skipping...")
        return None
    state = torch.load(path, map_location=DEVICE)
    if "state_dict" in state: state = state["state_dict"]
    if is_seg:
        model.load_state_dict(state, strict=True)
    else:
        new_state = {k.replace("model.", ""): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

def measure_speed_dl(model):
    if model is None: return 0.0
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu"):
            for _ in range(10): model(x)
            if DEVICE == "cuda": torch.cuda.synchronize()
            start = time.time()
            for _ in range(50): model(x)
            if DEVICE == "cuda": torch.cuda.synchronize()
    return 50 / (time.time() - start)

def measure_speed_cpu(func, sample_img):
    start = time.time()
    for _ in range(10): func(sample_img)
    return 10 / (time.time() - start)

# =========================
# Main Evaluation
# =========================
def evaluate(args):
    dataset = args.dataset
    set_seed()
    
    from src.model import HybridMambaUNet, VanillaUNet, RestormerUNet, NAFUNet, FFCUNet
    from monai.networks.nets import UNet as SegUNet

    print(f"\nEvaluating on {dataset}...")
    
    models_dl = {
        'unet': load_model(MODEL_PATHS['unet'], VanillaUNet(), is_seg=False),
        'restormer': load_model(MODEL_PATHS['restormer'], RestormerUNet(), is_seg=False),
        'nafnet': load_model(MODEL_PATHS['nafnet'], NAFUNet(), is_seg=False),
        'ffcnet': load_model(MODEL_PATHS['ffcnet'], FFCUNet(), is_seg=False),
        'freq': load_model(MODEL_PATHS['freq'], HybridMambaUNet(), is_seg=False)
    }

    seg_model = SegUNet(spatial_dims=2, in_channels=3, out_channels=1, channels=(16,32,64,128,256), strides=(2,2,2,2))
    seg_model_path = args.seg_model if args.seg_model else SEG_MODELS[dataset]
    seg_model = load_model(seg_model_path, seg_model, is_seg=True)
    lpips_metric = lpips.LPIPS(net="alex").to(DEVICE)

    if dataset == "DRIVE": suffix_img, suffix_mask = ".tif", ".gif"
    elif dataset == "CHASE": suffix_img, suffix_mask = ".jpg", ".png"
    elif dataset == "HRF": suffix_img, suffix_mask = ".jpg", ".tif"

    images = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(suffix_img)])
    
    METHODS =['in', 'clahe', 'msrcr', 'unet', 'restormer', 'nafnet', 'ffcnet', 'freq']
    NAMES   =["Input", "CLAHE", "MSRCR", "UNet", "Restormer", "NAFNet", "FFCUNet", "FreqMamba"]

    metrics = {metric: {m:[] for m in METHODS} for metric in['psnr', 'ssim', 'lpips', 'dice', 'auc', 'sen', 'spe']}

    if args.compute_hd95:
        metrics['hd95'] = {m:[] for m in METHODS}
    if args.compute_freq:
        metrics['l_freq'] = {m:[] for m in METHODS}
        metrics['h_freq'] = {m:[] for m in METHODS}
        
    metrics['dice']['frangi'] = []
    metrics['auc']['frangi'] =[]

    print(f"Evaluating {len(images)} images on {dataset}...")
    sample_for_speed = None

    for i, name in enumerate(tqdm(images)):
        img_path = os.path.join(DATA_DIR, name)
        if dataset == "DRIVE": 
            mask_name = f"{name.split('_')[0]}_manual1{suffix_mask}"
        elif dataset == "CHASE": 
            mask_name = name.replace(".jpg", "_1stHO.png")
        elif dataset == "HRF": 
            mask_name = name.replace(".jpg", ".tif")
        mask_path = os.path.join(MASK_DIR, mask_name)

        gt = load_image(img_path)
        mask = load_mask(mask_path)
        if sample_for_speed is None: sample_for_speed = gt

        if dataset == "DRIVE": 
            fov = load_fov_mask(os.path.join("./data/test/drive_fov_masks", name.replace(".tif", "_mask.gif")))
        elif dataset == "HRF": 
            fov = load_fov_mask(os.path.join("./data/test/hrf_fov_masks", name.replace(".jpg", "_mask.tif")))
        else: 
            fov = generate_fov_mask(gt)
        valid = fov.flatten() == 1

        seed_val = int(hashlib.md5(name.encode('utf-8')).hexdigest(), 16) % (2**32)
        np.random.seed(seed_val)
        random.seed(seed_val)

        degraded = degrade_image(gt, severity=2)
        
        img_dict = {
            'in': degraded,
            'clahe': apply_clahe(degraded),
            'msrcr': apply_msrcr(degraded)
        }

        t_in = img_to_tensor(degraded)
        with torch.no_grad():
            for m_key in ['unet', 'restormer', 'nafnet', 'ffcnet', 'freq']:
                if models_dl[m_key] is not None:
                    img_dict[m_key] = tensor_to_img(models_dl[m_key](t_in))
                else:
                    img_dict[m_key] = degraded

        for m in METHODS:
            img = img_dict[m]
            # 1. Quality
            metrics['psnr'][m].append(psnr(gt, img, data_range=255))
            metrics['ssim'][m].append(ssim(gt, img, channel_axis=-1, data_range=255))
            metrics['lpips'][m].append(lpips_metric(img_to_lpips(gt).detach(), img_to_lpips(img).detach()).item())
            
            # 2. Segmentation
            with torch.no_grad():
                prob = torch.sigmoid(seg_model(seg_preprocess(img))).cpu().numpy().squeeze()
            pred = (prob > 0.5).astype(np.uint8)
            
            metrics['dice'][m].append(dice_score(pred * fov, mask * fov))
            metrics['sen'][m].append(sensitivity(pred[fov==1], mask[fov==1]))
            metrics['spe'][m].append(specificity(pred[fov==1], mask[fov==1]))
            metrics['auc'][m].append(roc_auc_score(mask.flatten()[valid], prob.flatten()[valid]))

            if args.compute_hd95:
                metrics['hd95'][m].append(calculate_hd95(pred * fov, mask * fov))
            if args.compute_freq:
                l, h = fft_energy(img)
                metrics['l_freq'][m].append(l)
                metrics['h_freq'][m].append(h)

        # Frangi
        prob_f, pred_f = extract_vessels_frangi(degraded)
        metrics['dice']['frangi'].append(dice_score(pred_f * fov, mask * fov))
        metrics['auc']['frangi'].append(roc_auc_score(mask.flatten()[valid], prob_f.flatten()[valid]))

    def m_avg(metric, m): 
        return float(np.nanmean(metrics[metric][m]))

    print("\n================= Evaluation Summary =================")
    print("\n========== Table 1: Image Quality ==========")
    print("{:<12} {:>10} {:>10} {:>10}".format("Method", "PSNR", "SSIM", "LPIPS"))
    for m, name in zip(METHODS, NAMES):
        print("{:<12} {:>10.4f} {:>10.4f} {:>10.4f}".format(name, m_avg('psnr', m), m_avg('ssim', m), m_avg('lpips', m)))

    print("\n========== Table 2: Segmentation ==========")
    print("{:<12} {:>10} {:>10} {:>10} {:>10}".format("Method", "Dice", "AUC", "Sen", "Spe"))
    for m, name in zip(METHODS, NAMES):
        print("{:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(name, m_avg('dice', m), m_avg('auc', m), m_avg('sen', m), m_avg('spe', m)))
    print("{:<12} {:>10.4f} {:>10.4f} {:>10} {:>10}".format("Frangi", m_avg('dice','frangi'), m_avg('auc','frangi'), "N/A", "N/A"))

    if args.compute_hd95:
        print("\n========== Table 3: Boundary Accuracy ==========")
        print("{:<12} {:>10}".format("Method", "HD95"))
        for m, name in zip(METHODS, NAMES):
            print("{:<12} {:>10.4f}".format(name, m_avg('hd95', m)))

    if args.compute_efficiency:
        fps_dl = {k: measure_speed_dl(v) for k, v in models_dl.items() if v is not None}
        p_dl = {k: sum(p.numel() for p in v.parameters()) / 1e6 if v else 0 for k, v in models_dl.items() if v is not None}
        fps_clahe = measure_speed_cpu(apply_clahe, sample_for_speed)
        fps_msrcr = measure_speed_cpu(apply_msrcr, sample_for_speed)

        print("\n========== Table 4: Efficiency ==========")
        print("{:<12} {:>12} {:>12}".format("Method", "Params(M)", "FPS"))
        print("{:<12} {:>12.3f} {:>12.2f}".format("Input", 0.0, float('inf')))
        print("{:<12} {:>12.3f} {:>12.2f}".format("CLAHE", 0.0, fps_clahe))
        print("{:<12} {:>12.3f} {:>12.2f}".format("MSRCR", 0.0, fps_msrcr))
        print("{:<12} {:>12.3f} {:>12.2f}".format("UNet", p_dl.get('unet', 0.0), fps_dl.get('unet', 0.0)))
        print("{:<12} {:>12.3f} {:>12.2f}".format("Restormer", p_dl.get('restormer', 0.0), fps_dl.get('restormer', 0.0)))
        print("{:<12} {:>12.3f} {:>12.2f}".format("NAFNet", p_dl.get('nafnet', 0.0), fps_dl.get('nafnet', 0.0)))
        print("{:<12} {:>12.3f} {:>12.2f}".format("FFCUNet", p_dl.get('ffcnet', 0.0), fps_dl.get('ffcnet', 0.0)))
        print("{:<12} {:>12.3f} {:>12.2f}".format("FreqMamba", p_dl.get('freq', 0.0), fps_dl.get('freq', 0.0)))

    if args.compute_freq:
        print("\n========== Table 5: Frequency Analysis ==========")
        print("{:<12} {:>18} {:>15}".format("Method", "HighFreqEnergy", "HF Ratio"))
        for m, name in zip(METHODS, NAMES):
            hf_ratio = m_avg('h_freq', m) / (m_avg('h_freq', m) + m_avg('l_freq', m))
            print("{:<12} {:>18.2f} {:>15.4f}".format(name, m_avg('h_freq', m), hf_ratio))
            
    print("====================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="DRIVE", choices=["DRIVE", "CHASE", "HRF"])
    parser.add_argument("--seg_model", type=str, default=None)
    parser.add_argument("--compute_hd95", action="store_true", help="Whether to compute HD95 (slow)")
    parser.add_argument("--compute_freq", action="store_true", help="Whether to compute FFT energy")
    parser.add_argument("--compute_efficiency", action="store_true", help="Whether to compute FPS and Params")
    
    args = parser.parse_args()
    
    DATASETS = {
        "DRIVE": ("./data/test/drive_images", "./data/test/drive_masks"),
        "CHASE": ("./data/test/chase_images", "./data/test/chase_masks"),
        "HRF":   ("./data/test/hrf_images", "./data/test/hrf_masks"),
    }
    DATA_DIR, MASK_DIR = DATASETS[args.dataset]
    evaluate(args)