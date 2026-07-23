import os
import cv2
import random
import hashlib
import argparse
import numpy as np
from tqdm import tqdm

from src.utils import degrade_image   # 确保和 evaluate.py 使用同一个 degrade_image

IMG_SIZE = 512
DATA_ROOT = "./data/test"
DEGRADED_ROOT = "./data/degraded"
os.makedirs(DEGRADED_ROOT, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--severity", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--datasets", nargs="+", default=["DRIVE", "CHASE", "HRF"])
    args = parser.parse_args()

    print(f"生成退化图像 | severity={args.severity} | 数据集={args.datasets}\n")

    for dataset in args.datasets:
        img_dir = os.path.join(DATA_ROOT, f"{dataset.lower()}_images")
        out_dir = os.path.join(DEGRADED_ROOT, f"{dataset}_sev{args.severity}")
        os.makedirs(out_dir, exist_ok=True)

        images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.tif', '.jpg', '.png'))])

        for fname in tqdm(images, desc=f"{dataset}"):
            img_path = os.path.join(img_dir, fname)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # ================== 与 evaluate.py 完全一致的种子逻辑 ==================
            seed_val = int(hashlib.md5(fname.encode('utf-8')).hexdigest(), 16) % (2**32)
            random.seed(seed_val)
            np.random.seed(seed_val)

            degraded = degrade_image(img, severity=args.severity)

            # 保存（文件名保持一致）
            save_path = os.path.join(out_dir, fname)
            cv2.imwrite(save_path, cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR))

        print(f"✅ {dataset} 完成 → {out_dir}")

    print("\n🎉 所有退化图像生成完毕！")
    print(f"保存路径: {DEGRADED_ROOT}")

if __name__ == "__main__":
    main()