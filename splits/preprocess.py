import os
import pandas as pd
import shutil
import random
from tqdm import tqdm

CSV_FILES = {
    'train': './train_1.csv',
    'test': './test.csv',
    'val': './valid.csv'
}

IMAGE_DIRS = {
    'train': './train_images/train_images',
    'test': './test_images/test_images',
    'val': './val_images/val_images'
}

TARGET_DIR = "./data/aptos2019_imagesr"
SPLIT_DIR = "./splits"

TOTAL_IMAGES = 1280
SEED = 42

def prepare_dataset():
    random.seed(SEED)

    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)

    print("Collecting healthy images...")

    all_images = {}

    for dataset_name, csv_path in CSV_FILES.items():
        df = pd.read_csv(csv_path)
        df = df[df['diagnosis'] == 0]

        image_dir = IMAGE_DIRS[dataset_name]

        for img_id in tqdm(df['id_code'], desc=f"{dataset_name}"):

            for ext in ['png', 'jpg']:
                path = os.path.join(image_dir, f"{img_id}.{ext}")

                if os.path.exists(path):
                    size = os.path.getsize(path)
                    if img_id not in all_images or size > all_images[img_id]['size']:
                        all_images[img_id] = {
                            'id_code': img_id,
                            'path': path,
                            'size': size,
                            'format': ext
                        }
                    break

    all_images = list(all_images.values())
    print(f"{len(all_images)} unique healthy images collected.")

    all_images.sort(key=lambda x: x['size'], reverse=True)

    selected = all_images[:TOTAL_IMAGES]

    print(f"Selected top {len(selected)} images.")

    random.shuffle(selected)
    train = selected[:1024]
    val   = selected[1024:1152]
    test  = selected[1152:1280]

    print("finished：")
    print(f"train: {len(train)}")
    print(f"val:   {len(val)}")
    print(f"test:  {len(test)}")

    def copy_images(split, name):
        txt_path = os.path.join(SPLIT_DIR, f"{name}.txt")

        with open(txt_path, "w") as f:
            for img in tqdm(split, desc=f"copy {name}"):

                new_name = f"{img['id_code']}.{img['format']}"
                dst = os.path.join(TARGET_DIR, new_name)

                if not os.path.exists(dst):
                    shutil.copy(img['path'], dst)

                f.write(new_name + "\n")

    copy_images(train, "train")
    copy_images(val, "val")
    copy_images(test, "test")

if __name__ == "__main__":
    prepare_dataset()