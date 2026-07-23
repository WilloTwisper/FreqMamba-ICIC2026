import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import copy
import numpy as np

from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ReduceLROnPlateau
from src.model import HybridMambaUNet, VanillaUNet, RestormerUNet, NAFUNet, FFCUNet
from src.loss import FreqMambaLoss
from src.dataset import FundusDataset




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def validate(model, loader, criterion, device):

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for bad_img, good_img in loader:

            bad_img = bad_img.to(device)
            good_img = good_img.to(device)

            with autocast(device_type=device):
                output = model(bad_img)
                loss = criterion(output, good_img)

            total_loss += loss.item()

    return total_loss / len(loader)


def train(args):

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*50)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print("="*50)

    run_save_dir = os.path.join(args.save_dir, args.model, args.exp_name)
    os.makedirs(run_save_dir, exist_ok=True)

    train_dataset = FundusDataset(
        data_dir=args.data_dir,
        split_file="splits/train.txt",
        mode="train",
        image_size=512
    )

    val_dataset = FundusDataset(
        data_dir=args.data_dir,
        split_file="splits/val.txt",
        mode="val",
        image_size=512
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device=="cuda"),
        persistent_workers=(args.num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=(device=="cuda")
    )
        
    if args.model == "freqmamba":
        model = HybridMambaUNet(3,3).to(device)
    elif args.model == "restormer":
        model = RestormerUNet(3,3).to(device)
    elif args.model == "nafnet":
        model = NAFUNet(3,3).to(device)
    elif args.model == "ffcnet":
        model = FFCUNet(3,3).to(device)
    else:
        model = VanillaUNet(3,3).to(device)


    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad = False
        
    ema_decay = 0.999

    criterion = FreqMambaLoss(lambda_fft=args.lambda_fft, lambda_edge=args.lambda_edge).to(device)

    optimizer=optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6)

    scaler = GradScaler(enabled=(device=="cuda"))

    loss_history=[]
    val_history=[]

    best_val_loss = float("inf")
    patience = 30
    counter = 0

    for epoch in range(args.epochs):

        model.train()
        epoch_loss=0

        loop=tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}"
        )

        for bad_img,good_img in loop:

            bad_img=bad_img.to(device)
            good_img=good_img.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device):

                outputs=model(bad_img)

                loss=criterion(outputs,good_img)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            clip_grad_norm_(model.parameters(),1.0)

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
                model_buffers = dict(model.named_buffers())
                ema_buffers = dict(ema_model.named_buffers())
                for name, b in model_buffers.items():
                    if "freq_radius" not in name and b is not None and name in ema_buffers and ema_buffers[name] is not None:
                        ema_buffers[name].data.copy_(b.data)
                
                

            epoch_loss+=loss.item()

            loop.set_postfix(loss=loss.item())

        avg_loss=epoch_loss/len(train_loader)

        loss_history.append(avg_loss)
        
        val_loss = validate(ema_model, val_loader, criterion, device)
        val_history.append(val_loss)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1} | Train: {avg_loss:.4f} | Val: {val_loss:.4f}")
        if (epoch+1)%10==0:
            torch.save(
                ema_model.state_dict(),
                os.path.join(run_save_dir, f"{args.model}_epoch{epoch+1}.pth") 
            )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                ema_model.state_dict(),
                os.path.join(run_save_dir, f"{args.model}_best.pth")
            )
            counter = 0
            print(f"Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            counter += 1
            print(f"Early stopping counter: {counter} / {patience}")

        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    final_path=os.path.join(run_save_dir, f"{args.model}_final.pth")
    torch.save(ema_model.state_dict(), final_path)
    save_loss_plot(
        loss_history,
        val_history,
        os.path.join(run_save_dir, "loss_curve.png")
    )
    print(f"Training Finished. Saved to {run_save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="freqmamba", 
                        choices=["freqmamba", "unet", "restormer", "nafnet", "ffcnet"])

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/aptos2019_images"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=4e-4
    )

    parser.add_argument("--exp_name", type=str, default="full_model", help="use to distinguish different ablation experiments")
    parser.add_argument("--lambda_fft", type=float, default=0.3, help="Weight for the FFT loss (set to 0 for ablation)")
    parser.add_argument("--lambda_edge", type=float, default=0.3, help="Weight for the edge loss (set to 0 for ablation)")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    train(args)