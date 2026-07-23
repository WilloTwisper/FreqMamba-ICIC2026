import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Charbonnier Loss
# ==========================================
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

# ==========================================
# 2. Edge Loss (Sobel)
# ==========================================
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],[-1, 0, 1]], dtype=torch.float32)

        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],[ 1,  2,  1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        pred_gray = pred.mean(1, keepdim=True)
        target_gray = target.mean(1, keepdim=True)

        pred_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_y = F.conv2d(pred_gray, self.sobel_y, padding=1)

        target_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_y = F.conv2d(target_gray, self.sobel_y, padding=1)

        loss_x = F.l1_loss(pred_x, target_x)
        loss_y = F.l1_loss(pred_y, target_y)

        return loss_x + loss_y

# ==========================================
# 3. FFT Loss
# ==========================================
def fft_loss(pred, target):
    pred_fft = torch.fft.rfft2(pred.float(), norm='ortho')
    target_fft = torch.fft.rfft2(target, norm='ortho')

    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)

    pred_mag_log = torch.log(pred_mag + 1e-8)
    target_mag_log = torch.log(target_mag + 1e-8)

    return F.l1_loss(pred_mag_log, target_mag_log)

# ==========================================
# 4. FreqMamba Loss
# ==========================================
class FreqMambaLoss(nn.Module):
    def __init__(self, lambda_fft=0.1, lambda_edge=0.1):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.edge = EdgeLoss()
        
        self.lambda_fft = lambda_fft
        self.lambda_edge = lambda_edge

    def forward(self, pred, target):
        loss_char = self.charbonnier(pred, target)
        loss_fft = fft_loss(pred, target)
        loss_edge = self.edge(pred, target)

        loss = loss_char + (self.lambda_fft * loss_fft) + (self.lambda_edge * loss_edge)
        return loss