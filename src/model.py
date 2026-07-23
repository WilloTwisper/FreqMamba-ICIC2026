import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)

class FrequencySpatialMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mamba_h = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_v = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
   
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

        self.freq_gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        self.amp_modulator = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 1)
        )

        self.act = nn.GELU()

        self.fusion_gate = nn.Parameter(torch.tensor([0.5]))

        self.pos_embed = nn.Parameter(torch.randn(1, dim, 128, 128))
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.1)
        self.cutoff = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.15)
        self.sharpness = nn.Parameter(torch.ones(1, dim, 1, 1) * 8.0)

        self.register_buffer("freq_radius", None)

    def forward(self, x):
        B, C, H, W = x.shape

        pos_2d = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)

        x_h = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        pos_h = pos_2d.permute(0, 2, 3, 1).reshape(1, H * W, C)
        x_h = x_h + pos_h
        out_h = self.mamba_h(self.norm(x_h)) 
        out_h = out_h.reshape(B, H, W, C).permute(0, 3, 1, 2)

        x_v = x.transpose(2, 3).permute(0, 2, 3, 1).reshape(B, H * W, C)
        pos_v = pos_2d.transpose(2, 3).permute(0, 2, 3, 1).reshape(1, H * W, C)
        x_v = x_v + pos_v
        out_v = self.mamba_v(self.norm(x_v)) 
        out_v = out_v.reshape(B, W, H, C).permute(0, 3, 2, 1).transpose(2, 3)

        out_spatial = self.spatial_fusion(out_h + out_v)

        x_fp32 = x.float()
        fft_x = torch.fft.rfft2(x_fp32, norm='ortho')
        mag = torch.abs(fft_x)
        phase = torch.angle(fft_x)

        mag_mean = mag.mean(dim=(-2, -1), keepdim=True) + 1e-6
        mag_norm = mag / mag_mean

        gate = self.freq_gate(mag_norm.type_as(x)).float()
        learned_mag = self.amp_modulator(mag_norm.type_as(x)).float()
        enhance = torch.tanh(learned_mag)

        if getattr(self, "freq_radius", None) is None or self.freq_radius.shape[-2:] != (H, W // 2 + 1):
            fy = torch.fft.fftfreq(H, device=x.device)
            fx = torch.fft.rfftfreq(W, device=x.device)
            fy_mesh, fx_mesh = torch.meshgrid(fy, fx, indexing='ij')
            self.freq_radius = torch.sqrt(fy_mesh**2 + fx_mesh**2).unsqueeze(0).unsqueeze(0)

        current_freq_radius = self.freq_radius.to(x.device)

        alpha = torch.clamp(self.alpha, 0.0, 0.5)
        cutoff = torch.clamp(self.cutoff, 0.01, 0.49)
        sharpness = torch.clamp(self.sharpness, 1.0, 20.0)

        freq_weight = torch.sigmoid(sharpness * (current_freq_radius - cutoff))

        mag_modulated = mag_norm * (1.0 + alpha * gate * enhance * freq_weight)
        mag_modulated = mag_modulated * mag_mean
        fft_modulated = mag_modulated * torch.exp(1j * phase.float())

        out_freq = torch.fft.irfft2(fft_modulated, s=(H, W), norm='ortho').type_as(x)

        fusion_w = torch.sigmoid(self.fusion_gate)
        out = fusion_w * out_spatial + (1 - fusion_w) * out_freq
        out = self.act(out)

        return x + out

class HybridMambaUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32):
        super().__init__()

        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, 1, 1), ResBlock(base_ch))
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 4, 2, 1) 
        
        self.enc2 = nn.Sequential(ResBlock(base_ch*2), ResBlock(base_ch*2))
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1) 

        self.bottleneck = nn.Sequential(
            FrequencySpatialMambaBlock(base_ch*4),
            FrequencySpatialMambaBlock(base_ch*4),
            FrequencySpatialMambaBlock(base_ch*4)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch*4, base_ch*2, 3, 1, 1)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*2, 3, 1, 1), 
            ResBlock(base_ch*2)
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch*2, base_ch, 3, 1, 1)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, 1, 1), 
            ResBlock(base_ch)
        )

        self.tail = nn.Conv2d(base_ch, out_ch, 3, 1, 1)
        

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x_mid = self.bottleneck(self.down2(x2))
        x_up2 = self.up2(x_mid)
        x_dec2 = self.dec2(torch.cat([x_up2, x2], dim=1))
        
        x_up1 = self.up1(x_dec2)
        x_dec1 = self.dec1(torch.cat([x_up1, x1], dim=1))

        enhanced = x + self.tail(x_dec1)
        return enhanced

class VanillaUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32):
        super().__init__()

        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, 1, 1), ResBlock(base_ch))
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 4, 2, 1) 
        
        self.enc2 = nn.Sequential(ResBlock(base_ch*2), ResBlock(base_ch*2))
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1) 
        self.bottleneck = nn.Sequential(
            ResBlock(base_ch*4),
            ResBlock(base_ch*4),
            ResBlock(base_ch*4)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch*4, base_ch*2, 3, 1, 1)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*2, 3, 1, 1), 
            ResBlock(base_ch*2)
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch*2, base_ch, 3, 1, 1)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, 1, 1), 
            ResBlock(base_ch)
        )

        self.tail = nn.Conv2d(base_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        
        x_mid = self.bottleneck(self.down2(x2)) 
        
        x_up2 = self.up2(x_mid)
        x_dec2 = self.dec2(torch.cat([x_up2, x2], dim=1))
        
        x_up1 = self.up1(x_dec2)
        x_dec1 = self.dec1(torch.cat([x_up1, x1], dim=1))
        
        return x + self.tail(x_dec1)

class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Sequential(
            nn.Conv2d(channels, channels * 3, 1, bias=False),
            nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=False)
        )
        self.project_out = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, c, h, w)
        return self.project_out(out)

class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network"""
    def __init__(self, channels, expansion_factor=2.66):
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Sequential(
            nn.Conv2d(channels, hidden_channels * 2, 1, bias=False),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1, groups=hidden_channels * 2, bias=False)
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, 1, bias=False)

    def forward(self, x):
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

class RestormerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MDTA(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = GDFN(dim)

    def forward(self, x):
        x_norm1 = x.permute(0, 2, 3, 1)
        x = x + self.attn(self.norm1(x_norm1).permute(0, 3, 1, 2))
        
        x_norm2 = x.permute(0, 2, 3, 1)
        x = x + self.ffn(self.norm2(x_norm2).permute(0, 3, 1, 2))
        return x

class RestormerUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, 1, 1), ResBlock(base_ch))
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 4, 2, 1) 
        self.enc2 = nn.Sequential(ResBlock(base_ch*2), ResBlock(base_ch*2))
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1) 

        self.bottleneck = nn.Sequential(
            RestormerBlock(base_ch*4),
            RestormerBlock(base_ch*4),
            RestormerBlock(base_ch*4)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(base_ch*4, base_ch*2, 3, 1, 1))
        self.dec2 = nn.Sequential(nn.Conv2d(base_ch*4, base_ch*2, 3, 1, 1), ResBlock(base_ch*2))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(base_ch*2, base_ch, 3, 1, 1))
        self.dec1 = nn.Sequential(nn.Conv2d(base_ch*2, base_ch, 3, 1, 1), ResBlock(base_ch))
        self.tail = nn.Conv2d(base_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x_mid = self.bottleneck(self.down2(x2)) 
        x_up2 = self.up2(x_mid)
        x_dec2 = self.dec2(torch.cat([x_up2, x2], dim=1))
        x_up1 = self.up1(x_dec2)
        x_dec1 = self.dec1(torch.cat([x_up1, x1], dim=1))
        return x + self.tail(x_dec1)
        
class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, c * 2, 1)
        self.dwconv = nn.Conv2d(c * 2, c * 2, 3, 1, 1, groups=c * 2)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1)
        )
        self.conv2 = nn.Conv2d(c, c, 1)
        
        self.norm2 = LayerNorm2d(c)
        self.conv3 = nn.Conv2d(c, c * 2, 1)
        self.conv4 = nn.Conv2d(c, c, 1)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        inp = x
        x1 = self.norm1(x)
        x1 = self.dwconv(self.conv1(x1))
        x1_1, x1_2 = x1.chunk(2, dim=1)
        x1 = x1_1 * x1_2             
        x1 = x1 * self.sca(x1)         
        x1 = self.conv2(x1)
        y = inp + x1 * self.beta         
        inp2 = y
        x2 = self.norm2(y)
        x2 = self.conv3(x2)
        x2_1, x2_2 = x2.chunk(2, dim=1)
        x2 = x2_1 * x2_2                  
        x2 = self.conv4(x2)
        return inp2 + x2 * self.gamma    

class NAFUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, 1, 1), ResBlock(base_ch))
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 4, 2, 1) 
        self.enc2 = nn.Sequential(ResBlock(base_ch*2), ResBlock(base_ch*2))
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1) 

        self.bottleneck = nn.Sequential(NAFBlock(base_ch*4), NAFBlock(base_ch*4), NAFBlock(base_ch*4))

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(base_ch*4, base_ch*2, 3, 1, 1))
        self.dec2 = nn.Sequential(nn.Conv2d(base_ch*4, base_ch*2, 3, 1, 1), ResBlock(base_ch*2))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(base_ch*2, base_ch, 3, 1, 1))
        self.dec1 = nn.Sequential(nn.Conv2d(base_ch*2, base_ch, 3, 1, 1), ResBlock(base_ch))
        self.tail = nn.Conv2d(base_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x_mid = self.bottleneck(self.down2(x2)) 
        x_up2 = self.up2(x_mid)
        x_dec2 = self.dec2(torch.cat([x_up2, x2], dim=1))
        x_up1 = self.up1(x_dec2)
        x_dec1 = self.dec1(torch.cat([x_up1, x1], dim=1))
        return x + self.tail(x_dec1)


class FFCBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )
        
        self.freq_conv = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 2, 1, bias=False)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        xs = self.spatial_conv(x)
        fft_x = torch.fft.rfft2(x.float(), norm='ortho')
        xf = torch.cat([fft_x.real, fft_x.imag], dim=1)
        xf = self.freq_conv(xf)
        real_out, imag_out = xf.chunk(2, dim=1)
        fft_out = torch.complex(real_out, imag_out)
        xf_out = torch.fft.irfft2(fft_out, s=x.shape[-2:], norm='ortho').type_as(x)
        return self.act(x + xs + xf_out)

class FFCUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, 1, 1), ResBlock(base_ch))
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 4, 2, 1) 
        self.enc2 = nn.Sequential(ResBlock(base_ch*2), ResBlock(base_ch*2))
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1) 

        self.bottleneck = nn.Sequential(FFCBlock(base_ch*4), FFCBlock(base_ch*4), FFCBlock(base_ch*4))

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(base_ch*4, base_ch*2, 3, 1, 1))
        self.dec2 = nn.Sequential(nn.Conv2d(base_ch*4, base_ch*2, 3, 1, 1), ResBlock(base_ch*2))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(base_ch*2, base_ch, 3, 1, 1))
        self.dec1 = nn.Sequential(nn.Conv2d(base_ch*2, base_ch, 3, 1, 1), ResBlock(base_ch))
        self.tail = nn.Conv2d(base_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x_mid = self.bottleneck(self.down2(x2)) 
        x_up2 = self.up2(x_mid)
        x_dec2 = self.dec2(torch.cat([x_up2, x2], dim=1))
        x_up1 = self.up1(x_dec2)
        x_dec1 = self.dec1(torch.cat([x_up1, x1], dim=1))
        return x + self.tail(x_dec1)