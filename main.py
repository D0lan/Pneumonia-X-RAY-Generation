#!/usr/bin/env python3
"""
cGAN 512×512 (NORMAL vs PNEUMONIA) with stabilizers:
- Projection D at 512 + lightweight 128 multi-scale
- CondInstanceNorm generator + Feature Matching loss
- Self-attention in D at 16×16 (VRAM-safe)
- Hinge + gentler R1 (gamma=2.0, every 32 iters)
- TTUR: lr_g=2e-4, lr_d=1.5e-4
- EMA of G for sampling
- Balanced sampler; light DiffAug
"""

import argparse, random, pathlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm as SN
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, utils as vutils
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# ------------------------------
# Utils
# ------------------------------

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def save_image_grid(tensor, path, nrow=8):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(tensor, path, nrow=nrow, normalize=True, value_range=(-1, 1))

def save_images_individual(tensor, outdir, prefix="sample", start_idx=1):
    out = pathlib.Path(outdir); out.mkdir(parents=True, exist_ok=True)
    for j, img in enumerate(tensor, start=start_idx):
        vutils.save_image(img, out / f"{prefix}_{j:04d}.png", normalize=True, value_range=(-1, 1))

# ------------------------------
# Data
# ------------------------------

def make_loader(data_root: str, batch_size: int, num_workers: int, balanced: bool = True):
    def pad_to_square(img):
        w, h = img.size; s = max(w, h)
        pad = ((s - w)//2, (s - h)//2, (s - w + 1)//2, (s - h + 1)//2)
        return transforms.functional.pad(img, pad, fill=0)

    tfm = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Lambda(pad_to_square),
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    ds = datasets.ImageFolder(root=data_root, transform=tfm)

    if balanced:
        labels = [y for _, y in ds.samples]
        import numpy as np
        counts = np.bincount(labels)
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    return ds, loader

# ------------------------------
# Blocks
# ------------------------------

class CondIN2d(nn.Module):
    """Conditional InstanceNorm: IN (affine=False) + FiLM from label embedding."""
    def __init__(self, num_features, emb_dim):
        super().__init__()
        self.inorm = nn.InstanceNorm2d(num_features, affine=False, eps=1e-5)
        self.gamma = nn.Linear(emb_dim, num_features)
        self.beta  = nn.Linear(emb_dim, num_features)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)
        nn.init.zeros_(self.gamma.bias);   nn.init.ones_(self.gamma.weight)

    def forward(self, x, e):
        x = self.inorm(x)
        g = self.gamma(e).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(e).unsqueeze(-1).unsqueeze(-1)
        return g * x + b

def up_block(in_c, out_c, emb_dim):
    return nn.ModuleDict({
        "ups": nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        "conv": nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
        "cbn": CondIN2d(out_c, emb_dim),
        "act": nn.ReLU(inplace=True),
    })

def up_forward(block, x, e):
    x = block["ups"](x); x = block["conv"](x); x = block["cbn"](x, e); return block["act"](x)

def d_block(in_c, out_c):
    return nn.Sequential(SN(nn.Conv2d(in_c, out_c, 3, 2, 1, bias=False)),
                         nn.LeakyReLU(0.2, inplace=True))

class SelfAttention2d(nn.Module):
    """Keep SA only at <=16×16 to be memory-safe."""
    def __init__(self, in_ch, use_sn=True):
        super().__init__()
        Conv = (lambda *a, **k: SN(nn.Conv2d(*a, **k))) if use_sn else nn.Conv2d
        c_mid = max(1, in_ch // 8)
        self.q = Conv(in_ch, c_mid, 1, bias=False)
        self.k = Conv(in_ch, c_mid, 1, bias=False)
        self.v = Conv(in_ch, in_ch,  1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).view(B, -1, H*W)                   # B,Cq,N
        k = self.k(x).view(B, -1, H*W)                   # B,Ck,N
        v = self.v(x).view(B,  C, H*W)                   # B,C,N
        attn = torch.softmax((q.transpose(1,2) @ k) / (q.size(1)**0.5), dim=-1)  # B,N,N
        y = (attn @ v.transpose(1,2)).transpose(1,2).contiguous().view(B, C, H, W)
        return x + self.gamma * y

# ------------------------------
# Generator 512
# ------------------------------

class Generator512(nn.Module):
    def __init__(self, z_dim=128, ngf=32, out_channels=1, num_classes=2, emb_dim=None):
        super().__init__()
        self.z_dim = z_dim
        self.emb_dim = emb_dim or z_dim
        self.label_emb = nn.Embedding(num_classes, self.emb_dim)
        self.fc = nn.ConvTranspose2d(z_dim, ngf*16, 4, 1, 0, bias=False)
        self.cbn0 = CondIN2d(ngf*16, self.emb_dim)
        self.act0 = nn.ReLU(True)

        self.up1 = up_block(ngf*16, ngf*8,  self.emb_dim)  # 8
        self.up2 = up_block(ngf*8,  ngf*4,  self.emb_dim)  # 16
        self.up3 = up_block(ngf*4,  ngf*2,  self.emb_dim)  # 32
        self.up4 = up_block(ngf*2,  ngf,    self.emb_dim)  # 64
        self.up5 = up_block(ngf,    ngf//2, self.emb_dim)  # 128
        self.up6 = up_block(ngf//2, ngf//4, self.emb_dim)  # 256
        self.up7 = up_block(ngf//4, ngf//8, self.emb_dim)  # 512

        self.final_conv = nn.Conv2d(ngf//8, out_channels, 3, 1, 1, bias=False)

    def forward(self, z, y):
        e = self.label_emb(y)
        x = self.fc(z); x = self.cbn0(x, e); x = self.act0(x)
        x = up_forward(self.up1, x, e)
        x = up_forward(self.up2, x, e)
        x = up_forward(self.up3, x, e)
        x = up_forward(self.up4, x, e)
        x = up_forward(self.up5, x, e)
        x = up_forward(self.up6, x, e)
        x = up_forward(self.up7, x, e)
        x = self.final_conv(x)
        return torch.tanh(x)

# ------------------------------
# Discriminators
# ------------------------------

class Discriminator512(nn.Module):
    """Main projection D (512 path) with SA at 16×16 only."""
    def __init__(self, ndf=32, in_channels=1, num_classes=2):
        super().__init__()
        C = ndf
        self.head = nn.Sequential(SN(nn.Conv2d(in_channels, C, 3, 1, 1, bias=False)),
                                  nn.LeakyReLU(0.2, inplace=True))
        # 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
        self.down1 = d_block(C,     C*2)   # 256
        self.down2 = d_block(C*2,   C*4)   # 128
        self.down3 = d_block(C*4,   C*8)   # 64
        self.down4 = d_block(C*8,   C*8)   # 32
        self.down5 = d_block(C*8,   C*8)   # 16
        self.sa16  = SelfAttention2d(C*8)  # SA at 16×16
        self.down6 = d_block(C*8,   C*8)   # 8
        self.down7 = d_block(C*8,   C*8)   # 4
        self.down8 = d_block(C*8,   C*8)   # 2
        self.down9 = d_block(C*8,   C*8)   # 1

        self.uncond = SN(nn.Conv2d(C*8, 1, 1, 1, 0, bias=False))
        self.embed  = SN(nn.Embedding(num_classes, C*8))

    def forward(self, x, y, return_feats=False):
        feats = []
        h = self.head(x); feats.append(h)
        h = self.down1(h); feats.append(h)
        h = self.down2(h); feats.append(h)
        h = self.down3(h); feats.append(h)
        h = self.down4(h); feats.append(h)
        h = self.down5(h); feats.append(h)
        h = self.sa16(h); feats.append(h)
        h = self.down6(h); feats.append(h)
        h = self.down7(h); feats.append(h)
        h = self.down8(h); feats.append(h)
        h = self.down9(h); feats.append(h)

        uncond = self.uncond(h).view(-1)
        h_pool = torch.sum(h, dim=(2,3))
        proj = torch.sum(self.embed(y) * h_pool, dim=1)
        out = uncond + proj
        return (out, feats) if return_feats else out

class Discriminator128(nn.Module):
    """Light projection D for 128 view (no SA)."""
    def __init__(self, ndf=32, in_channels=1, num_classes=2):
        super().__init__()
        C = ndf
        self.head = nn.Sequential(SN(nn.Conv2d(in_channels, C, 3, 1, 1, bias=False)),
                                  nn.LeakyReLU(0.2, inplace=True))
        # 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
        self.d1 = d_block(C,     C*2)
        self.d2 = d_block(C*2,   C*4)
        self.d3 = d_block(C*4,   C*8)
        self.d4 = d_block(C*8,   C*8)
        self.d5 = d_block(C*8,   C*8)
        self.d6 = d_block(C*8,   C*8)
        self.d7 = d_block(C*8,   C*8)
        self.uncond = SN(nn.Conv2d(C*8, 1, 1, 1, 0, bias=False))
        self.embed  = SN(nn.Embedding(num_classes, C*8))
    def forward(self, x, y):
        h = self.head(x)
        h = self.d1(h); h = self.d2(h); h = self.d3(h)
        h = self.d4(h); h = self.d5(h); h = self.d6(h); h = self.d7(h)
        uncond = self.uncond(h).view(-1)
        h_pool = torch.sum(h, dim=(2,3))
        proj = torch.sum(self.embed(y) * h_pool, dim=1)
        return uncond + proj

# ------------------------------
# EMA
# ------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay=0.999, device=None):
        self.decay = decay
        self.shadow = {k: v.detach().clone().to(device if device else v.device)
                       for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1. - self.decay)
            else:
                self.shadow[k] = v.detach().clone()
    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

# ------------------------------
# Training
# ------------------------------

@dataclass
class Config:
    data_root: str = "data/chest_xray/train"
    out_dir: str = "runs/cgan512_stabilized"
    z_dim: int = 128
    batch_size: int = 16
    epochs: int = 50
    lr_g: float = 2e-4
    lr_d: float = 1.5e-4
    beta1: float = 0.0
    beta2: float = 0.99
    ngf: int = 32
    ndf: int = 32
    num_workers: int = 8
    seed: int = 42
    log_every: int = 50
    resume: bool = False
    amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes: int = 2
    balanced_sampler: bool = True
    r1_gamma: float = 2.0
    r1_every: int = 32
    small_view: int = 128
    small_weight: float = 1.0
    fm_lambda: float = 10.0
    ema: float = 0.999

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.num_classes = cfg.num_classes

        # Data
        self.ds, self.loader = make_loader(cfg.data_root, cfg.batch_size, cfg.num_workers, balanced=cfg.balanced_sampler)
        print("Class mapping:", self.ds.class_to_idx)

        # Models
        self.netG = Generator512(z_dim=cfg.z_dim, ngf=cfg.ngf, out_channels=1, num_classes=cfg.num_classes).to(self.device)
        self.netG_ema = Generator512(z_dim=cfg.z_dim, ngf=cfg.ngf, out_channels=1, num_classes=cfg.num_classes).to(self.device)
        self.netD_big   = Discriminator512(ndf=cfg.ndf, in_channels=1, num_classes=cfg.num_classes).to(self.device)
        self.netD_small = Discriminator128(ndf=cfg.ndf, in_channels=1, num_classes=cfg.num_classes).to(self.device)

        self.netG.apply(self.weights_init)
        self.netG_ema.load_state_dict(self.netG.state_dict())

        self.optG = optim.Adam(self.netG.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
        self.optD = optim.Adam(list(self.netD_big.parameters()) + list(self.netD_small.parameters()),
                               lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

        self.use_amp = cfg.amp and self.device.type == "cuda"
        self.scalerG = GradScaler(enabled=self.use_amp)
        self.scalerD = GradScaler(enabled=self.use_amp)

        # EMA helper
        self.ema = EMA(self.netG, decay=cfg.ema, device=self.device)

        # Fixed noise/labels (interleaved)
        N = 64
        self.fixed_z = torch.randn(N, cfg.z_dim, 1, 1, device=self.device)
        classes = torch.arange(self.num_classes, device=self.device)
        self.fixed_y = classes.repeat_interleave(N // self.num_classes)

        # Dirs
        self.out_dir = pathlib.Path(cfg.out_dir)
        (self.out_dir / "samples").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "samples_by_class").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "samples_individual_512").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "ckpt").mkdir(parents=True, exist_ok=True)

        # Optionally resume
        if cfg.resume: self._try_load()

        # Save a sanity grid of real images
        real_batch, _ = next(iter(self.loader))
        save_image_grid(real_batch[:64], self.out_dir / "samples" / "real_grid_512.png", nrow=8)

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if getattr(m, "weight", None) is not None:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            pass

    def _save(self, epoch, it):
        state = {
            'epoch': epoch, 'it': it,
            'netG': self.netG.state_dict(),
            'netG_ema': self.netG_ema.state_dict(),
            'netD_big': self.netD_big.state_dict(),
            'netD_small': self.netD_small.state_dict(),
            'optG': self.optG.state_dict(),
            'optD': self.optD.state_dict(),
            'cfg': self.cfg.__dict__,
        }
        torch.save(state, self.out_dir / "ckpt" / "last.pt")

    def _try_load(self):
        ckpt_path = self.out_dir / "ckpt" / "last.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device)
            self.netG.load_state_dict(state['netG'])
            self.netG_ema.load_state_dict(state['netG_ema'])
            self.netD_big.load_state_dict(state['netD_big'])
            self.netD_small.load_state_dict(state['netD_small'])
            self.optG.load_state_dict(state['optG'])
            self.optD.load_state_dict(state['optD'])
            print(f"Resumed from {ckpt_path}")

    # -------- DiffAug --------
    @staticmethod
    def diffaug_translate(x, ratio=0.02):
        B, C, H, W = x.shape
        tx, ty = int(H * ratio), int(W * ratio)
        shift_x = torch.randint(-tx, tx + 1, (B, 1, 1), device=x.device)
        shift_y = torch.randint(-ty, ty + 1, (B, 1, 1), device=x.device)
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device),
                                torch.linspace(-1, 1, W, device=x.device), indexing="ij")
        gx = gx.unsqueeze(0) + (2.0 * shift_y / max(W - 1, 1))
        gy = gy.unsqueeze(0) + (2.0 * shift_x / max(H - 1, 1))
        grid = torch.stack((gx, gy), dim=-1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    @staticmethod
    def diffaug_cutout(x, ratio=0.03):
        B, C, H, W = x.shape
        ch, cw = int(H * ratio), int(W * ratio)
        cy = torch.randint(0, H, (B,), device=x.device)
        cx = torch.randint(0, W, (B,), device=x.device)
        mask = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)
        for i in range(B):
            y1 = max(0, cy[i].item() - ch//2); y2 = min(H, cy[i].item() + ch//2)
            x1 = max(0, cx[i].item() - cw//2); x2 = min(W, cx[i].item() + cw//2)
            if y2 > y1 and x2 > x1:
                mask[i, :, y1:y2, x1:x2] = 0
        return x * mask

    @staticmethod
    def diffaug(x, use_translate=True, use_cutout=True):
        if use_translate: x = Trainer.diffaug_translate(x, 0.02)
        if use_cutout:    x = Trainer.diffaug_cutout(x, 0.03)
        return x

    # -------- Hinge / R1 / FM --------
    @staticmethod
    def d_hinge_loss(real_logits, fake_logits):
        return (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())

    @staticmethod
    def g_hinge_loss(fake_logits):
        return -fake_logits.mean()

    @staticmethod
    def r1_regularization(d_out, real_imgs, gamma=2.0):
        grads = torch.autograd.grad(outputs=d_out.sum(), inputs=real_imgs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return 0.5 * gamma * grads.pow(2).flatten(1).sum(1).mean()

    @staticmethod
    def fm_loss(fake_feats, real_feats):
        # Feature Matching: L1 difference of mean feature activations across layers
        losses = []
        for f_fake, f_real in zip(fake_feats, real_feats):
            if f_fake.dim() == 4:
                mf = f_fake.mean(dim=(0,2,3))
                mr = f_real.mean(dim=(0,2,3))
                losses.append((mf - mr).abs().mean())
        return sum(losses) / max(1, len(losses))

    # -------- Train --------
    def train(self):
        cfg = self.cfg
        global_step = 0
        small = cfg.small_view

        for epoch in range(1, cfg.epochs + 1):
            pbar = tqdm(self.loader, desc=f"Epoch {epoch}/{cfg.epochs}")

            for i, (imgs, labels) in enumerate(pbar, 1):
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                bsz = imgs.size(0)

                # ----- Train D (both scales) -----
                self.netD_big.train(); self.netD_small.train()
                self.optD.zero_grad(set_to_none=True)

                real_leaf = imgs.detach(); real_leaf.requires_grad_(True)
                real_big = self.diffaug(real_leaf)
                real_small = F.interpolate(real_big, (small, small), mode="bilinear", align_corners=False)

                with autocast(enabled=self.use_amp):
                    # real
                    real_log_big, real_feats_big = self.netD_big(real_big, labels, return_feats=True)
                    real_log_small = self.netD_small(real_small, labels)

                    # fake (no grad to G)
                    y_fake = torch.randint(0, self.num_classes, (bsz,), device=self.device)
                    z = torch.randn(bsz, cfg.z_dim, 1, 1, device=self.device)
                    with torch.no_grad():
                        fake = self.netG(z, y_fake)
                    fake_big = self.diffaug(fake)
                    fake_small = F.interpolate(fake_big, (small, small), mode="bilinear", align_corners=False)
                    fake_log_big = self.netD_big(fake_big, y_fake)
                    fake_log_small = self.netD_small(fake_small, y_fake)

                    d_loss_big   = self.d_hinge_loss(real_log_big,   fake_log_big)
                    d_loss_small = self.d_hinge_loss(real_log_small, fake_log_small)
                    d_loss = d_loss_big + cfg.small_weight * d_loss_small

                if (global_step % cfg.r1_every) == 0:
                    with torch.cuda.amp.autocast(enabled=False):
                        d_loss = d_loss + self.r1_regularization(real_log_big.float(), real_big, gamma=cfg.r1_gamma)

                self.scalerD.scale(d_loss).backward()
                self.scalerD.step(self.optD)
                self.scalerD.update()

                # ----- Train G (both scales) + Feature Matching -----
                self.netG.train()
                self.optG.zero_grad(set_to_none=True)

                with autocast(enabled=self.use_amp):
                    y_fake = torch.randint(0, self.num_classes, (bsz,), device=self.device)
                    z = torch.randn(bsz, cfg.z_dim, 1, 1, device=self.device)
                    fake = self.netG(z, y_fake)

                    fake_big = fake
                    fake_small = F.interpolate(fake_big, (small, small), mode="bilinear", align_corners=False)

                    fake_log_big, fake_feats_big = self.netD_big(fake_big, y_fake, return_feats=True)
                    fake_log_small = self.netD_small(fake_small, y_fake)

                    g_loss_big   = self.g_hinge_loss(fake_log_big)
                    g_loss_small = self.g_hinge_loss(fake_log_small)
                    # FM uses current real batch features (stop-grad)
                    with torch.no_grad():
                        _, real_feats_big = self.netD_big(imgs, labels, return_feats=True)
                    fm = self.fm_loss(fake_feats_big, real_feats_big)

                    g_loss = g_loss_big + cfg.small_weight * g_loss_small + cfg.fm_lambda * fm

                self.scalerG.scale(g_loss).backward()
                self.scalerG.step(self.optG)
                self.scalerG.update()

                # EMA update & copy to ema net (for sampling)
                self.ema.update(self.netG)
                self.ema.copy_to(self.netG_ema)

                if i % cfg.log_every == 0:
                    with torch.no_grad():
                        d_real_m = real_log_big.mean().item()
                        d_fake_m = fake_log_big.mean().item()
                    pbar.set_postfix({
                        "d": f"{d_loss.item():.3f}",
                        "g": f"{g_loss.item():.3f}",
                        "Dreal": f"{d_real_m:.2f}",
                        "Dfake": f"{d_fake_m:.2f}",
                        "fm": f"{fm.item():.3f}",
                    })
                global_step += 1

            # ----- Epoch end: sample with EMA -----
            with torch.no_grad():
                self.netG_ema.eval()
                samples = self.netG_ema(self.fixed_z, self.fixed_y).cpu()

            save_image_grid(samples, self.out_dir / "samples" / f"epoch_{epoch:03d}_512.png", nrow=8)
            for c in range(self.num_classes):
                idx = (self.fixed_y.cpu() == c).nonzero(as_tuple=False).view(-1)[:32]
                if idx.numel() > 0:
                    save_image_grid(samples[idx], self.out_dir / "samples_by_class" / f"epoch_{epoch:03d}_cls{c}.png", nrow=8)
            save_images_individual(samples, self.out_dir / "samples_individual_512" / f"epoch_{epoch:03d}", prefix="x")

            self._save(epoch, i)

        print("Training complete.")

# ------------------------------
# CLI
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stabilized cGAN 512×512 (multi-scale, EMA, FM)")
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='runs/cgan512_stabilized')
    p.add_argument('--z_dim', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr_g', type=float, default=2e-4)
    p.add_argument('--lr_d', type=float, default=1.5e-4)
    p.add_argument('--beta1', type=float, default=0.0)
    p.add_argument('--beta2', type=float, default=0.99)
    p.add_argument('--ngf', type=int, default=32)
    p.add_argument('--ndf', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--balanced_sampler', action='store_true', default=True)
    p.add_argument('--r1_gamma', type=float, default=2.0)
    p.add_argument('--r1_every', type=int, default=32)
    p.add_argument('--small_view', type=int, default=128)
    p.add_argument('--small_weight', type=float, default=1.0)
    p.add_argument('--fm_lambda', type=float, default=10.0)
    p.add_argument('--ema', type=float, default=0.999)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config(**vars(args))
    trainer = Trainer(cfg); trainer.train()

if __name__ == '__main__':
    main()
