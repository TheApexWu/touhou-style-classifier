"""
Full Touhou Diffusion Training

Uses ALL available tracks, saves checkpoints, manages memory.
Designed to run safely on M2 Mac alongside other work.

Usage:
    python scripts/train_touhou_diffusion_full.py

    # Resume from checkpoint:
    python scripts/train_touhou_diffusion_full.py --resume
"""

import argparse
import gc
import math
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio import AudioLoader, find_audio_files
from src.features.spectral import SpectralFeatureExtractor


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    'epochs': 500,
    'batch_size': 8,  # Small batch to be gentle on memory
    'lr': 2e-4,
    'checkpoint_every': 100,
    'sample_every': 100,
    'cache_clear_every': 50,
    'crops_per_track': 3,
    'timesteps': 1000,
    'cfg_dropout': 0.1,
    'cfg_scale': 3.0,
}

CIRCLES = {
    0: ('IOSYS', 'IOSYS'),
    1: ('UNDEAD CORPORATION', 'UNDEAD_CORPORATION'),
    2: ('暁Records', 'Akatsuki_Records'),
    3: ('SOUND HOLIC', 'SOUND_HOLIC'),
    4: ('Liz Triangle', 'Liz_Triangle'),
}

CIRCLE_NAMES = ['IOSYS', 'UNDEAD', 'Akatsuki', 'SOUND HOLIC', 'Liz Tri']


# ============================================================
# MODEL
# ============================================================

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.gelu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class TouhouUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_dim=128, num_classes=5):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim * 2),
        )
        self.class_embed = nn.Embedding(num_classes + 1, time_dim * 2)

        self.enc1 = ConvBlock(in_ch, base_ch, time_dim * 2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, time_dim * 2)
        self.pool = nn.MaxPool2d(2)
        self.bot = ConvBlock(base_ch * 2, base_ch * 2, time_dim * 2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch, time_dim * 2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, time_dim * 2)
        self.out = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t, class_label=None):
        t_emb = self.time_embed(t)
        if class_label is None:
            c_emb = self.class_embed(torch.full((x.size(0),), 5, device=x.device))
        else:
            c_emb = self.class_embed(class_label)
        emb = t_emb + c_emb

        e1 = self.enc1(x, emb)
        e2 = self.enc2(self.pool(e1), emb)
        b = self.bot(self.pool(e2), emb)

        d2 = F.interpolate(b, scale_factor=2, mode='nearest')
        d2 = self.dec2(torch.cat([d2, e2], dim=1), emb)

        d1 = F.interpolate(d2, scale_factor=2, mode='nearest')
        d1 = self.dec1(torch.cat([d1, e1], dim=1), emb)

        return self.out(d1)


class NoiseSchedule:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)


# ============================================================
# DATA LOADING
# ============================================================

def load_all_spectrograms(data_dir, crops_per_track=3):
    """Load spectrograms from ALL tracks."""
    loader = AudioLoader()
    extractor = SpectralFeatureExtractor()

    spectrograms = []
    labels = []

    print("\nLoading ALL spectrograms...")
    start_time = time.time()

    for class_id, (circle_name, dirname) in CIRCLES.items():
        circle_dir = data_dir / dirname
        if not circle_dir.exists():
            print(f"  {circle_name}: NOT FOUND")
            continue

        files = list(find_audio_files(circle_dir))
        count = 0

        for f in files:
            try:
                waveform, _ = loader.load(f)
                mel = extractor.mel_spectrogram(waveform)

                if mel.shape[1] < 64:
                    continue

                # Multiple random crops per track
                for _ in range(crops_per_track):
                    start = np.random.randint(0, mel.shape[1] - 64)
                    crop = mel[:64, start:start+64]
                    crop = (crop + 40) / 40
                    crop = np.clip(crop, -1, 1)
                    spectrograms.append(torch.tensor(crop, dtype=torch.float32))
                    labels.append(class_id)
                    count += 1

            except Exception:
                continue

        print(f"  {circle_name}: {count} spectrograms")

    elapsed = time.time() - start_time
    print(f"\nTotal: {len(spectrograms)} spectrograms in {elapsed:.1f}s")

    X = torch.stack(spectrograms).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)

    return X, y


# ============================================================
# SAMPLING
# ============================================================

@torch.no_grad()
def sample(model, noise_schedule, class_id, n_samples=1, device='cpu', cfg_scale=3.0):
    model.eval()

    betas = noise_schedule.betas.to(device)
    alphas = noise_schedule.alphas.to(device)
    alpha_cumprod = noise_schedule.alpha_cumprod.to(device)

    x = torch.randn(n_samples, 1, 64, 64, device=device)
    labels = torch.full((n_samples,), class_id, device=device, dtype=torch.long)

    for t in reversed(range(noise_schedule.timesteps)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

        pred_cond = model(x, t_batch, class_label=labels)
        pred_uncond = model(x, t_batch, class_label=None)
        pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

        alpha_t = alphas[t]
        alpha_bar_t = alpha_cumprod[t]
        mean = (1 / alpha_t.sqrt()) * (x - (betas[t] / (1 - alpha_bar_t).sqrt()) * pred)

        if t > 0:
            var = betas[t] * (1 - alpha_cumprod[t-1]) / (1 - alpha_bar_t)
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = mean

    model.train()
    return x


def save_samples(model, noise_schedule, X, y, device, path, epoch):
    """Generate and save sample comparison image."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i in range(5):
        # Real
        idx = (y == i).nonzero()[0].item()
        axes[0, i].imshow(X[idx, 0].numpy(), aspect='auto', origin='lower', cmap='magma')
        axes[0, i].set_title(CIRCLE_NAMES[i])
        axes[0, i].axis('off')

        # Generated
        gen = sample(model, noise_schedule, i, 1, device, CONFIG['cfg_scale'])
        axes[1, i].imshow(gen[0, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='magma')
        axes[1, i].axis('off')

    axes[0, 0].text(-0.1, 0.5, 'Real', transform=axes[0, 0].transAxes,
                    fontsize=12, va='center', rotation=90, fontweight='bold')
    axes[1, 0].text(-0.1, 0.5, 'Gen', transform=axes[1, 0].transAxes,
                    fontsize=12, va='center', rotation=90, fontweight='bold')

    plt.suptitle(f'Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# TRAINING
# ============================================================

def clear_cache():
    """Clear memory caches to prevent OOM."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def train(resume=False):
    print("=" * 60)
    print("FULL TOUHOU DIFFUSION TRAINING")
    print("=" * 60)
    print(f"Config: {CONFIG}")

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    checkpoint_path = output_dir / "touhou_diffusion_checkpoint.pt"
    final_model_path = output_dir / "touhou_diffusion_full.pt"

    # Load data
    X, y = load_all_spectrograms(data_dir, crops_per_track=CONFIG['crops_per_track'])

    # Create model and optimizer
    model = TouhouUNet(in_ch=1, base_ch=64, num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=1e-6
    )
    noise_schedule = NoiseSchedule(timesteps=CONFIG['timesteps'])

    start_epoch = 0
    losses = []

    # Resume if requested
    if resume and checkpoint_path.exists():
        print(f"\nResuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        print(f"Resumed from epoch {start_epoch}")

    # Move schedule to device
    sqrt_alpha = noise_schedule.sqrt_alpha_cumprod.to(device)
    sqrt_one_minus = noise_schedule.sqrt_one_minus_alpha_cumprod.to(device)

    # DataLoader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    print(f"\nTraining from epoch {start_epoch} to {CONFIG['epochs']}...")
    print(f"Batches per epoch: {len(loader)}")

    training_start = time.time()

    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        for specs, labels in loader:
            specs, labels = specs.to(device), labels.to(device)

            t = torch.randint(0, CONFIG['timesteps'], (specs.size(0),), device=device)
            noise = torch.randn_like(specs)
            noisy = sqrt_alpha[t, None, None, None] * specs + sqrt_one_minus[t, None, None, None] * noise

            # CFG dropout
            if torch.rand(1).item() < CONFIG['cfg_dropout']:
                pred = model(noisy, t, class_label=None)
            else:
                pred = model(noisy, t, class_label=labels)

            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        epoch_time = time.time() - epoch_start

        # Progress
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - training_start
            remaining = elapsed / (epoch - start_epoch + 1) * (CONFIG['epochs'] - epoch - 1) if epoch > start_epoch else 0
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {avg_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s | ETA: {remaining/60:.0f}min")

        # Clear cache periodically
        if (epoch + 1) % CONFIG['cache_clear_every'] == 0:
            clear_cache()

        # Save checkpoint
        if (epoch + 1) % CONFIG['checkpoint_every'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'losses': losses,
            }, checkpoint_path)
            print(f"  Checkpoint saved: epoch {epoch+1}")

        # Save sample images
        if (epoch + 1) % CONFIG['sample_every'] == 0:
            sample_path = output_dir / f"touhou_full_epoch_{epoch+1}.png"
            save_samples(model, noise_schedule, X, y, device, sample_path, epoch+1)
            print(f"  Samples saved: {sample_path}")
            clear_cache()

    # Final save
    total_time = time.time() - training_start
    print(f"\nTraining complete in {total_time/60:.1f} minutes")

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'losses': losses,
        'circles': CIRCLES,
    }, final_model_path)
    print(f"Final model saved: {final_model_path}")

    # Final samples
    save_samples(model, noise_schedule, X, y, device, output_dir / "touhou_full_final.png", CONFIG['epochs'])

    # Loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(output_dir / "touhou_full_loss.png", dpi=150)
    plt.close()

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    train(resume=args.resume)
