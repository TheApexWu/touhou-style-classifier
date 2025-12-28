"""
Touhou Conditional Diffusion

Generate spectrograms in each circle's style using conditional diffusion.
This ties the diffusion learning back to the original classification project.

Usage:
    python scripts/experiment_touhou_diffusion.py              # Train and generate
    python scripts/experiment_touhou_diffusion.py --generate   # Generate only (requires saved model)
"""

import argparse
import math
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
# CIRCLE DEFINITIONS
# ============================================================

CIRCLES = {
    0: ('IOSYS', 'IOSYS'),
    1: ('UNDEAD CORPORATION', 'UNDEAD_CORPORATION'),
    2: ('暁Records', 'Akatsuki_Records'),
    3: ('SOUND HOLIC', 'SOUND_HOLIC'),
    4: ('Liz Triangle', 'Liz_Triangle'),
}

CIRCLE_NAMES = ['IOSYS', 'UNDEAD CORP', 'Akatsuki', 'SOUND HOLIC', 'Liz Triangle']


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class SinusoidalEmbedding(nn.Module):
    """Encode timestep as sinusoidal embedding (like Transformer positional encoding)."""
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
    """Convolutional block with time conditioning."""
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
    """
    U-Net for Touhou spectrogram diffusion.

    Conditional on circle (0-4) using class embedding.
    Same architecture principle as Stable Diffusion, just smaller.
    """
    def __init__(self, in_ch=1, base_ch=64, time_dim=128, num_classes=5):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim * 2),
        )

        # Class embedding (circle identity)
        # +1 for null class (unconditional, used in CFG)
        self.class_embed = nn.Embedding(num_classes + 1, time_dim * 2)

        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch, time_dim * 2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, time_dim * 2)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bot = ConvBlock(base_ch * 2, base_ch * 2, time_dim * 2)

        # Decoder (with skip connections)
        self.dec2 = ConvBlock(base_ch * 4, base_ch, time_dim * 2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, time_dim * 2)

        self.out = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t, class_label=None):
        # Time + class embedding
        t_emb = self.time_embed(t)

        if class_label is None:
            # Null class for classifier-free guidance
            c_emb = self.class_embed(torch.full((x.size(0),), 5, device=x.device))
        else:
            c_emb = self.class_embed(class_label)

        emb = t_emb + c_emb

        # U-Net forward
        e1 = self.enc1(x, emb)
        e2 = self.enc2(self.pool(e1), emb)
        b = self.bot(self.pool(e2), emb)

        d2 = F.interpolate(b, scale_factor=2, mode='nearest')
        d2 = self.dec2(torch.cat([d2, e2], dim=1), emb)

        d1 = F.interpolate(d2, scale_factor=2, mode='nearest')
        d1 = self.dec1(torch.cat([d1, e1], dim=1), emb)

        return self.out(d1)


# ============================================================
# NOISE SCHEDULE
# ============================================================

class NoiseSchedule:
    """Linear noise schedule for diffusion."""
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

def load_touhou_spectrograms(data_dir, samples_per_circle=40):
    """Load mel spectrograms from all 5 circles."""
    loader = AudioLoader()
    extractor = SpectralFeatureExtractor()

    spectrograms = []
    labels = []

    print("Loading spectrograms...")

    for class_id, (circle_name, dirname) in CIRCLES.items():
        circle_dir = data_dir / dirname
        if not circle_dir.exists():
            print(f"  {circle_name}: NOT FOUND")
            continue

        files = list(find_audio_files(circle_dir))
        np.random.shuffle(files)

        count = 0
        for f in files:
            if count >= samples_per_circle:
                break
            try:
                waveform, _ = loader.load(f)
                mel = extractor.mel_spectrogram(waveform)

                if mel.shape[1] < 64:
                    continue

                # Random crop to 64x64
                start = np.random.randint(0, mel.shape[1] - 64)
                mel = mel[:64, start:start+64]

                # Normalize to [-1, 1]
                mel = (mel + 40) / 40
                mel = np.clip(mel, -1, 1)

                spectrograms.append(torch.tensor(mel, dtype=torch.float32))
                labels.append(class_id)
                count += 1
            except Exception:
                continue

        print(f"  {circle_name}: {count} samples")

    X = torch.stack(spectrograms).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)

    return X, y


# ============================================================
# TRAINING
# ============================================================

def train(model, X, y, noise_schedule, epochs=100, device='cpu', cfg_dropout=0.1):
    """Train diffusion model with classifier-free guidance."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    sqrt_alpha = noise_schedule.sqrt_alpha_cumprod.to(device)
    sqrt_one_minus = noise_schedule.sqrt_one_minus_alpha_cumprod.to(device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for specs, labels in loader:
            specs, labels = specs.to(device), labels.to(device)

            t = torch.randint(0, noise_schedule.timesteps, (specs.size(0),), device=device)
            noise = torch.randn_like(specs)
            noisy = sqrt_alpha[t, None, None, None] * specs + sqrt_one_minus[t, None, None, None] * noise

            # CFG dropout: 10% unconditional
            if torch.rand(1).item() < cfg_dropout:
                pred = model(noisy, t, class_label=None)
            else:
                pred = model(noisy, t, class_label=labels)

            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    return model


# ============================================================
# SAMPLING
# ============================================================

@torch.no_grad()
def sample(model, noise_schedule, class_id, n_samples=1, device='cpu', cfg_scale=3.0):
    """Generate spectrograms with classifier-free guidance."""
    model.eval()

    betas = noise_schedule.betas.to(device)
    alphas = noise_schedule.alphas.to(device)
    alpha_cumprod = noise_schedule.alpha_cumprod.to(device)

    x = torch.randn(n_samples, 1, 64, 64, device=device)
    labels = torch.full((n_samples,), class_id, device=device, dtype=torch.long)

    for t in reversed(range(noise_schedule.timesteps)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

        # Classifier-free guidance
        pred_cond = model(x, t_batch, class_label=labels)
        pred_uncond = model(x, t_batch, class_label=None)
        pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

        # DDPM update step
        alpha_t = alphas[t]
        alpha_bar_t = alpha_cumprod[t]
        mean = (1 / alpha_t.sqrt()) * (x - (betas[t] / (1 - alpha_bar_t).sqrt()) * pred)

        if t > 0:
            var = betas[t] * (1 - alpha_cumprod[t-1]) / (1 - alpha_bar_t)
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = mean

    return x


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_results(model, noise_schedule, X, y, device, save_path):
    """Create comparison of real vs generated spectrograms."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # Top row: Real samples
    for i in range(5):
        idx = (y == i).nonzero()[0].item()
        axes[0, i].imshow(X[idx, 0].numpy(), aspect='auto', origin='lower', cmap='magma')
        axes[0, i].set_title(CIRCLE_NAMES[i], fontsize=10)
        axes[0, i].axis('off')

    axes[0, 0].text(-0.15, 0.5, 'Real', transform=axes[0, 0].transAxes,
                    fontsize=12, va='center', rotation=90, fontweight='bold')

    # Bottom row: Generated
    for i in range(5):
        gen = sample(model, noise_schedule, class_id=i, n_samples=1, device=device, cfg_scale=3.0)
        axes[1, i].imshow(gen[0, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='magma')
        axes[1, i].axis('off')

    axes[1, 0].text(-0.15, 0.5, 'Generated', transform=axes[1, 0].transAxes,
                    fontsize=12, va='center', rotation=90, fontweight='bold')

    plt.suptitle('Touhou Conditional Diffusion: Generate Spectrogram by Circle Style', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Touhou Conditional Diffusion")
    parser.add_argument('--generate', action='store_true', help='Generate only (load saved model)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--samples-per-circle', type=int, default=40, help='Samples per circle')
    args = parser.parse_args()

    print("=" * 60)
    print("TOUHOU CONDITIONAL DIFFUSION")
    print("Generate spectrograms in each circle's style")
    print("=" * 60)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "touhou_diffusion_model.pt"

    noise_schedule = NoiseSchedule(timesteps=1000)

    if args.generate and model_path.exists():
        # Load saved model
        print(f"\nLoading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model = TouhouUNet(in_ch=1, base_ch=64, num_classes=5)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        # Still need data for visualization
        X, y = load_touhou_spectrograms(data_dir, samples_per_circle=5)
    else:
        # Train from scratch
        X, y = load_touhou_spectrograms(data_dir, samples_per_circle=args.samples_per_circle)
        print(f"\nTotal: {len(X)} spectrograms")

        model = TouhouUNet(in_ch=1, base_ch=64, num_classes=5)
        model = train(model, X, y, noise_schedule, epochs=args.epochs, device=device)

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'circles': CIRCLES,
        }, model_path)
        print(f"\nSaved model: {model_path}")

    # Generate and visualize
    print("\nGenerating spectrograms for each circle...")
    visualize_results(model, noise_schedule, X, y, device, output_dir / "touhou_conditional_diffusion.png")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
