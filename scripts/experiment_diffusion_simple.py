"""
Simple Diffusion Experiment

A minimal, educational implementation of diffusion models.
Designed to run on M2 Mac without GPU requirements.

Goal: Understand the core diffusion loop by training on a tiny dataset.
We'll use 2D points (easy to visualize) before moving to spectrograms.

Usage:
    python scripts/experiment_diffusion_simple.py --demo      # Quick visualization
    python scripts/experiment_diffusion_simple.py --train     # Train on 2D data
    python scripts/experiment_diffusion_simple.py --spectrogram  # Train on mel spectrograms
"""

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# CORE DIFFUSION COMPONENTS
# =============================================================================

class NoiseSchedule:
    """
    The noise schedule defines how much noise to add at each timestep.

    Key insight: We use a "variance schedule" (beta_t) that increases over time.
    - t=0: Almost no noise (beta very small)
    - t=T: Mostly noise (beta larger)

    Two common schedules:
    - Linear: beta increases linearly (original DDPM)
    - Cosine: More gradual noise addition (better for low-res)
    """

    def __init__(self, timesteps=1000, schedule='linear'):
        self.timesteps = timesteps

        if schedule == 'linear':
            # Original DDPM schedule
            self.betas = torch.linspace(1e-4, 0.02, timesteps)
        elif schedule == 'cosine':
            # Improved schedule from "Improved DDPM"
            steps = torch.linspace(0, timesteps, timesteps + 1)
            alpha_bar = torch.cos((steps / timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.999)

        # Precompute useful quantities
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)

        # For q(x_t | x_0) - adding noise directly to clean data
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)

        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1 - self.alpha_cumprod_prev) / (1 - self.alpha_cumprod)

    def add_noise(self, x_0, t, noise=None):
        """
        Forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I)

        Key insight: We can jump directly to any timestep t!
        No need to iterate through all intermediate steps.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, *([1] * (x_0.dim() - 1)))
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t].view(-1, *([1] * (x_0.dim() - 1)))

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise


class SinusoidalPositionEmbedding(nn.Module):
    """
    Encode the timestep t as a vector using sinusoidal embeddings.

    Why? The model needs to know "how noisy is this input?"
    Same idea as positional encoding in Transformers.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


# =============================================================================
# SIMPLE NOISE PREDICTOR (MLP for 2D data)
# =============================================================================

class SimpleDenoisingMLP(nn.Module):
    """
    A minimal denoising network for 2D point data.

    Input: noisy point (2D) + timestep embedding
    Output: predicted noise (2D)

    This is the "epsilon_theta" in the DDPM paper.
    """

    def __init__(self, data_dim=2, hidden_dim=128, time_dim=64):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main network
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x, t):
        """
        x: (batch, 2) - noisy data points
        t: (batch,) - timestep indices
        """
        t_emb = self.time_embed(t.float())
        x_input = torch.cat([x, t_emb], dim=-1)
        return self.net(x_input)


# =============================================================================
# SIMPLE U-NET FOR SPECTROGRAMS
# =============================================================================

class SimpleUNet(nn.Module):
    """
    A minimal U-Net for image/spectrogram denoising.

    Architecture:
    - Encoder: Conv blocks that downsample
    - Bottleneck: Process at lowest resolution
    - Decoder: Conv blocks that upsample (with skip connections)

    Time embedding is added at each resolution level.
    """

    def __init__(self, in_channels=1, base_channels=32, time_dim=64):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 4)

        # Decoder (with skip connections, so input is 2x channels)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)

        # Output
        self.out = nn.Conv2d(base_channels, in_channels, 1)

        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x, t):
        """
        x: (batch, 1, H, W) - noisy spectrogram
        t: (batch,) - timestep indices
        """
        # Time embedding
        t_emb = self.time_embed(t.float())  # (batch, base_channels * 4)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck (add time embedding)
        b = self.bottleneck(self.pool(e3))
        b = b + t_emb.view(t_emb.size(0), -1, 1, 1)  # Add time info

        # Decoder with skip connections
        d3 = F.interpolate(b, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_diffusion(model, data_loader, noise_schedule, epochs=100, lr=1e-3, device='cpu'):
    """
    Training loop for diffusion models.

    Algorithm (from DDPM paper):
    1. Sample x_0 from data
    2. Sample t uniformly from {1, ..., T}
    3. Sample noise epsilon ~ N(0, I)
    4. Compute noisy x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
    5. Train model to predict epsilon from x_t
    6. Loss = MSE(predicted_epsilon, actual_epsilon)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Move schedule to device
    noise_schedule.sqrt_alpha_cumprod = noise_schedule.sqrt_alpha_cumprod.to(device)
    noise_schedule.sqrt_one_minus_alpha_cumprod = noise_schedule.sqrt_one_minus_alpha_cumprod.to(device)

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in data_loader:
            x_0 = batch[0].to(device)
            batch_size = x_0.size(0)

            # Sample random timesteps
            t = torch.randint(0, noise_schedule.timesteps, (batch_size,), device=device)

            # Sample noise and create noisy data
            noise = torch.randn_like(x_0)
            x_t, _ = noise_schedule.add_noise(x_0, t, noise)

            # Predict noise
            pred_noise = model(x_t, t)

            # Simple MSE loss
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    return losses


# =============================================================================
# SAMPLING (Reverse Process)
# =============================================================================

@torch.no_grad()
def sample_ddpm(model, noise_schedule, shape, device='cpu'):
    """
    DDPM sampling: iteratively denoise from pure noise.

    Algorithm:
    1. Start with x_T ~ N(0, I)
    2. For t = T, T-1, ..., 1:
       - Predict noise: epsilon_theta(x_t, t)
       - Compute x_{t-1} using posterior mean + variance
    3. Return x_0

    This is slow (1000 steps) but simple to understand.
    DDIM is faster (can skip steps).
    """
    model.eval()

    # Move schedule tensors to device
    betas = noise_schedule.betas.to(device)
    alphas = noise_schedule.alphas.to(device)
    alpha_cumprod = noise_schedule.alpha_cumprod.to(device)
    alpha_cumprod_prev = noise_schedule.alpha_cumprod_prev.to(device)
    posterior_variance = noise_schedule.posterior_variance.to(device)

    # Start from pure noise
    x = torch.randn(shape, device=device)

    # Iterate backwards through timesteps
    for t in reversed(range(noise_schedule.timesteps)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Predict noise
        pred_noise = model(x, t_batch)

        # Compute posterior mean
        alpha_t = alphas[t]
        alpha_bar_t = alpha_cumprod[t]
        beta_t = betas[t]

        # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon)
        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_bar_t)
        mean = coef1 * (x - coef2 * pred_noise)

        if t > 0:
            # Add noise (except at t=0)
            noise = torch.randn_like(x)
            sigma = torch.sqrt(posterior_variance[t])
            x = mean + sigma * noise
        else:
            x = mean

    return x


@torch.no_grad()
def sample_ddim(model, noise_schedule, shape, device='cpu', steps=50, eta=0.0):
    """
    DDIM sampling: deterministic (eta=0) or stochastic (eta>0).

    Key insight: We can skip timesteps! Instead of 1000 steps, use 50.
    When eta=0, sampling is deterministic (same noise -> same output).

    This is what most people use in practice.
    """
    model.eval()

    # Create timestep sequence (skip steps) - use float then convert
    timesteps = torch.linspace(noise_schedule.timesteps - 1, 0, steps + 1, device=device)
    timesteps = timesteps.long()

    alpha_cumprod = noise_schedule.alpha_cumprod.to(device)

    # Start from pure noise
    x = torch.randn(shape, device=device)

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_prev = timesteps[i + 1]

        t_batch = torch.full((shape[0],), t.item(), device=device, dtype=torch.long)

        # Predict noise
        pred_noise = model(x, t_batch)

        # Current and previous alpha_bar
        alpha_bar_t = alpha_cumprod[t]
        alpha_bar_prev = alpha_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

        # Predict x_0 and CLAMP to prevent explosion
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, -10, 10)  # Prevent numerical explosion

        # DDIM update (eta=0 means deterministic)
        if eta == 0:
            # Deterministic DDIM
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * pred_noise
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
        else:
            # Stochastic DDIM
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t + 1e-8)) * torch.sqrt(1 - alpha_bar_t / (alpha_bar_prev + 1e-8))
            dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=0)) * pred_noise
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * torch.randn_like(x)

    return x


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def create_2d_dataset(n_samples=1000, noise=0.1):
    """Create a simple 2D dataset (two moons or spiral)."""
    from sklearn.datasets import make_moons

    X, _ = make_moons(n_samples=n_samples, noise=noise)
    X = torch.tensor(X, dtype=torch.float32)
    return X


def demo_forward_process():
    """Visualize the forward (noising) process."""
    print("\n" + "=" * 60)
    print("DEMO: Forward Diffusion Process")
    print("=" * 60)

    # Create data
    X = create_2d_dataset(500)
    noise_schedule = NoiseSchedule(timesteps=1000)

    # Show data at different noise levels
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    timesteps = [0, 100, 300, 600, 999]

    for ax, t in zip(axes, timesteps):
        t_tensor = torch.full((len(X),), t)
        noisy_X, _ = noise_schedule.add_noise(X, t_tensor)

        ax.scatter(noisy_X[:, 0].numpy(), noisy_X[:, 1].numpy(), s=5, alpha=0.5)
        ax.set_title(f't = {t}')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')

    plt.suptitle('Forward Process: Gradually Adding Noise', fontsize=14)
    plt.tight_layout()

    save_path = Path(__file__).parent.parent / 'outputs' / 'diffusion_forward_demo.png'
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def demo_training_2d():
    """Train diffusion on 2D data and generate samples."""
    print("\n" + "=" * 60)
    print("DEMO: Training Diffusion on 2D Data")
    print("=" * 60)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create dataset
    X = create_2d_dataset(2000, noise=0.05)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Create model and schedule
    model = SimpleDenoisingMLP(data_dim=2, hidden_dim=256)  # Larger model
    noise_schedule = NoiseSchedule(timesteps=1000, schedule='linear')  # More timesteps, linear schedule

    # Train longer for better results
    print("\nTraining (this should take ~2 minutes)...")
    losses = train_diffusion(model, loader, noise_schedule, epochs=200, lr=1e-3, device=device)

    # Generate samples
    print("\nGenerating samples...")
    model.to(device)
    generated = sample_ddim(model, noise_schedule, shape=(500, 2), device=device, steps=100)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Training data
    axes[0].scatter(X[:, 0].numpy(), X[:, 1].numpy(), s=5, alpha=0.5, c='blue')
    axes[0].set_title('Training Data')
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-2, 2)

    # Generated samples
    gen_np = generated.cpu().numpy()
    axes[1].scatter(gen_np[:, 0], gen_np[:, 1], s=5, alpha=0.5, c='red')
    axes[1].set_title('Generated Samples')
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-2, 2)

    # Loss curve
    axes[2].plot(losses)
    axes[2].set_title('Training Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MSE Loss')

    plt.tight_layout()

    save_path = Path(__file__).parent.parent / 'outputs' / 'diffusion_2d_demo.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()

    print("\nDone! Check outputs/diffusion_2d_demo.png")


def demo_spectrogram_diffusion():
    """
    Train diffusion on mel spectrograms from your Touhou dataset.
    Uses a simple U-Net architecture.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data.audio import AudioLoader, find_audio_files
    from src.features.spectral import SpectralFeatureExtractor

    print("\n" + "=" * 60)
    print("DEMO: Spectrogram Diffusion")
    print("=" * 60)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load some spectrograms from your data
    loader = AudioLoader()
    extractor = SpectralFeatureExtractor()

    data_dir = Path(__file__).parent.parent / 'data' / 'raw' / 'IOSYS'

    if not data_dir.exists():
        print(f"\nData directory not found: {data_dir}")
        print("Generating synthetic spectrograms instead...")

        # Generate synthetic spectrograms
        n_samples = 100
        H, W = 64, 64  # Small for speed
        spectrograms = torch.randn(n_samples, 1, H, W) * 0.5

        # Add some structure (fake harmonics)
        for i in range(n_samples):
            for h in [10, 20, 30, 40]:
                spectrograms[i, 0, h:h+3, :] += torch.randn(3, W) * 0.3
    else:
        print(f"\nLoading spectrograms from: {data_dir}")
        files = find_audio_files(data_dir)[:50]  # Limit for speed

        spectrograms = []
        for f in files:
            try:
                waveform, _ = loader.load(f)
                mel = extractor.mel_spectrogram(waveform)

                # Crop to fixed size (64x64 for speed) and normalize
                # Take middle section of time axis for variety
                if mel.shape[1] < 64:
                    continue  # Skip short files

                # Random crop from time axis
                start = np.random.randint(0, mel.shape[1] - 64)
                mel = mel[:64, start:start+64]  # (64 freq_bins, 64 time_frames)

                # Normalize to roughly [-1, 1] range (mel is in dB, typically -80 to 0)
                mel = (mel + 40) / 40  # Shift and scale
                mel = np.clip(mel, -1, 1)

                spectrograms.append(torch.tensor(mel, dtype=torch.float32))
            except Exception as e:
                continue

        if len(spectrograms) < 10:
            print("Not enough spectrograms loaded, using synthetic data")
            n_samples = 100
            H, W = 64, 64
            spectrograms = torch.randn(n_samples, 1, H, W) * 0.5
        else:
            spectrograms = torch.stack(spectrograms).unsqueeze(1)

    print(f"Spectrogram shape: {spectrograms.shape}")

    # Create dataset
    dataset = TensorDataset(spectrograms)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Create model and schedule
    model = SimpleUNet(in_channels=1, base_channels=32)
    noise_schedule = NoiseSchedule(timesteps=500, schedule='cosine')

    # Train (short demo)
    print("\nTraining (quick demo, 50 epochs)...")
    losses = train_diffusion(model, loader, noise_schedule, epochs=50, lr=1e-3, device=device)

    # Generate samples
    print("\nGenerating spectrograms...")
    model.to(device)
    sample_shape = (4, 1, spectrograms.shape[2], spectrograms.shape[3])
    generated = sample_ddim(model, noise_schedule, shape=sample_shape, device=device, steps=50)

    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    # Real spectrograms
    for i in range(4):
        axes[0, i].imshow(spectrograms[i, 0].numpy(), aspect='auto', origin='lower', cmap='magma')
        axes[0, i].set_title(f'Real {i+1}')
        axes[0, i].axis('off')

    # Generated spectrograms
    for i in range(4):
        axes[1, i].imshow(generated[i, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='magma')
        axes[1, i].set_title(f'Generated {i+1}')
        axes[1, i].axis('off')

    plt.suptitle('Top: Real Spectrograms, Bottom: Generated', fontsize=14)
    plt.tight_layout()

    save_path = Path(__file__).parent.parent / 'outputs' / 'diffusion_spectrogram_demo.png'
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()

    # Also save loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses)
    ax.set_title('Spectrogram Diffusion Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')

    save_path = Path(__file__).parent.parent / 'outputs' / 'diffusion_spectrogram_loss.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("\nDone! Check outputs/diffusion_spectrogram_*.png")


def main():
    parser = argparse.ArgumentParser(description="Simple diffusion experiments")
    parser.add_argument('--demo', action='store_true', help='Quick forward process visualization')
    parser.add_argument('--train', action='store_true', help='Train on 2D data')
    parser.add_argument('--spectrogram', action='store_true', help='Train on spectrograms')
    args = parser.parse_args()

    if args.demo:
        demo_forward_process()
    elif args.train:
        demo_training_2d()
    elif args.spectrogram:
        demo_spectrogram_diffusion()
    else:
        print("Simple Diffusion Experiment")
        print("-" * 40)
        print("\nOptions:")
        print("  --demo         Quick visualization of forward process")
        print("  --train        Train diffusion on 2D data (fast)")
        print("  --spectrogram  Train on mel spectrograms")
        print("\nExample:")
        print("  python scripts/experiment_diffusion_simple.py --train")


if __name__ == "__main__":
    main()
