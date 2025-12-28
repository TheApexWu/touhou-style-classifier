# Diffusion Models: A Comprehensive Reference

*From fundamentals to state-of-the-art architectures. Living document - add findings as you learn.*

---

## Table of Contents
1. [Core Intuition](#core-intuition)
2. [The Math](#the-math)
3. [Key Components](#key-components)
4. [Training vs Inference](#training-vs-inference)
5. [Conditioning (Text, Class, etc.)](#conditioning)
6. [Latent Diffusion](#latent-diffusion)
7. [Audio Diffusion](#audio-diffusion)
8. [Architectures (Spellbrush-Relevant)](#architectures)
9. [Implementation Patterns](#implementation-patterns)
10. [Resources](#resources)

---

## Core Intuition

### The Fundamental Insight

```
DESTRUCTION IS EASY. LEARNING TO REVERSE IT IS POWERFUL.

Step 1: Take any data (image, audio, etc.)
Step 2: Gradually add noise until it's pure static
Step 3: Train a neural network to REVERSE each step
Step 4: To generate: start from noise, apply learned reverse steps

This works because:
  - Forward process is fixed (just add Gaussian noise)
  - Reverse process is learned (predict what noise was added)
  - Each step is a SMALL change (easier to learn than one big jump)
```

### Visual: The Diffusion Process

```
FORWARD PROCESS (Fixed, not learned):
══════════════════════════════════════════════════════════════

  t=0          t=250        t=500        t=750        t=1000
  Clean        Light noise  Medium       Heavy        Pure noise

  ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐
  │▓▓▓▓▓▓│     │▓▓░▓▓▓│     │░▓░░▓░│     │░░░░░░│     │░░░░░░│
  │▓▓▓▓▓▓│ ──► │▓░▓▓░▓│ ──► │░░▓░░░│ ──► │░░░░░░│ ──► │░░░░░░│
  │▓▓▓▓▓▓│     │▓▓▓░▓▓│     │░░░░▓░│     │░░░░░░│     │░░░░░░│
  └──────┘     └──────┘     └──────┘     └──────┘     └──────┘

  q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)

  "Each step: slightly shrink signal, add small noise"


REVERSE PROCESS (Learned by neural network):
══════════════════════════════════════════════════════════════

  t=1000       t=750        t=500        t=250        t=0
  Pure noise   Emerging     Structure    Details      Clean!

  ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐
  │░░░░░░│     │░░░░░░│     │░▓░░▓░│     │▓▓░▓▓▓│     │▓▓▓▓▓▓│
  │░░░░░░│ ──► │░░░░░░│ ──► │░░▓░░░│ ──► │▓░▓▓░▓│ ──► │▓▓▓▓▓▓│
  │░░░░░░│     │░░░░░░│     │░░░░▓░│     │▓▓▓░▓▓│     │▓▓▓▓▓▓│
  └──────┘     └──────┘     └──────┘     └──────┘     └──────┘

  pθ(xₜ₋₁ | xₜ) = N(xₜ₋₁; μθ(xₜ, t), Σθ(xₜ, t))

  "Neural net predicts: what noise was added? Remove it."
```

### Why This Works Better Than GANs/VAEs

```
GANs (2014):
────────────
Generator tries to fool Discriminator
  ✗ Unstable training (min-max game)
  ✗ Mode collapse (generates same thing)
  ✗ Hard to control output

VAEs (2013):
────────────
Encode to latent → Decode back
  ✗ Blurry outputs (averaging effect)
  ✗ Limited expressiveness

DIFFUSION (2020):
─────────────────
Iteratively denoise
  ✓ Stable training (simple MSE loss)
  ✓ No mode collapse (probabilistic)
  ✓ High quality (iterative refinement)
  ✓ Easy to condition (classifier-free guidance)
  ✗ Slow inference (many steps needed)
```

---

## The Math

### Notation

```
x₀     = clean data (image, audio, etc.)
xₜ     = noisy version at timestep t
T      = total timesteps (typically 1000)
βₜ     = noise schedule at step t (small values, e.g., 0.0001 to 0.02)
αₜ     = 1 - βₜ
ᾱₜ     = cumulative product: α₁ × α₂ × ... × αₜ
ε      = noise ~ N(0, I)
εθ     = noise predicted by neural network
```

### Forward Process (Adding Noise)

```
SINGLE STEP:
  xₜ = √(1-βₜ) × xₜ₋₁ + √βₜ × ε

DIRECT JUMP (closed form - key insight!):
  xₜ = √ᾱₜ × x₀ + √(1-ᾱₜ) × ε

  This means: we can jump directly to ANY noise level!
  No need to iterate through all steps during training.
```

### Reverse Process (Removing Noise)

```
GOAL: Learn p(xₜ₋₁ | xₜ)

APPROACH 1 - Predict noise (most common):
  Network predicts: εθ(xₜ, t) ≈ ε

  Loss = ||ε - εθ(xₜ, t)||²

  "Given noisy image and timestep, predict the noise"

APPROACH 2 - Predict x₀ directly:
  Network predicts: x̂₀ = fθ(xₜ, t)

  Loss = ||x₀ - fθ(xₜ, t)||²

APPROACH 3 - Predict velocity (used in some models):
  v = √ᾱₜ × ε - √(1-ᾱₜ) × x₀
  Network predicts: vθ(xₜ, t) ≈ v
```

### The Training Algorithm

```python
# DDPM Training (simplified)
def train_step(model, x_0):
    # 1. Sample random timestep
    t = torch.randint(0, T, (batch_size,))

    # 2. Sample noise
    epsilon = torch.randn_like(x_0)

    # 3. Create noisy version (closed-form jump)
    x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * epsilon

    # 4. Predict noise
    epsilon_pred = model(x_t, t)

    # 5. Simple MSE loss
    loss = F.mse_loss(epsilon_pred, epsilon)

    return loss
```

### The Sampling Algorithm

```python
# DDPM Sampling (simplified)
def sample(model, shape):
    # Start from pure noise
    x = torch.randn(shape)

    # Iteratively denoise
    for t in reversed(range(T)):
        # Predict noise
        epsilon_pred = model(x, t)

        # Compute mean of p(x_{t-1} | x_t)
        mean = (1 / sqrt(alpha[t])) * (
            x - (beta[t] / sqrt(1 - alpha_bar[t])) * epsilon_pred
        )

        # Add noise (except at t=0)
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + sqrt(beta[t]) * noise
        else:
            x = mean

    return x
```

---

## Key Components

### 1. Noise Schedule

```
The noise schedule βₜ controls how fast we add noise.

LINEAR (original DDPM):
  β₁ = 0.0001, βₜ = 0.02 (linear interpolation)
  Problem: Too slow at start, too fast at end

COSINE (improved):
  ᾱₜ = cos²((t/T + s) / (1+s) × π/2)
  Better: Smoother degradation of signal

LEARNED:
  Some models learn the schedule

         Signal remaining (ᾱₜ)
    1.0 ┤██████████████████████████████████
        │████████████████████████████░░░░░░  Linear
    0.5 ┤████████████████░░░░░░░░░░░░░░░░░░
        │██████████░░░░░░░░░░░░░░░░░░░░░░░░  Cosine
    0.0 ┤░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        └─────────────────────────────────→ t
         0                              1000
```

### 2. The Neural Network (Denoiser)

```
INPUT:  Noisy data xₜ + timestep t
OUTPUT: Predicted noise ε (or x₀, or v)

ARCHITECTURE CHOICES:

For Images - U-Net:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Input ──► [Down] ──► [Down] ──► [Middle] ──► [Up] ──► Output
│              64       128        256       128      64      │
│               │                              ▲              │
│               └──────── Skip Connection ─────┘              │
│                                                             │
│   + Time embedding injected at each layer                   │
│   + Self-attention at lower resolutions                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

For Sequences - Transformer:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   [Patch Embed] ──► [Transformer Blocks] ──► [Unpatch]      │
│                           │                                 │
│                    Self-attention                           │
│                    Cross-attention (for conditioning)       │
│                    + Time embedding                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. Time Embedding

```python
# Sinusoidal time embedding (like positional encoding in Transformers)
def get_time_embedding(timesteps, dim):
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half_dim) / half_dim
    )
    args = timesteps[:, None] * freqs[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return embedding

# Injected into network via:
# - Addition to feature maps
# - Adaptive normalization (scale and shift)
# - Cross-attention
```

---

## Training vs Inference

### Training

```
EFFICIENT: Only one forward pass per sample

for batch in dataloader:
    x_0 = batch                              # Clean data
    t = random_timesteps(batch_size)         # Random t ∈ [0, T]
    noise = torch.randn_like(x_0)            # Sample noise
    x_t = q_sample(x_0, t, noise)            # Add noise (closed form)
    noise_pred = model(x_t, t)               # Predict noise
    loss = F.mse_loss(noise_pred, noise)     # Simple loss
    loss.backward()
    optimizer.step()

TRAINING TIME:
  - Similar to other neural nets
  - Stable, no adversarial dynamics
  - Can use standard optimizers (Adam)
```

### Inference (Sampling)

```
SLOW: Need many forward passes (50-1000 steps)

DDPM (original):
  - 1000 steps
  - ~30 seconds per image on GPU

DDIM (faster):
  - Skip steps (50-100 steps)
  - Deterministic sampling option
  - ~3 seconds per image

DPM-Solver:
  - Even fewer steps (10-20)
  - Higher-order ODE solvers
  - ~0.5 seconds per image

CONSISTENCY MODELS:
  - 1-4 steps
  - Distilled from diffusion
  - Near real-time
```

---

## Conditioning

### Classifier-Free Guidance (CFG)

```
THE KEY TECHNIQUE for controllable generation.

IDEA: Train model both with and without conditioning.
      At inference, extrapolate AWAY from unconditional.

TRAINING:
  - 10-20% of time: drop condition (use null embedding)
  - Rest of time: use actual condition

INFERENCE:
  noise_pred_uncond = model(x_t, t, null_condition)
  noise_pred_cond = model(x_t, t, condition)

  # Extrapolate toward conditional
  noise_pred = noise_pred_uncond + guidance_scale * (
      noise_pred_cond - noise_pred_uncond
  )

  guidance_scale = 7.5 typical for images
                   (higher = more adherence to condition)

VISUAL:
  ────────────────────────────────────────────────────────────
  Unconditional        Conditional         High guidance
  (random image)       (matches prompt)    (strongly matches)

  guidance=1.0         guidance=7.5        guidance=15.0
  ────────────────────────────────────────────────────────────
```

### Types of Conditioning

```
TEXT (most common):
  - Encode text with CLIP/T5/BERT
  - Cross-attention in U-Net/Transformer
  - "A photo of a cat sitting on a chair"

CLASS LABEL:
  - Embed class ID
  - Add to time embedding
  - "Class 42 = tabby cat"

IMAGE:
  - ControlNet, IP-Adapter
  - Encode reference image
  - Use as additional conditioning

AUDIO:
  - Text descriptions ("upbeat electronic music")
  - Reference audio (style transfer)
  - Musical features (tempo, key, genre)
```

---

## Latent Diffusion

### The Problem with Pixel-Space Diffusion

```
RAW PIXELS:
  512×512×3 image = 786,432 dimensions
  Very expensive to process

  U-Net at full resolution:
  - Billions of parameters
  - Huge memory
  - Slow training
```

### The Solution: Compress First

```
LATENT DIFFUSION (Stable Diffusion, etc.):

┌───────────────────────────────────────────────────────────────┐
│                                                               │
│   Image ──► ENCODER ──► Latent ──► Diffusion ──► Latent ──► DECODER ──► Image
│   512×512     (VAE)     64×64      (U-Net)      64×64      (VAE)      512×512
│                          │                                    │
│                     8× smaller                           Reconstruct
│                                                               │
└───────────────────────────────────────────────────────────────┘

BENEFITS:
  ✓ 64× fewer dimensions (8×8 = 64)
  ✓ Much faster training/inference
  ✓ Latent space is "cleaner" (less redundancy)
  ✓ Same quality with less compute

THE VAE:
  - Trained separately (frozen during diffusion training)
  - Compresses to latent, reconstructs back
  - Stable Diffusion uses 4-channel latent (not 3-channel RGB)
```

---

## Audio Diffusion

### Representations for Audio

```
WAVEFORM (raw samples):
  ┌──────────────────────────────────────────┐
  │ ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿ │
  └──────────────────────────────────────────┘
  - 44,100 samples/second × seconds = huge!
  - Preserves phase perfectly
  - Hard to learn long-range structure

SPECTROGRAM (time-frequency):
  ┌──────────────────────────────────────────┐
  │ ░░▓▓░░░░▓▓░░▓▓▓▓░░░░░░▓▓▓▓▓▓░░░░░░░░ │  High freq
  │ ░░▓▓▓░░░▓▓▓░▓▓▓▓░░░░░▓▓▓▓▓▓▓░░░░░░░░ │
  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  Low freq
  └──────────────────────────────────────────┘
           Time →
  - 2D image-like (can use image diffusion!)
  - Loses phase (need vocoder to reconstruct)
  - Riffusion uses this approach

MEL SPECTROGRAM:
  - Like spectrogram but perceptually scaled
  - Matches human hearing
  - Common for audio ML

LATENT (compressed):
  - Encode audio with VAE/Encodec
  - Diffuse in latent space
  - AudioLDM, Stable Audio use this
```

### Audio Diffusion Models

```
RIFFUSION (2022):
────────────────
- Fine-tuned Stable Diffusion on spectrograms
- Text → Spectrogram → Audio
- Simple but effective
- Limited quality (spectrogram → audio artifacts)

AUDIOLDM (2023):
────────────────
- Latent diffusion for audio
- CLAP for text conditioning
- AudioMAE encoder
- Better quality than Riffusion

STABLE AUDIO (2024):
────────────────────
- Stability AI's music model
- Latent diffusion on compressed audio
- Long-form generation (up to 90 seconds)
- Text + timing conditioning

MUSICGEN (2023):
────────────────
- Meta's model (not diffusion - autoregressive)
- But relevant for comparison
- Uses Encodec for compression

DANCE DIFFUSION (2022):
───────────────────────
- Waveform diffusion (no compression)
- For music generation
- Open source, good for learning
```

---

## Architectures

### U-Net (Image Diffusion Standard)

```
THE WORKHORSE OF DIFFUSION

┌─────────────────────────────────────────────────────────────────────┐
│                              U-NET                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT: x_t (noisy image) + t (timestep) + c (condition)            │
│                                                                      │
│              ┌─────────┐                                            │
│              │ Time    │                                            │
│              │ Embed   │──────────────────────────────────┐         │
│              └─────────┘                                  │         │
│                                                           ▼         │
│  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐  │
│  │ Conv │──►│ Down │──►│ Down │──►│Middle│──►│  Up  │──►│  Up  │   │
│  │  In  │   │Block │   │Block │   │Block │   │Block │   │Block │   │
│  └──────┘   └──────┘   └──────┘   └──────┘   └──────┘   └──────┘   │
│     64        128        256        512        256        128       │
│               │                                  ▲                   │
│               └────────── Skip Connection ───────┘                   │
│                                                                      │
│  EACH BLOCK CONTAINS:                                               │
│    - ResNet blocks                                                   │
│    - Self-attention (at low resolutions)                            │
│    - Cross-attention for conditioning                                │
│    - Time embedding injection                                        │
│                                                                      │
│  OUTPUT: ε_θ (predicted noise)                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### DiT (Diffusion Transformer) - SPELLBRUSH RELEVANT

```
NEWER ARCHITECTURE, USED IN STATE-OF-THE-ART MODELS

Why Transformers for diffusion?
  - Scale better than U-Net
  - Used in Sora, SD3, FLUX
  - Spellbrush likely uses this for niji・journey

┌─────────────────────────────────────────────────────────────────────┐
│                              DiT                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT: x_t ──► Patchify ──► [P1, P2, P3, ..., Pn]                  │
│                              (like ViT)                              │
│                                                                      │
│         ┌─────────────┐                                             │
│         │ t embedding │                                             │
│         │ c embedding │  (condition)                                │
│         └──────┬──────┘                                             │
│                │                                                     │
│                ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │   TRANSFORMER BLOCK (× N layers)                            │   │
│  │                                                              │   │
│  │   ┌─────────────────────────────────────────────────────┐  │   │
│  │   │ LayerNorm ─► Self-Attention ─► LayerNorm ─► FFN     │  │   │
│  │   │                    │                                 │  │   │
│  │   │              (all patches                           │  │   │
│  │   │               attend to                             │  │   │
│  │   │               each other)                           │  │   │
│  │   └─────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  │   + AdaLN (Adaptive LayerNorm) modulated by t and c        │   │
│  │   + No U-Net style skip connections                         │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  OUTPUT: Unpatchify ──► ε_θ (predicted noise)                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

DiT ADVANTAGES:
  ✓ Scales predictably (more params = better)
  ✓ Simpler architecture than U-Net
  ✓ Better for very large models
  ✓ Unified architecture (same as LLMs)
```

### SD3 / FLUX Architecture - CUTTING EDGE

```
STABLE DIFFUSION 3 / BLACK FOREST LABS FLUX

KEY INNOVATION: MM-DiT (Multimodal DiT)
  - Text and image tokens processed TOGETHER
  - Not cross-attention, but JOINT attention

┌─────────────────────────────────────────────────────────────────────┐
│                            MM-DiT                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Text tokens:  [T1, T2, T3, ..., Tm]  (from T5/CLIP)                │
│  Image tokens: [I1, I2, I3, ..., In]  (from patchify)               │
│                                                                      │
│  CONCATENATE: [T1, T2, ..., Tm, I1, I2, ..., In]                    │
│                                                                      │
│  JOINT SELF-ATTENTION:                                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Every token attends to every other token                    │  │
│  │  Text ←→ Text, Text ←→ Image, Image ←→ Image                │  │
│  │                                                               │  │
│  │  (This is different from cross-attention where text is       │  │
│  │   separate and only attends to image)                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  RESULT: Better text-image alignment                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

FLUX SPECIFICS:
  - Rectified flow (not DDPM)
  - Faster sampling
  - Better text rendering
  - State-of-the-art quality
```

### Spellbrush Interview Deep Dive

```
NIJI・JOURNEY CONTEXT
═══════════════════════════════════════════════════════════════

What They Do:
  - Anime/illustration image generation
  - Collaboration with Midjourney
  - "Tight loop between research and production"
  - 4-person AI team, high ownership

Likely Architecture Stack:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  TEXT ENCODERS (frozen)                                     │
│    └─► T5-XXL + CLIP (like SD3/FLUX)                       │
│    └─► Possibly fine-tuned on anime captions               │
│                                                              │
│  DIFFUSION BACKBONE                                         │
│    └─► DiT or MM-DiT (Transformer-based)                   │
│    └─► Trained on TPUs using JAX                           │
│    └─► Custom modifications for anime style                │
│                                                              │
│  VAE (Latent Space)                                         │
│    └─► 8x or 16x compression                               │
│    └─► Possibly fine-tuned on anime for sharper lines      │
│                                                              │
│  TRAINING DATA                                               │
│    └─► Massive anime/illustration dataset                   │
│    └─► Danbooru tags as conditioning (character, style)    │
│    └─► Multi-resolution training                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Key Technical Challenges (Interview Topics):
─────────────────────────────────────────────

1. ANIME-SPECIFIC ISSUES
   - Line art quality: Diffusion often blurs lines
   - Color flatness: Anime uses solid colors vs gradients
   - Eye/face consistency: Critical for character accuracy
   - Resolution: 1024px+ for illustration quality

2. CONDITIONING COMPLEXITY
   - Character identity: Same character, different poses
   - Style control: Specific artist styles
   - Composition: Layouts, backgrounds, foreground
   - Danbooru tag understanding

3. SCALE & EFFICIENCY (Why JAX/TPU)
   - Training billions of images
   - TPU pod training (v4-256 or larger)
   - Mixed precision, gradient checkpointing
   - Distributed data parallel

Questions You Might Be Asked:
────────────────────────────
• "How would you improve anime line quality in diffusion?"
  → Fine-tune VAE decoder on line art, edge-aware loss

• "How does classifier-free guidance work?"
  → Train with 10% caption dropout, use CFG scale 7-12

• "DiT vs U-Net tradeoffs?"
  → DiT: scales better, simpler. U-Net: more inductive bias

• "How would you implement character consistency?"
  → Image conditioning (IP-Adapter), reference attention

• "JAX vs PyTorch for training?"
  → JAX: native TPU, functional. PyTorch: larger ecosystem

YOUR TOUHOU PROJECT AS TALKING POINT:
─────────────────────────────────────
"I built a Touhou doujin classifier that distinguishes 5 circles
with 90% accuracy. This taught me about:
  - Audio domain adaptation (MFCCs capture production style)
  - Why pretrained embeddings fail on niche domains
  - Feature engineering vs learned representations

For niji・journey, similar principles apply:
  - Anime has distinct features vs photorealism
  - Fine-tuning beats generic pretrained models
  - Domain expertise matters (I know Touhou/doujin culture)"
```

---

## Implementation Patterns

### Minimal Diffusion Training Loop

```python
import torch
import torch.nn.functional as F

class SimpleDiffusion:
    def __init__(self, model, T=1000):
        self.model = model
        self.T = T

        # Cosine schedule
        steps = torch.linspace(0, T, T + 1)
        alpha_bar = torch.cos((steps / T + 0.008) / 1.008 * torch.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]

        self.alpha_bar = alpha_bar
        self.beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        self.beta = torch.clamp(self.beta, 0.0001, 0.02)

    def q_sample(self, x_0, t, noise=None):
        """Forward process: add noise to x_0"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.alpha_bar[t].sqrt()
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bar[t]).sqrt()

        # Reshape for broadcasting
        sqrt_alpha_bar = sqrt_alpha_bar[:, None, None, None]
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar[:, None, None, None]

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def train_step(self, x_0):
        """One training step"""
        batch_size = x_0.shape[0]

        # Random timesteps
        t = torch.randint(0, self.T, (batch_size,))

        # Sample noise
        noise = torch.randn_like(x_0)

        # Create noisy sample
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        noise_pred = self.model(x_t, t)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def sample(self, shape):
        """Generate samples via reverse process"""
        x = torch.randn(shape)

        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, dtype=torch.long)

            # Predict noise
            noise_pred = self.model(x, t_batch)

            # Compute x_{t-1}
            alpha = 1 - self.beta[t]
            alpha_bar = self.alpha_bar[t]
            alpha_bar_prev = self.alpha_bar[t - 1] if t > 0 else 1.0

            # Mean
            coef1 = 1 / alpha.sqrt()
            coef2 = self.beta[t] / (1 - alpha_bar).sqrt()
            mean = coef1 * (x - coef2 * noise_pred)

            # Variance
            if t > 0:
                var = self.beta[t] * (1 - alpha_bar_prev) / (1 - alpha_bar)
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean

        return x
```

### Minimal U-Net for Diffusion

```python
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):
        # Sinusoidal embedding
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # Add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.skip(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_dim=256):
        super().__init__()

        self.time_embed = TimeEmbedding(time_dim)

        # Encoder
        self.down1 = ResBlock(in_ch, base_ch, time_dim * 4)
        self.down2 = ResBlock(base_ch, base_ch * 2, time_dim * 4)
        self.pool = nn.MaxPool2d(2)

        # Middle
        self.mid = ResBlock(base_ch * 2, base_ch * 2, time_dim * 4)

        # Decoder
        self.up2 = ResBlock(base_ch * 4, base_ch, time_dim * 4)  # *4 for skip
        self.up1 = ResBlock(base_ch * 2, base_ch, time_dim * 4)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Output
        self.out = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        # Encoder
        h1 = self.down1(x, t_emb)
        h2 = self.down2(self.pool(h1), t_emb)

        # Middle
        h = self.mid(self.pool(h2), t_emb)

        # Decoder with skip connections
        h = self.upsample(h)
        h = self.up2(torch.cat([h, h2], dim=1), t_emb)
        h = self.upsample(h)
        h = self.up1(torch.cat([h, h1], dim=1), t_emb)

        return self.out(h)
```

---

## Resources

### Papers (Chronological)
1. **DDPM** (2020): "Denoising Diffusion Probabilistic Models" - Ho et al.
2. **DDIM** (2021): "Denoising Diffusion Implicit Models" - Song et al.
3. **Classifier-Free Guidance** (2022): "Classifier-Free Diffusion Guidance" - Ho & Salimans
4. **Latent Diffusion** (2022): "High-Resolution Image Synthesis with Latent Diffusion Models" - Rombach et al.
5. **DiT** (2023): "Scalable Diffusion Models with Transformers" - Peebles & Xie
6. **SD3** (2024): "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" - Esser et al.

### Code Repositories
- [diffusers](https://github.com/huggingface/diffusers) - HuggingFace's library
- [k-diffusion](https://github.com/crowsonkb/k-diffusion) - Katherine Crowson's implementations
- [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) - Clean PyTorch

### For Audio
- [AudioLDM](https://github.com/haoheliu/AudioLDM)
- [Riffusion](https://github.com/riffusion/riffusion)
- [Dance Diffusion](https://github.com/Harmonai-org/sample-generator)

---

## Notes Section

*Add your learnings as you go:*

### [Date] - Topic
- Note 1
- Note 2

---

*Document created for Spellbrush prep. Last updated: 2024-12*
