# Diffusion Quality Improvement Plan

Concrete execution plan to improve Touhou conditional diffusion from proof-of-concept to presentable quality.

---

## Current State

```
Model:          TouhouUNet (64 base channels, ~4.2MB)
Data:           200 spectrograms (40 per circle)
Training:       100 epochs, ~5 min on M2
Resolution:     64×64 mel spectrograms
Quality:        Learns texture, weak conditioning
```

---

## Target State

```
Model:          TouhouUNet+ with attention (~50MB)
Data:           4000+ spectrograms (all available + augmentation)
Training:       500 epochs
Resolution:     64×64 (or 128×128 with latent)
Quality:        Clear circle-specific patterns, recognizable style transfer
```

---

## Execution Pipeline

### Phase A: Data Expansion (No GPU needed)

**What:**
- Use all 944 tracks instead of 200
- Extract multiple crops per track (5-10 random windows)
- Add data augmentation

**Code Changes:**
```python
# In experiment_touhou_diffusion.py, modify load_touhou_spectrograms():

samples_per_circle = None  # Use ALL tracks
crops_per_track = 5        # Multiple windows per song

# Add augmentation:
def augment_spectrogram(mel):
    # Time shift (already doing random crop)
    # Frequency masking
    if np.random.rand() < 0.3:
        f_start = np.random.randint(0, 50)
        f_width = np.random.randint(5, 15)
        mel[f_start:f_start+f_width, :] *= 0.5
    # Noise injection
    if np.random.rand() < 0.2:
        mel += np.random.randn(*mel.shape) * 0.05
    return mel
```

**Expected Data Size:**
```
IOSYS:           322 × 5 crops = 1,610
UNDEAD CORP:      61 × 5 crops =   305  (oversample 3× = 915)
暁Records:        279 × 5 crops = 1,395
SOUND HOLIC:     200 × 5 crops = 1,000
Liz Triangle:     82 × 5 crops =   410  (oversample 2× = 820)
────────────────────────────────────────
TOTAL:           ~5,740 spectrograms (29× more data)
```

**Time:** 10-15 min to extract all spectrograms
**GPU:** None needed
**Storage:** ~500MB for cached spectrograms

---

### Phase B: Training Improvements

**What:**
- Increase epochs to 500
- Add learning rate scheduling
- Add gradient clipping
- Log sample images during training

**Code Changes:**
```python
# Learning rate schedule
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Sample logging every 50 epochs
if (epoch + 1) % 50 == 0:
    sample = generate_sample(model, class_id=0)
    save_image(sample, f'outputs/training_sample_epoch_{epoch}.png')
```

**Time Estimates:**

| Setup | Epochs | Time per Epoch | Total |
|-------|--------|----------------|-------|
| M2 Mac (current) | 500 | ~30 sec | ~4 hours |
| M2 Mac (5.7k data) | 500 | ~3 min | ~25 hours |
| RTX 3090 | 500 | ~20 sec | ~3 hours |
| A100 | 500 | ~8 sec | ~1 hour |

**Recommendation:** Run overnight on M2, or use cloud GPU for faster iteration.

---

### Phase C: Architecture Improvements

**What:**
- Add self-attention at lower resolutions
- Increase base channels to 128
- Add more residual connections

**Code Changes:**
```python
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention
        q = q.view(B, C, -1).transpose(1, 2)  # B, HW, C
        k = k.view(B, C, -1)                   # B, C, HW
        v = v.view(B, C, -1).transpose(1, 2)  # B, HW, C

        attn = torch.softmax(q @ k / (C ** 0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).view(B, C, H, W)

        return x + self.proj(out)


class TouhouUNetWithAttention(nn.Module):
    def __init__(self, in_ch=1, base_ch=128, time_dim=256, num_classes=5):
        # ... same as before but with:
        # - base_ch=128 (was 64)
        # - time_dim=256 (was 128)
        # - Add SelfAttention after bottleneck

        self.bot_attn = SelfAttention(base_ch * 2)
```

**Model Size:**
```
Current (64 base):    ~4.2 MB,  ~1.1M params
With attention (128): ~50 MB,   ~13M params
```

**Time Impact:** ~2× slower per epoch due to attention
**Memory Impact:** ~2× more VRAM needed

---

### Phase D: Latent Diffusion (Optional, Major Effort)

**What:**
- Train a VAE to compress 64×64 → 8×8 latent
- Diffuse in latent space
- Decode back to spectrogram

**Why:**
- 64× fewer pixels to diffuse
- Much faster training
- Better for higher resolutions

**Effort:**
- VAE training: ~1 day
- Latent diffusion: ~1 day
- Total: 2-3 days of focused work

**Skip for now unless going to 256×256 resolution.**

---

## Recommended Execution Plan

### Option 1: M2 Mac Only (Budget)

```
Day 1 (2 hours active, overnight training):
  [x] Implement Phase A (data expansion)
  [x] Implement Phase B (training improvements)
  [ ] Start 500-epoch training run overnight

Day 2 (1 hour):
  [ ] Evaluate results
  [ ] If good: done
  [ ] If not: implement Phase C attention

Day 2-3 (overnight):
  [ ] Run Phase C training if needed

Total: ~2-3 days elapsed, 4-5 hours active work
Cost: $0
```

### Option 2: Cloud GPU (Faster Iteration)

```
Session 1 (2 hours on RTX 3090/A100):
  [ ] Upload data to cloud
  [ ] Run Phase A+B training (500 epochs)
  [ ] Evaluate

Session 2 (2 hours if needed):
  [ ] Implement Phase C
  [ ] Run training with attention
  [ ] Final evaluation

Total: ~4 hours GPU time
Cost: ~$4-8 (Lambda Labs @ $1-2/hr)
```

---

## Quick Wins vs Full Improvement

### Quick Win (Do Today, M2)
```
Change only:
  - samples_per_circle = 100 (instead of 40)
  - epochs = 300 (instead of 100)

Expected improvement: Noticeable but not dramatic
Time: ~2 hours on M2
```

### Full Improvement (Do Over Weekend)
```
All of Phase A + B + C

Expected improvement: Significant, interview-worthy
Time: 1-2 days elapsed
```

---

## Decision Points

Before executing, decide:

1. **Time budget?**
   - [ ] Quick win today (2 hrs)
   - [ ] Full improvement (1-2 days)

2. **Compute?**
   - [ ] M2 Mac only ($0, slower)
   - [ ] Cloud GPU ($5-10, faster)

3. **Architecture?**
   - [ ] Keep simple U-Net (faster, good enough?)
   - [ ] Add attention (better quality, 2× slower)

---

## Files to Modify

```
scripts/experiment_touhou_diffusion.py
  - load_touhou_spectrograms()  → Phase A
  - train()                     → Phase B
  - TouhouUNet                  → Phase C

New files (optional):
  scripts/cache_spectrograms.py   → Pre-extract all data
  src/models/attention_unet.py    → Phase C architecture
```

---

*Created: 2024-12*
