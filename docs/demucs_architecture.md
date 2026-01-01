# Demucs: Deep Music Source Separation

*A comprehensive technical reference for understanding neural audio source separation*

---

## Table of Contents
1. [What is Source Separation?](#what-is-source-separation)
2. [The Evolution of Demucs](#the-evolution-of-demucs)
3. [Architecture Deep Dive](#architecture-deep-dive)
4. [Training and Loss Functions](#training-and-loss-functions)
5. [Industry and Research Applications](#industry-and-research-applications)
6. [When to Use Demucs](#when-to-use-demucs)
7. [Alternatives and Trade-offs](#alternatives-and-trade-offs)

---

## What is Source Separation?

Source separation is the task of isolating individual sound sources from a mixture. Given a song, we want to extract:

```
                    ┌─────────────┐
                    │   Vocals    │ ♪ Lead singer, backing vocals
                    ├─────────────┤
   Mixed Audio  ──► │    Drums    │ ♪ Kick, snare, hi-hat, cymbals
      (Song)        ├─────────────┤
                    │    Bass     │ ♪ Bass guitar, synth bass
                    ├─────────────┤
                    │    Other    │ ♪ Guitar, piano, synths, etc.
                    └─────────────┘
```

**The Fundamental Challenge**: Audio mixing is a *lossy* operation. When you sum waveforms, phase information interacts destructively and constructively. There's no closed-form inverse - separation requires learning the statistical regularities of what each source "looks like" in the time-frequency domain.

**Historical Approaches**:
- **ICA (Independent Component Analysis)**: Assumes sources are statistically independent. Works for simple cases, fails on music.
- **NMF (Non-negative Matrix Factorization)**: Learns spectral templates. Better, but limited expressiveness.
- **HPSS (Harmonic-Percussive Separation)**: Classical signal processing. Fast but coarse.
- **Deep Learning**: Learn the separation function directly from data. State-of-the-art since ~2018.

---

## The Evolution of Demucs

Demucs is developed by **Meta AI Research** (formerly Facebook AI Research). The name combines "de-" (separation) + "mucs" (music).

### Timeline

| Version | Year | Key Innovation |
|---------|------|----------------|
| Demucs v1 | 2019 | Waveform-domain U-Net |
| Demucs v2 | 2021 | Hybrid time-frequency approach |
| Hybrid Demucs (htdemucs) | 2022 | Transformer attention + hybrid domain |
| Demucs v4 | 2023 | Improved architecture, better drums |

### Why Waveform Domain?

Most audio models work in the **spectrogram domain** (time-frequency representation). Demucs was revolutionary for working directly on **raw waveforms**:

```
  Spectrogram-based                    Waveform-based (Demucs)
  ─────────────────                    ─────────────────────────

  Waveform ──► STFT ──► Model ──► iSTFT ──► Waveform
        │                                        │
        └── Phase information often lost ────────┘

  vs.

  Waveform ──────────► Model ──────────► Waveform
                         │
                 No phase artifacts
```

**Advantage**: Waveform models preserve phase perfectly, avoiding the "phasiness" artifacts common in spectrogram methods.

**Disadvantage**: Waveforms are high-dimensional (44,100 samples/second × 2 channels). Requires efficient architectures.

---

## Architecture Deep Dive

### The Hybrid Transformer Demucs (htdemucs)

The current state-of-the-art model combines three key components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HYBRID TRANSFORMER DEMUCS                             │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                         INPUT: Stereo Waveform                        │  │
│   │                      [2 channels × ~10 sec × 44.1kHz]                 │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                     │                                        │
│                    ┌────────────────┴────────────────┐                      │
│                    ▼                                 ▼                      │
│   ┌─────────────────────────┐         ┌─────────────────────────┐          │
│   │    TEMPORAL ENCODER     │         │   SPECTRAL ENCODER      │          │
│   │    (Waveform Domain)    │         │   (STFT Domain)         │          │
│   │                         │         │                         │          │
│   │  Conv1D layers with     │         │  Conv2D layers on       │          │
│   │  stride=4 downsampling  │         │  magnitude spectrogram  │          │
│   │                         │         │                         │          │
│   │  4 encoder layers:      │         │  4 encoder layers:      │          │
│   │  • 48 → 96 channels     │         │  • 48 → 96 channels     │          │
│   │  • 96 → 192 channels    │         │  • 96 → 192 channels    │          │
│   │  • 192 → 384 channels   │         │  • 192 → 384 channels   │          │
│   │  • 384 → 768 channels   │         │  • 384 → 768 channels   │          │
│   └───────────┬─────────────┘         └───────────┬─────────────┘          │
│               │                                   │                         │
│               └───────────────┬───────────────────┘                         │
│                               ▼                                             │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                      CROSS-DOMAIN TRANSFORMER                         │  │
│   │                                                                       │  │
│   │    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │  │
│   │    │   Self      │    │   Cross     │    │   Self      │             │  │
│   │    │ Attention   │───►│ Attention   │───►│ Attention   │             │  │
│   │    │ (Temporal)  │    │ (T↔S)       │    │ (Spectral)  │             │  │
│   │    └─────────────┘    └─────────────┘    └─────────────┘             │  │
│   │                                                                       │  │
│   │    5 transformer layers with:                                         │  │
│   │    • 8 attention heads                                                │  │
│   │    • 384/512 hidden dimension                                         │  │
│   │    • Cross-attention between domains                                  │  │
│   └───────────────────────────────┬──────────────────────────────────────┘  │
│                                   │                                         │
│                    ┌──────────────┴──────────────┐                         │
│                    ▼                             ▼                         │
│   ┌─────────────────────────┐         ┌─────────────────────────┐          │
│   │    TEMPORAL DECODER     │         │   SPECTRAL DECODER      │          │
│   │                         │         │                         │          │
│   │  Transposed Conv1D      │         │  Transposed Conv2D      │          │
│   │  with skip connections  │         │  with skip connections  │          │
│   │  from encoder           │         │  from encoder           │          │
│   │                         │         │                         │          │
│   │  × 4 sources            │         │  × 4 sources            │          │
│   └───────────┬─────────────┘         └───────────┬─────────────┘          │
│               │                                   │                         │
│               └───────────────┬───────────────────┘                         │
│                               ▼                                             │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                         FUSION LAYER                                  │  │
│   │                                                                       │  │
│   │    Learned weighted combination of temporal and spectral outputs      │  │
│   │    Output: 4 stems × 2 channels × samples                            │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                     │                                        │
│                                     ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │     OUTPUT: 4 Separated Stems (Vocals, Drums, Bass, Other)           │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. The U-Net Encoder-Decoder

The backbone is a **U-Net architecture**, proven effective for image segmentation and adapted for 1D audio:

```
ENCODER (Downsampling)                    DECODER (Upsampling)

Input: [2, 441000]                        Output: [8, 441000] (4 sources × 2 ch)
        │                                         ▲
        ▼                                         │
┌───────────────┐                         ┌───────────────┐
│ Conv1D        │                         │ TransConv1D   │
│ k=8, s=4      │ ─────────────────────►  │ k=8, s=4      │
│ 2→48 ch       │      skip connection    │ 96→8 ch       │
└───────┬───────┘                         └───────▲───────┘
        │ /4                                      │ ×4
        ▼                                         │
┌───────────────┐                         ┌───────────────┐
│ Conv1D        │                         │ TransConv1D   │
│ k=8, s=4      │ ─────────────────────►  │ k=8, s=4      │
│ 48→96 ch      │      skip connection    │ 192→48 ch     │
└───────┬───────┘                         └───────▲───────┘
        │ /4                                      │ ×4
        ▼                                         │
┌───────────────┐                         ┌───────────────┐
│ Conv1D        │                         │ TransConv1D   │
│ k=8, s=4      │ ─────────────────────►  │ k=8, s=4      │
│ 96→192 ch     │      skip connection    │ 384→96 ch     │
└───────┬───────┘                         └───────▲───────┘
        │ /4                                      │ ×4
        ▼                                         │
┌───────────────┐                         ┌───────────────┐
│ Conv1D        │                         │ TransConv1D   │
│ k=8, s=4      │ ─────────────────────►  │ k=8, s=4      │
│ 192→384 ch    │      skip connection    │ 768→192 ch    │
└───────┬───────┘                         └───────▲───────┘
        │ /4                                      │
        ▼                                         │
    BOTTLENECK ────► TRANSFORMER ─────────────────┘
    [384, 430]
```

**Key insight**: Each encoder layer reduces temporal resolution by 4× while increasing channel depth. The decoder reverses this. Skip connections preserve fine-grained details.

**Receptive field**: After 4 layers of stride-4, the bottleneck sees the full ~10 second context. This is crucial for understanding musical structure.

#### 2. The Transformer Bottleneck

At the bottleneck, a **Transformer** processes the compressed representation:

```
┌────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER BLOCK (×5)                       │
│                                                                 │
│   Input: [batch, seq_len, hidden_dim]                          │
│           └─ ~430 tokens for 10 seconds                        │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  Multi-Head Self-Attention                               │  │
│   │                                                          │  │
│   │       Q ────┐                                            │  │
│   │             │     Attention                              │  │
│   │       K ────┼───► Weights ───► Weighted Sum              │  │
│   │             │     (softmax)                              │  │
│   │       V ────┘                                            │  │
│   │                                                          │  │
│   │  Each token attends to ALL other tokens                  │  │
│   │  8 heads capture different musical relationships:        │  │
│   │    • Head 1: rhythmic patterns                           │  │
│   │    • Head 2: harmonic progressions                       │  │
│   │    • Head 3: vocal phrases                               │  │
│   │    • ...                                                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  Layer Norm + Residual Connection                        │  │
│   └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  Feed-Forward Network                                    │  │
│   │                                                          │  │
│   │  Linear(384 → 1536) → GELU → Linear(1536 → 384)         │  │
│   │                                                          │  │
│   │  4× expansion allows learning complex transformations    │  │
│   └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  Layer Norm + Residual Connection                        │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**Why Transformer?** Self-attention allows each time position to "see" the entire song. This is crucial for:
- Understanding that a vocal note at t=5s is part of a phrase starting at t=3s
- Recognizing drum patterns that repeat across the song
- Separating guitar from bass based on their different spectral evolution

#### 3. Cross-Domain Attention (Hybrid Models)

The hybrid model processes **both** waveform and spectrogram representations:

```
     TEMPORAL PATH                      SPECTRAL PATH
     (Waveform)                         (Spectrogram)
          │                                   │
          ▼                                   ▼
   ┌─────────────┐                     ┌─────────────┐
   │  Temporal   │                     │  Spectral   │
   │  Encoder    │                     │  Encoder    │
   └──────┬──────┘                     └──────┬──────┘
          │                                   │
          ▼                                   ▼
   ┌─────────────┐                     ┌─────────────┐
   │   Self-     │                     │   Self-     │
   │  Attention  │                     │  Attention  │
   │  (temporal) │                     │ (spectral)  │
   └──────┬──────┘                     └──────┬──────┘
          │                                   │
          └──────────────┬────────────────────┘
                         ▼
               ┌─────────────────┐
               │ Cross-Attention │
               │                 │
               │  Q from temporal│
               │  K,V from       │
               │  spectral       │
               │  (and vice      │
               │   versa)        │
               └────────┬────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   ┌─────────────┐             ┌─────────────┐
   │  Temporal   │             │  Spectral   │
   │  Decoder    │             │  Decoder    │
   └──────┬──────┘             └──────┬──────┘
          │                           │
          └──────────────┬────────────┘
                         ▼
                  ┌─────────────┐
                  │   Fusion    │
                  │             │
                  │  Learned    │
                  │  weighted   │
                  │  average    │
                  └─────────────┘
```

**Why hybrid?**
- Spectrograms: Explicit frequency information (easier to distinguish bass from vocals)
- Waveforms: Perfect phase preservation (no artifacts)
- Cross-attention: Each domain can query relevant information from the other

---

## Training and Loss Functions

### The Training Data

Demucs is trained on **MUSDB18** (150 songs) + proprietary Meta datasets (10,000+ songs):

```
Training Sample Structure:
──────────────────────────

  mix.wav ──────────────────► Model ──────────────────► predicted_stems
                                                              │
                                                              ▼
  vocals.wav ─────────┐                              ┌── pred_vocals
  drums.wav ──────────┼── ground truth ──────────────┼── pred_drums
  bass.wav ───────────┤                              ├── pred_bass
  other.wav ──────────┘                              └── pred_other
                                                              │
                                                              ▼
                                                         L1 Loss
```

### Loss Function

The primary loss is **L1 (Mean Absolute Error)** in the waveform domain:

```python
def compute_loss(predicted_stems, target_stems):
    """
    L1 loss summed across all stems.

    predicted_stems: [batch, 4_stems, 2_channels, samples]
    target_stems: [batch, 4_stems, 2_channels, samples]
    """
    loss = 0
    for stem_idx in range(4):  # vocals, drums, bass, other
        pred = predicted_stems[:, stem_idx]
        target = target_stems[:, stem_idx]

        # L1 in waveform domain
        loss += torch.mean(torch.abs(pred - target))

        # Optional: L1 in STFT domain for spectral coherence
        pred_spec = torch.stft(pred, n_fft=4096, return_complex=True)
        target_spec = torch.stft(target, n_fft=4096, return_complex=True)
        loss += 0.1 * torch.mean(torch.abs(pred_spec - target_spec))

    return loss
```

**Why L1 over L2 (MSE)?**
- L1 is more robust to outliers
- L2 tends to produce "blurry" outputs that minimize average error
- L1 produces crisper, more natural-sounding separations

### Data Augmentation

Critical for generalization:

```
┌────────────────────────────────────────────────────────────────┐
│                    DATA AUGMENTATION                            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Random Stem Remixing                                       │
│     • Recombine stems from different songs                      │
│     • Prevents overfitting to specific mix balances            │
│                                                                 │
│  2. Pitch Shifting (±2 semitones)                              │
│     • Transpose stems independently                             │
│     • Helps with varied vocal ranges                            │
│                                                                 │
│  3. Time Stretching (±10%)                                     │
│     • Vary tempo without changing pitch                         │
│     • Improves robustness to different tempos                   │
│                                                                 │
│  4. Random EQ/Filtering                                         │
│     • Apply random frequency boosts/cuts                        │
│     • Simulates different mixing styles                         │
│                                                                 │
│  5. Channel Swapping                                            │
│     • Randomly swap L/R channels                                │
│     • Prevents overfitting to stereo positioning                │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Industry and Research Applications

### Music Production

```
┌─────────────────────────────────────────────────────────────┐
│                    COMMERCIAL APPLICATIONS                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. REMIXING & SAMPLING                                     │
│     └─► Isolate vocals for remixes without original stems   │
│     └─► Sample drum breaks from full mixes                  │
│     └─► Create acapellas from commercial releases           │
│                                                              │
│  2. KARAOKE GENERATION                                      │
│     └─► Remove vocals for karaoke tracks                    │
│     └─► Services: Lalal.ai, Moises.ai, iZotope RX          │
│                                                              │
│  3. TRANSCRIPTION & ANALYSIS                                │
│     └─► Isolate bass for transcription                      │
│     └─► Separate drums for rhythm analysis                  │
│     └─► Research: MIR (Music Information Retrieval)         │
│                                                              │
│  4. REMASTERING                                             │
│     └─► Adjust individual stem levels in old recordings     │
│     └─► Fix mixing issues without original multitrack       │
│                                                              │
│  5. GAME & FILM                                             │
│     └─► Interactive music (duck vocals during dialogue)     │
│     └─► Adaptive game soundtracks                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Research Applications

| Domain | Application | Citation |
|--------|-------------|----------|
| MIR | Melody extraction from vocals stem | Salamon et al. |
| ASR | Improved speech recognition in music | Watanabe et al. |
| Music Generation | Train on clean stems | Dhariwal et al. (Jukebox) |
| Cover Detection | Compare stems separately | Yesiler et al. |
| Mood Classification | Analyze instrumental vs vocal mood | Our project (Touhou) |

### Production Tools Using Demucs-like Technology

- **iZotope RX 10**: Professional audio repair ($$$$)
- **Lalal.ai**: Consumer-friendly web service
- **Moises.ai**: Mobile app for musicians
- **Ultimate Vocal Remover (UVR)**: Free, open-source GUI
- **Spleeter (Deezer)**: Open-source alternative (simpler architecture)

---

## When to Use Demucs

### Decision Framework

```
                        Start
                          │
                          ▼
              ┌─────────────────────┐
              │ Need stems for      │
              │ downstream task?    │
              └──────────┬──────────┘
                         │
            ┌────────────┴────────────┐
            │ YES                     │ NO
            ▼                         ▼
  ┌─────────────────────┐   ┌─────────────────────┐
  │ Is separation       │   │ Skip separation,    │
  │ quality critical?   │   │ use mixed audio     │
  └──────────┬──────────┘   └─────────────────────┘
             │
    ┌────────┴────────┐
    │ YES             │ NO
    ▼                 ▼
┌─────────────┐  ┌─────────────────────────────┐
│ Use Demucs  │  │ Can you tolerate artifacts? │
│ (htdemucs)  │  └──────────────┬──────────────┘
│             │                 │
│ ~30-60s/    │      ┌──────────┴──────────┐
│ track       │      │ YES                 │ NO
└─────────────┘      ▼                     ▼
              ┌─────────────┐       ┌─────────────┐
              │ Use HPSS    │       │ Use Demucs  │
              │ (classical) │       │ anyway      │
              │             │       │             │
              │ ~1s/track   │       │             │
              └─────────────┘       └─────────────┘
```

### For This Project (Touhou Classifier)

```
Current Situation:
──────────────────
• 828 samples
• Demucs: 828 × 30s = ~7 hours minimum
• HPSS: 828 × 1s = ~14 minutes

Recommendation:
───────────────
1. Run HPSS experiment first (~15 min)
2. If percussive features help → vocals ARE the problem
   └─► Then Demucs is worth the 7+ hours
3. If percussive features don't help → vocals NOT the problem
   └─► Don't waste time on Demucs
```

---

## Alternatives and Trade-offs

### Comparison Table

| Method | Speed | Quality | Artifacts | GPU Needed |
|--------|-------|---------|-----------|------------|
| **HPSS** | 1s/track | Low | Frequency bleed | No |
| **Spleeter** | 5s/track | Medium | Some bleed | Optional |
| **Demucs v4** | 30s/track | High | Minimal | Yes |
| **htdemucs** | 45s/track | Highest | Very low | Yes |

### HPSS vs Demucs Visual Comparison

```
                 HPSS                              Demucs
    (Harmonic-Percussive Split)         (Neural 4-stem Separation)

         Mixed Audio                          Mixed Audio
              │                                    │
              ▼                                    ▼
    ┌─────────────────┐                  ┌─────────────────┐
    │  Median Filter  │                  │  Deep Neural    │
    │  on Spectrogram │                  │  Network        │
    └────────┬────────┘                  └────────┬────────┘
             │                                    │
      ┌──────┴──────┐              ┌──────┬───────┴───────┬──────┐
      ▼             ▼              ▼      ▼               ▼      ▼
  Harmonic    Percussive      Vocals   Drums          Bass   Other

  Contains:   Contains:       Clean    Isolated       Clean   Rest
  • Vocals    • Drums         voice    transients     bass    (guitar,
  • Melody    • Attacks                                       synth,
  • Pads      • Clicks                                        piano)
  • Bass
  (blended)   (blended)
```

### When HPSS Suffices

HPSS is sufficient when you need:
- **Speed over quality**: Prototyping, hypothesis testing
- **Binary split**: Just harmonic vs percussive, not 4 stems
- **CPU-only environment**: No GPU available
- **Classical signal processing baseline**: For comparison

---

## Implementation Notes

### Running Demucs

```bash
# Install
pip install demucs torch torchaudio

# Basic usage (outputs to ./separated/htdemucs/track_name/)
python -m demucs track.mp3

# Specify model
python -m demucs -n htdemucs track.mp3      # Best quality
python -m demucs -n htdemucs_ft track.mp3   # Fine-tuned variant
python -m demucs -n mdx_extra track.mp3     # Experimental

# Batch processing
python -m demucs *.flac

# Output stems
# vocals.wav, drums.wav, bass.wav, other.wav
```

### Model Sizes

| Model | Parameters | Download | VRAM |
|-------|------------|----------|------|
| htdemucs | 42M | ~160MB | ~4GB |
| htdemucs_ft | 42M | ~160MB | ~4GB |
| mdx_extra | 58M | ~220MB | ~6GB |

---

## References

1. Défossez, A., et al. (2019). *Music Source Separation in the Waveform Domain*. arXiv:1911.13254
2. Défossez, A. (2021). *Hybrid Spectrogram and Waveform Source Separation*. ISMIR 2021
3. Rouard, S., et al. (2022). *Hybrid Transformers for Music Source Separation*. ICASSP 2023
4. Stoller, D., et al. (2018). *Wave-U-Net: A Multi-Scale Neural Network for Audio Source Separation*

---

*Document created for touhou-style-classifier project. Last updated: 2024-12*
