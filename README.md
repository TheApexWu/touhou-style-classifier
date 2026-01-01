# Touhou Arrangement Style Classifier

Classify which doujin circle arranged a Touhou track based on audio features.

## Results

### Classification Accuracy

| Model | Accuracy | Notes |
|-------|----------|-------|
| **Random Forest (expanded)** | **89.5%** ± 2.2% | Best model, 200 estimators, 431 features |
| Random Forest (stratified) | 80.0% ± 22.5% | GroupKFold by source track (stricter) |
| Random Forest (baseline) | 76.2% | Simple train/test split |

### Embeddings Comparison

| Method | Accuracy | Feature Dim | Time/Sample |
|--------|----------|-------------|-------------|
| **Handcrafted** | **76.0%** | 431 | 2.28s |
| CLAP (pretrained) | 57.0% | 512 | 0.14s |
| MERT (music-specific) | 52.0% | 768 | 5.43s |

**Key finding:** Handcrafted features outperform pretrained audio embeddings by 19-24% on this task. Domain-specific feature engineering beats transfer learning for niche music classification.

### Per-Circle Performance (Handcrafted, 76% overall)

| Circle | Accuracy | Style |
|--------|----------|-------|
| UNDEAD CORPORATION | 95% | Death metal - most distinctive |
| 暁Records | 80% | Rock, vocal |
| Liz Triangle | 75% | Acoustic, folk |
| IOSYS | 70% | Electronic, denpa |
| SOUND HOLIC | 60% | Eurobeat, trance - hardest |

## Target Circles

| Circle | Style | Tracks |
|--------|-------|--------|
| IOSYS | Electronic, denpa | 324 |
| UNDEAD CORPORATION | Death metal | 63 |
| 暁Records (Akatsuki) | Rock, vocal | 281 |
| SOUND HOLIC | Eurobeat, trance | 202 |
| Liz Triangle | Acoustic, folk | 84 |

## Diffusion Experiments

Implemented DDPM from scratch for understanding generative modeling:

- **NoiseSchedule**: Linear and cosine β schedules
- **U-Net**: Skip connections, GroupNorm, sinusoidal time embeddings
- **Sampling**: Both DDPM (1000 steps) and DDIM (50 steps, deterministic)
- **Training**: 500 epochs on 2832 mel spectrograms (~5.5 hours on M2 Mac)
- **Conditional generation**: Class-conditioned with classifier-free guidance (CFG scale 3.0)

See `scripts/experiment_diffusion_simple.py` for educational implementation.

## Setup

```bash
cd touhou-style-classifier
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

1. Download metadata database:
```bash
mkdir -p data/metadata
wget -O data/metadata/touhou-music.db \
  https://github.com/solaasan/Touhou-Music-Database/raw/main/touhou-music.db
```

2. Add audio files to `data/raw/` organized by circle.

## Usage

```bash
# Test feature extraction
python runtest.py path/to/song.flac --save output.png

# Train baseline classifier
python scripts/train_baseline.py

# Train with source track stratification
python scripts/train_stratified.py

# Compare embeddings (CLAP vs MERT vs handcrafted)
python scripts/experiment_embeddings.py

# Run diffusion experiments
python scripts/experiment_diffusion_simple.py --train
python scripts/experiment_diffusion_simple.py --spectrogram
```

## Project Structure

```
src/
├── data/
│   ├── database.py    # SQLite queries
│   └── audio.py       # Audio loading
├── features/
│   └── spectral.py    # Mel spectrograms, MFCCs
└── models/
    └── classifier.py  # Circle classifier

scripts/
├── train_baseline.py           # Random Forest baseline
├── train_stratified.py         # GroupKFold by source track
├── experiment_embeddings.py    # CLAP vs MERT vs handcrafted
├── experiment_diffusion_simple.py  # DDPM from scratch
└── train_touhou_diffusion_full.py  # Full conditional diffusion

outputs/
├── embeddings_comparison_results.json
├── touhou_diffusion_full.pt    # Trained diffusion model
└── *.png                       # Visualizations
```

## Technical Details

### Feature Extraction (431 dimensions)

- **Mel spectrogram**: 128 mels, summarized (mean, std per band)
- **MFCCs**: 20 coefficients + delta + delta-delta
- **Chroma**: 12 pitch classes
- **Spectral contrast**: 7 bands
- **Spectral stats**: Centroid, bandwidth, rolloff, flatness (mean, std, min, max each)
- **Tempo**: BPM estimate

### Why Handcrafted > Pretrained?

1. **Domain mismatch**: CLAP/MERT trained on general audio, not Touhou arrangements
2. **Small dataset**: 954 tracks doesn't benefit from transfer learning
3. **Style vs content**: Pretrained models capture content (instruments, genre); circle style is more subtle (production choices, arrangement patterns)
4. **Interpretability**: We can explain why UNDEAD CORPORATION is distinctive (low spectral centroid, high spectral contrast)

## License

MIT
