# Touhou Arrangement Style Classifier

ML classifier that identifies which doujin circle (fan arrangement group) created a Touhou arrangement based on audio features. Trained on 954 tracks from 5 major circles.

**Live demo**: [touhou-web.vercel.app/classifier](https://touhou-web.vercel.app/classifier)

## Results

### Classification Accuracy

| Model | Accuracy | Notes |
|-------|----------|-------|
| **Random Forest (expanded)** | **89.5%** ± 2.2% | Best model, 200 estimators, 431 features |
| Random Forest (stratified) | 80.0% ± 22.5% | GroupKFold by source track (stricter eval) |
| Random Forest (baseline) | 76.2% | Simple train/test split |

### Handcrafted vs Pretrained Embeddings

| Method | Accuracy | Feature Dim | Time/Sample |
|--------|----------|-------------|-------------|
| **Handcrafted** | **76.0%** | 431 | 2.28s |
| CLAP (pretrained) | 57.0% | 512 | 0.14s |
| MERT (music-specific) | 52.0% | 768 | 5.43s |

**Key finding**: Handcrafted features outperform pretrained audio embeddings by 19-24%. Domain-specific feature engineering beats transfer learning for niche music classification.

### Per-Circle Performance

| Circle | Accuracy | Style |
|--------|----------|-------|
| UNDEAD CORPORATION | 95% | Death metal (most distinctive) |
| 暁Records | 80% | Rock, vocal |
| Liz Triangle | 75% | Acoustic, folk |
| IOSYS | 70% | Electronic, denpa |
| SOUND HOLIC | 60% | Eurobeat, trance (hardest to classify) |

## Target Circles

| Circle | Style | Tracks |
|--------|-------|--------|
| IOSYS | Electronic, denpa | 324 |
| UNDEAD CORPORATION | Death metal | 63 |
| 暁Records (Akatsuki) | Rock, vocal | 281 |
| SOUND HOLIC | Eurobeat, trance | 202 |
| Liz Triangle | Acoustic, folk | 84 |

## Why Handcrafted > Pretrained?

1. **Domain mismatch**: CLAP/MERT trained on general audio, not Touhou arrangements
2. **Small dataset**: 954 tracks doesn't benefit from transfer learning
3. **Style vs content**: Pretrained models capture content (instruments, genre); circle style is subtler (production choices, mixing, arrangement patterns)
4. **Interpretability**: We can explain why UNDEAD CORPORATION is distinctive (low spectral centroid = dark/heavy, high spectral contrast = metal dynamics)

## Diffusion Experiments

Implemented DDPM from scratch as a learning exercise for generative modeling:

- **NoiseSchedule**: Linear and cosine β schedules
- **U-Net**: Skip connections, GroupNorm, sinusoidal time embeddings
- **Sampling**: DDPM (1000 steps) and DDIM (50 steps, deterministic)
- **Training**: 500 epochs on 2832 mel spectrograms (~5.5 hours on M2 Mac)
- **Conditioning**: Class-conditioned with classifier-free guidance (CFG scale 3.0)

See `scripts/experiment_diffusion_simple.py` for the educational implementation.

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

## Feature Extraction (431 dimensions)

- **Mel spectrogram**: 128 mels, summarized (mean, std per band) = 256
- **MFCCs**: 20 coefficients + delta + delta-delta = 60
- **Chroma**: 12 pitch classes
- **Spectral contrast**: 7 bands
- **Spectral stats**: Centroid, bandwidth, rolloff, flatness (mean, std, min, max each) = 16
- **Tempo**: BPM estimate

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

## Related Projects

- [touhou-composition-analysis](https://github.com/TheApexWu/touhou-composition-analysis): Computational musicology analyzing ZUN's 379 original tracks across 19 games
- [touhou-web](https://github.com/TheApexWu/touhou-web): Interactive web demo for both projects

## License

MIT
