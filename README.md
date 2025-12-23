# Touhou Arrangement Style Classifier

Classify which doujin circle arranged a Touhou track based on audio features.

## Target Circles

| Circle | Style |
|--------|-------|
| IOSYS | Electronic, denpa |
| UNDEAD CORPORATION | Death metal |
| 暁Records (Akatsuki) | Rock, vocal |
| SOUND HOLIC | Eurobeat, trance |
| Liz Triangle | Acoustic, folk |

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

Test feature extraction on an audio file:
```bash
python runtest.py path/to/song.mp3
python runtest.py path/to/song.flac --save output.png
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
```

## Approach

1. **Features**: Mel spectrograms (CNN) + MFCCs/spectral stats (Random Forest)
2. **Baseline**: Random Forest on summarized spectral features
3. **Deep**: CNN on mel spectrogram chunks
4. **Evaluation**: Per-circle accuracy, confusion matrix, t-SNE
