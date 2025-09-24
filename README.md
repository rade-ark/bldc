# BLDC AE + CNN on Mel Features

This repo reproduces the Kaggle notebook: BLDC AE CNN on mel features.

## Quickstart

1. Create virtualenv and install requirements

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Put your dataset in `data/raw/<label>/*.wav`.
3. Run preprocessing (optional)

```bash
bash scripts/preprocess.sh
```

4. Train autoencoder

```bash
python -m src.train --mode ae
```

5. Train classifier (after AE checkpoint exists)

```bash
python -m src.train --mode clf
```

6. Evaluate

```bash
python -m src.evaluate --ae checkpoints/ae.pth --clf checkpoints/classifier.pth
```
