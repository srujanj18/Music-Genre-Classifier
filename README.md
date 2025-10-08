# Music Genre Classifier (GTZAN) — PyTorch + Flask

**What’s inside**
- `src/` : training, preprocessing, model, dataset, inference utilities
- `app/` : Flask web app (upload file -> predict)
- `requirements.txt` : Python dependencies
- `run_local.sh` : helper to create venv and install deps (Unix)

**Notes**
- This repo expects Python 3.8+.
- The GTZAN auto-download script attempts to fetch the dataset from a known mirror
  (if the direct download is blocked, download GTZAN manually and place it under `data/gtzan/`).
- See `src/config.py` to change paths and parameters.

**Quick start (local)**
1. Create virtualenv: `python3 -m venv venv && source venv/bin/activate`
2. Install deps: `pip install -r requirements.txt`
3. Preprocess (will attempt auto-download): `python -m src.preprocess_save --out data/spectrograms/ --gtzan-dir data/gtzan/`
4. Train: `python -m src.train --data data/spectrograms/ --epochs 30`
5. Run Flask app: `python app/app.py` and open `http://127.0.0.1:5000`
