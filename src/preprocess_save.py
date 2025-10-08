"""
Preprocess GTZAN into mel-spectrogram .npy files.
It attempts to auto-download GTZAN from a public mirror if not present.
"""
import os, argparse, shutil, requests, tarfile
from tqdm import tqdm
from .utils import audio_to_mel, ensure_dir
from .config import GTZAN_DIR, SPECTRO_DIR, GENRES

GTZAN_URL = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'

def download_gtzan(dest):
    print('Downloading GTZAN (may take a while)...')
    r = requests.get(GTZAN_URL, stream=True, timeout=30)
    if r.status_code != 200:
        raise RuntimeError('Could not download GTZAN. Please download manually and place under data/gtzan.')
    tmp = dest + '.tar.gz'
    with open(tmp, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
    print('Extracting...')
    with tarfile.open(tmp, 'r:gz') as tar:
        tar.extractall(path=os.path.dirname(dest))
    os.remove(tmp)

def convert_all(gtzan_dir, out_dir):
    # Detect if 'genres_original' exists
    if os.path.isdir(os.path.join(gtzan_dir, "genres_original")):
        print("Detected 'genres_original' folder, using that as base...")
        gtzan_dir = os.path.join(gtzan_dir, "genres_original")

    ensure_dir(out_dir)
    for g in GENRES:
        in_g = os.path.join(gtzan_dir, g)
        out_g = os.path.join(out_dir, g)
        ensure_dir(out_g)
        if not os.path.isdir(in_g):
            print(f"Skipping missing genre folder: {in_g}")
            continue
        for file in os.listdir(in_g):
            if not (file.endswith('.au') or file.endswith('.wav')):
                continue
            in_file = os.path.join(in_g, file)
            out_file = os.path.join(out_g, file.rsplit('.', 1)[0] + '.npy')
            if os.path.exists(out_file):
                continue
            try:
                import numpy as np
                S = audio_to_mel(in_file)
                np.save(out_file, S)
            except Exception as e:
                print(f"Error processing {in_file}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gtzan-dir', default=GTZAN_DIR)
    parser.add_argument('--out', default=SPECTRO_DIR)
    args = parser.parse_args()
    if not os.path.isdir(args.gtzan_dir):
        try:
            download_gtzan(args.gtzan_dir)
        except Exception as e:
            print('Auto-download failed:', e)
            print('Please download GTZAN manually and extract to', args.gtzan_dir)
            return
    convert_all(args.gtzan_dir, args.out)
    print('✅ Done. Spectrograms saved to', args.out)

if __name__ == '__main__':
    main()
