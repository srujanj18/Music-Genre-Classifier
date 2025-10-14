"""
Preprocess GTZAN into mel-spectrogram .npy files.
It attempts to auto-download GTZAN from a public mirror if not present.
"""
import os, argparse, shutil, requests, tarfile
from tqdm import tqdm
from utils import audio_to_mel, ensure_dir
from config import GTZAN_DIR, SPECTRO_DIR, GENRES
import kagglehub

def download_gtzan(dest):
    print('Downloading GTZAN (may take a while)...')
    try:
        path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
        print(f'Dataset downloaded to: {path}')
        # Assuming the dataset is downloaded to a directory, copy it to dest
        if os.path.isdir(path):
            shutil.copytree(path, dest, dirs_exist_ok=True)
        else:
            raise RuntimeError('Downloaded path is not a directory.')
    except Exception as e:
        print('Auto-download failed:', e)
        print('Please download GTZAN manually and extract to', dest)
        raise

def convert_all(gtzan_dir, out_dir):
    # Detect if 'genres_original' exists
    if os.path.isdir(os.path.join(gtzan_dir, "Data/genres_original")):
        print("Detected 'Data/genres_original' folder, using that as base...")
        gtzan_dir = os.path.join(gtzan_dir, "Data/genres_original")

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
    # Check if data is already available
    data_dir = os.path.join(os.path.dirname(args.gtzan_dir), 'Data')
    if os.path.isdir(data_dir):
        print(f'Data directory found at {data_dir}, skipping download.')
        args.gtzan_dir = data_dir
    elif not os.path.isdir(args.gtzan_dir):
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
