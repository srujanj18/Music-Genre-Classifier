import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
GTZAN_DIR = os.path.join(DATA_DIR, 'gtzan')
SPECTRO_DIR = os.path.join(DATA_DIR, 'spectrograms')
SAMPLE_RATE = 22050
DURATION = 30
N_MELS = 128
GENRES = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
