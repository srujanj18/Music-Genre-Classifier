# Music Genre Classifier

A deep learning-based music genre classification system built with PyTorch and Flask. This project uses Convolutional Neural Networks (CNNs) to classify audio files into 10 music genres based on mel spectrograms extracted from the GTZAN dataset.

## Features

- **Deep Learning Model**: Custom CNN architecture for genre classification
- **Web Interface**: Flask-based web application for easy audio file upload and prediction
- **Audio Processing**: Mel spectrogram extraction using Librosa
- **Dataset Support**: GTZAN dataset with 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
- **GPU Support**: Automatic GPU detection and utilization
- **Evaluation Metrics**: Accuracy, confusion matrix, and classification reports

## Project Structure

```
music-genre-classifier/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration settings and constants
│   ├── dataset.py         # PyTorch dataset class for GTZAN
│   ├── eval.py            # Model evaluation utilities
│   ├── inference.py       # Model loading and prediction functions
│   ├── model.py           # CNN model architecture
│   ├── preprocess_save.py # Audio preprocessing and spectrogram generation
│   ├── train.py           # Training script
│   └── utils.py           # Utility functions for audio processing
├── app/
│   ├── app.py             # Flask web application
│   ├── static/            # Static files (CSS, JS, images)
│   └── templates/
│       └── index.html     # Web interface template
├── Data/                  # Directory for dataset and processed data
├── .gitignore
├── README.md
├── requirements.txt       # Python dependencies
├── run_local.sh           # Helper script for local setup
└── sample_output1.png     # Sample output images
```

## Technologies Used

- **Python 3.8+**
- **PyTorch**: Deep learning framework
- **TorchAudio**: Audio processing with PyTorch
- **Librosa**: Audio and music analysis
- **Flask**: Web framework for the application
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **Matplotlib**: Plotting and visualization
- **SoundFile**: Audio file I/O

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/srujanj18/Music-Genre-Classifier.git
   cd music-genre-classifier
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or use the provided script:
   ```bash
   # On Unix systems:
   ./run_local.sh
   ```

## Dataset

This project uses the GTZAN dataset, which contains 1000 audio tracks of 30 seconds each, categorized into 10 genres:

- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

The dataset will be automatically downloaded during preprocessing if not present.

## Training

### Data Preprocessing

First, preprocess the audio files to generate mel spectrograms:

```bash
python -m src.preprocess_save --out data/spectrograms/ --gtzan-dir data/gtzan/
```

This will:
- Download GTZAN dataset if not present
- Extract mel spectrograms from audio files
- Save processed data for training

### Train the Model

Train the CNN model:

```bash
python -m src.train --data data/spectrograms/ --epochs 30
```

Training parameters can be modified in `src/config.py` or via command-line arguments.

The trained model will be saved as `best_model.pth` in the project root.

## Inference

### Command Line

Use the inference script for single file prediction:

```python
from src.inference import load_model, predict_file

model = load_model('best_model.pth')
result = predict_file(model, 'path/to/audio/file.wav')
print(f"Predicted genre: {result['genre']}")
print(f"Confidence scores: {result['probs']}")
```

### Web Application

Run the Flask web app for interactive predictions:

```bash
python app/app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`

The web interface allows you to:
- Upload audio files
- Get instant genre predictions
- View confidence scores for all genres

## Model Architecture

The model uses a simple CNN architecture:

- 3 Convolutional layers with batch normalization
- Max pooling after each conv layer
- Dropout for regularization
- Adaptive average pooling
- 2 Fully connected layers
- Output: 10 classes (genres)

Input: Mel spectrogram (1 channel, variable time dimension, 128 mel bins)

## Evaluation

Evaluate the trained model:

```bash
python -m src.eval --model best_model.pth --data data/spectrograms/
```

This will generate:
- Accuracy metrics
- Confusion matrix
- Classification report
- Visual plots (saved as images)

## Configuration

Modify settings in `src/config.py`:

- `SAMPLE_RATE`: Audio sample rate (default: 22050)
- `DURATION`: Audio clip duration in seconds (default: 30)
- `N_MELS`: Number of mel bins (default: 128)
- `GENRES`: List of genre classes
- Data directory paths

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- GTZAN dataset by George Tzanetakis
- PyTorch community
- Librosa developers

## Troubleshooting

- **Model not found error**: Ensure you've trained the model first using `src/train.py`
- **Dataset download issues**: Download GTZAN manually and place in `data/gtzan/`
- **Audio file errors**: Ensure audio files are in supported formats (WAV, MP3, etc.)
- **Memory issues**: Reduce batch size in training or use CPU if GPU memory is insufficient
