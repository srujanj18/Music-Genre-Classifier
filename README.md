
---
language: en
tags:
- music-genre-classification
- pytorch
- cnn
datasets:
- gtzan
license: mit
---

# Music Genre Classifier

This model classifies music into 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.

## Model Details

- **Model Type:** Convolutional Neural Network (CNN)
- **Input:** Mel spectrograms (128 mel bins, 30 seconds)
- **Output:** Genre classification probabilities

## Usage

```python
import torch
from model import SimpleCNN

model = SimpleCNN(num_classes=10)
model.load_state_dict(torch.load('pytorch_model.bin'))
model.eval()
# Preprocess audio to mel spectrogram and pass to model
```

## Training Data

Trained on GTZAN dataset with 1000 audio clips (100 per genre).

## Performance

Achieved high accuracy on validation set.
