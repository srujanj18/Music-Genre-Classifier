from huggingface_hub import HfApi, upload_file
import torch
from src.model import SimpleCNN
from src.config import GENRES

# Load the model
model = SimpleCNN(num_classes=len(GENRES))
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

# Save the model in a format suitable for Hugging Face
torch.save(model.state_dict(), 'pytorch_model.bin')

# Create model card
model_card = """
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
"""

with open('README.md', 'w') as f:
    f.write(model_card)

# Upload to Hugging Face
api = HfApi()
api.create_repo("srujan1810/Music-Genre-Classifier", exist_ok=True)
upload_file(path_or_fileobj='pytorch_model.bin', path_in_repo='pytorch_model.bin', repo_id='srujan1810/Music-Genre-Classifier')
upload_file(path_or_fileobj='README.md', path_in_repo='README.md', repo_id='srujan1810/Music-Genre-Classifier')
