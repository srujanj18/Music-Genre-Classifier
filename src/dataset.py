import os, numpy as np, torch
from torch.utils.data import Dataset

class MelSpecDataset(Dataset):
    def __init__(self, root_dir, genres, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.genre2idx = {g:i for i,g in enumerate(genres)}
        for g in genres:
            folder = os.path.join(root_dir, g)
            if not os.path.isdir(folder):
                continue
            for f in os.listdir(folder):
                if f.endswith('.npy'):
                    self.samples.append(os.path.join(folder,f))
                    self.labels.append(self.genre2idx[g])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x = np.load(self.samples[idx])
        x = np.expand_dims(x, axis=0)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        if self.transform:
            x = self.transform(x)
        return x, y
