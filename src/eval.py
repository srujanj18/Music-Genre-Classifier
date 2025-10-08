import os, torch, numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from dataset import MelSpecDataset
from model import SimpleCNN
from config import GENRES

def evaluate(data_dir, model_path='best_model.pth'):
    ds = MelSpecDataset(data_dir, GENRES)
    loader = __import__('torch').utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=len(GENRES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            ys.extend(yb.numpy().tolist())
    print(classification_report(ys, preds, target_names=GENRES))
    print('Confusion matrix:')
    print(confusion_matrix(ys, preds))

if __name__ == '__main__':
    evaluate(os.path.join('data','spectrograms'))
