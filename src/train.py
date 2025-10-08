import os, argparse, numpy as np, torch, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score
from .dataset import MelSpecDataset
from .model import SimpleCNN
from .config import GENRES

def train(args):
    dataset = MelSpecDataset(args.data, GENRES)
    n = len(dataset)
    train_n = int(0.8*n)
    val_n = n - train_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=len(GENRES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = torch.nn.CrossEntropyLoss()

    best_val = float('inf')
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = np.mean(losses) if losses else 0.0

        model.eval()
        ys, preds, val_losses = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
                preds.extend(pred.tolist())
                ys.extend(yb.cpu().numpy().tolist())
        val_loss = np.mean(val_losses) if val_losses else 0.0
        val_acc = accuracy_score(ys, preds) if ys else 0.0
        val_f1 = f1_score(ys, preds, average='macro') if ys else 0.0
        scheduler.step(val_loss)
        print(f'Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}')
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.join('data','spectrograms'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
