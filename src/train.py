import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm

from model_cnn import TinyMNISTCNN

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

def train_one_epoch(model, loader, optimzr, device):
    model.train()
    total = 0; correct = 0; loss_sum = 0.0
    for x,y in tqdm(loader, desc="train"):
        x,y = x.to(device), y.to(device)
        optimzr.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimzr.step()

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

def evaluate(model, loader, device):
    model.eval()
    total = 0; correct = 0; loss_sum = 0.0
    with torch.no_grad():
        for x,y in tqdm(loader, desc="eval"):
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return loss_sum/total, correct/total

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1], shape [1,28,28]
    ])

    train_ds = datasets.MNIST(root=str(RESULTS), train=True, transform=transform, download=True)
    test_ds  = datasets.MNIST(root=str(RESULTS), train=False, transform=transform, download=True)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = TinyMNISTCNN().to(device)
    optimzr = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0
    epochs = 3
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimzr, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        print(f"[E{ep}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | test loss={te_loss:.4f} acc={te_acc:.4f}")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), RESULTS / "mnist_cnn.pt")
            print(f"[SAVE] results/mnist_cnn.pt (acc={best_acc:.4f})")

    print("[DONE] Training complete.")
