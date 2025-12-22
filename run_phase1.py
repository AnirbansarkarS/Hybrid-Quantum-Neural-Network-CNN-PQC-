import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config
from utils.data import load_data
from models.classical_cnn import ClassicalCNN, ClassicalClassifier

def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    correct = 0
    total = 0
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
        
        optimizer.zero_grad()
        output, _ = model(data) # Get class logits
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        pbar.set_postfix(loss=running_loss/(batch_idx+1), acc=100.*correct/total)

def evaluate(model, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
            output, _ = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)
    acc = 100. * correct / len(loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({acc:.2f}%)\n')
    return acc

def main():
    print(f"Running on {Config.DEVICE}")
    
    # 1. Load Data
    print("Loading Data...")
    train_loader, test_loader = load_data()
    
    # 2. Build Model
    # We want the CNN to output 4 features (to be compatible with 4 qubits later)
    # But for Phase 1 baseline, we train it with a classical linear layer on top.
    print(f"Building Classical CNN (Output Features: {Config.N_FEATURES})...")
    cnn = ClassicalCNN(out_features=Config.N_FEATURES)
    
    model = ClassicalClassifier(cnn, n_classes=Config.N_CLASSES).to(Config.DEVICE)
    
    # 3. Train
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    
    print("Starting Training...")
    for epoch in range(1, Config.EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, epoch)
        evaluate(model, test_loader)
        
    # 4. Save
    torch.save(model.cnn.state_dict(), "models/classical_cnn_weights.pth")
    print("Saved CNN weights to models/classical_cnn_weights.pth")

if __name__ == "__main__":
    main()
