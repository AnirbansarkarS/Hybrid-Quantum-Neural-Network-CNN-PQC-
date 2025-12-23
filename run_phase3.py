import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config
from utils.data import load_data
from models.hybrid_model import HybridQNN

def train_hybrid(model, loader, optimizer, epoch, limit_batches=None):
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch} (Hybrid)")
    correct = 0
    total = 0
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(pbar):
        if limit_batches and batch_idx >= limit_batches:
            break
            
        data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        pbar.set_postfix(loss=running_loss/(batch_idx+1), acc=100.*correct/total)

def evaluate_hybrid(model, loader, limit_batches=None):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if limit_batches and batch_idx >= limit_batches:
                break
                
            data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= total
    acc = 100. * correct / total
    print(f'\nHybrid Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({acc:.2f}%)\n')
    return acc

def main():
    print(f"Running Hybrid Training on {Config.DEVICE}")
    print("Warning: Quantum simulation is slow. Training on a subset for demonstration.")
    
    # 1. Load Data
    train_loader, test_loader = load_data()
    
    # 2. Build Hybrid Model
    # Use pretrained path from Phase 1 if exists
    pretrained_path = "models/classical_cnn_weights.pth"
    model = HybridQNN(n_qubits=4, n_layers=1, pretrained_cnn_path=pretrained_path).to(Config.DEVICE)
    
    # 3. Train
    # We use a very small learning rate and fewer batches because quantum gradients are sensitive/slow
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    for epoch in range(1, 2): # Just 1 epoch for demo
        train_hybrid(model, train_loader, optimizer, epoch, limit_batches=10) # Limit to 10 batches for speed
        evaluate_hybrid(model, test_loader, limit_batches=5)
    
    end_time = time.time()
    print(f"Hybrid Training Time for demo: {end_time - start_time:.2f}s")
    
    # Save hybrid model
    torch.save(model.state_dict(), "models/hybrid_qnn_weights.pth")
    print("Saved Hybrid QNN weights to models/hybrid_qnn_weights.pth")

if __name__ == "__main__":
    main()
