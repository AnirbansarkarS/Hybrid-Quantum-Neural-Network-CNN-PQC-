import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config
from utils.data import load_data
from utils.metrics import plot_training_results, print_research_metrics
from models.classical_cnn import ClassicalCNN, ClassicalClassifier
from models.hybrid_model import HybridQNN

def train_model(model, loader, optimizer, epoch, desc="Training"):
    model.train()
    pbar = tqdm(loader, desc=f"{desc} Epoch {epoch}")
    correct = 0
    total = 0
    running_loss = 0.0
    
    for data, target in pbar:
        data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
        
        optimizer.zero_grad()
        output = model(data)
        if isinstance(output, tuple):
            output = output[0] # ClassicalClassifier returns (logits, features)
            
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        pbar.set_postfix(loss=running_loss/len(loader), acc=100.*correct/total)
    
    return running_loss/len(loader), 100.*correct/total

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return 100. * correct / total

def run_experiment():
    print(f"Starting comparison on {Config.DEVICE}")
    Config.BATCH_SIZE = 32 # Smaller batch for more updates per epoch
    train_loader, test_loader = load_data()
    
    # Use a small subset to ensure visible progress in reasonable time
    # Quantum simulation is the bottleneck
    train_limit = 1000 
    test_limit = 200
    
    # Filter datasets (Simple hack for demonstration)
    train_data = []
    for i, (d, t) in enumerate(train_loader.dataset):
        if i >= train_limit: break
        train_data.append((d, t))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    test_data = []
    for i, (d, t) in enumerate(test_loader.dataset):
        if i >= test_limit: break
        test_data.append((d, t))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=Config.BATCH_SIZE)

    epochs = 3
    
    # --- 1. Classical Training ---
    print("\n--- Training Classical CNN Baseline ---")
    classical_cnn = ClassicalCNN(out_features=4)
    classical_model = ClassicalClassifier(classical_cnn, n_classes=10).to(Config.DEVICE)
    optimizer_c = optim.Adam(classical_model.parameters(), lr=0.002)
    
    c_history = {'loss': [], 'acc': []}
    start_c = time.time()
    for epoch in range(1, epochs + 1):
        loss, acc = train_model(classical_model, train_loader, optimizer_c, epoch, desc="Classical")
        c_history['loss'].append(loss)
        c_history['acc'].append(acc)
    time_c = time.time() - start_c
    acc_c = evaluate_model(classical_model, test_loader)

    # --- 2. Hybrid QNN Training ---
    print("\n--- Training Hybrid QNN ---")
    # We use the same CNN architecture part but with the Quantum Layer
    hybrid_model = HybridQNN(n_qubits=4, n_layers=1).to(Config.DEVICE)
    optimizer_h = optim.Adam(hybrid_model.parameters(), lr=0.002)
    
    h_history = {'loss': [], 'acc': []}
    start_h = time.time()
    for epoch in range(1, epochs + 1):
        loss, acc = train_model(hybrid_model, train_loader, optimizer_h, epoch, desc="Hybrid")
        h_history['loss'].append(loss)
        h_history['acc'].append(acc)
    time_h = time.time() - start_h
    acc_h = evaluate_model(hybrid_model, test_loader)

    # --- 3. Comparison ---
    plot_training_results(c_history, h_history)
    
    classical_stats = {
        "Final Test Acc": f"{acc_c:.2f}%",
        "Training Time": f"{time_c:.2f}s",
        "Parameters": sum(p.numel() for p in classical_model.parameters())
    }
    
    hybrid_stats = {
        "Final Test Acc": f"{acc_h:.2f}%",
        "Training Time": f"{time_h:.2f}s",
        "Parameters": sum(p.numel() for p in hybrid_model.parameters())
    }
    
    print_research_metrics(classical_stats, hybrid_stats)

if __name__ == "__main__":
    run_experiment()
