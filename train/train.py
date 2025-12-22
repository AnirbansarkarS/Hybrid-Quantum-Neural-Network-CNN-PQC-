import torch
import torch.optim as optim
import torch.nn.functional as F

def train(model, loader, device):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
