import torch
from models.hybrid_model import HybridQNN

def test():
    print("Initializing HybridQNN...")
    try:
        model = HybridQNN(n_qubits=4, n_layers=1)
        print("Model initialized.")
        
        x = torch.randn(2, 1, 28, 28)
        print("Forward pass...")
        out = model(x)
        print(f"Output shape: {out.shape}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
