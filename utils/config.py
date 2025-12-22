import torch

class Config:
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data
    BATCH_SIZE = 64
    NUM_WORKERS = 0 # Set to 0 for Windows compatibility issues sometimes
    
    # Model
    N_FEATURES = 4    # Size of the feature vector (output of CNN)
    N_CLASSES = 10    # MNIST classes
    
    # Training
    LR = 0.001
    EPOCHS = 5
    SEED = 42
