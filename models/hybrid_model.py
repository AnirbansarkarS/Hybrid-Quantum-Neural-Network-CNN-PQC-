import torch
import torch.nn as nn
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
from .classical_cnn import ClassicalCNN
from .quantum_circuit import create_quantum_circuit

import os

class HybridQNN(nn.Module):
    def __init__(self, n_qubits=4, n_layers=1, pretrained_cnn_path=None):
        super().__init__()
        
        # 1. Classical CNN Feature Extractor
        self.cnn = ClassicalCNN(out_features=n_qubits)
        if pretrained_cnn_path and os.path.exists(pretrained_cnn_path):
            self.cnn.load_state_dict(torch.load(pretrained_cnn_path, map_location='cpu'))
            print(f"Loaded pretrained CNN weights from {pretrained_cnn_path}")
        
        # 2. Quantum Layer
        qc, input_params, weight_params = create_quantum_circuit(n_qubits=n_qubits, n_layers=n_layers)
        
        # We use EstimatorQNN to get expectation values
        # The observable is typically Z measurements on all qubits
        # For simplicity, we can use the default which usually returns one value per qubit or similar
        # Here we configure it to return expectation values for each qubit separately
        
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=input_params,
            weight_params=weight_params
        )
        
        # Wrap QNN for PyTorch
        # initial_weights can be randomized or set
        self.quantum_layer = TorchConnector(qnn)
        
        # 3. Post-processing (Optional, but often helps to map QNN output to classes)
        # Expected output from QNN is (Batch, 1) by default if it's one expectation value
        # or (Batch, N) if multiple observables. Qiskit EstimatorQNN with 1 observable = 1 output.
        # Let's add a linear layer to map to 10 classes
        # Note: Depending on how many observables are defined, the output size varies.
        # By default EstimatorQNN returns a single value if no observables are passed.
        # Let's check n_classes mapping.
        self.fc = nn.Linear(1, 10) 

    def forward(self, x):
        # 1. Classical Features
        x = self.cnn(x) # (Batch, 4)
        
        # 2. Quantum Processing
        # TorchConnector expects (Batch, Input_Dim)
        x = self.quantum_layer(x) # (Batch, 1)
        
        # 3. Output
        x = self.fc(x) # (Batch, 10)
        return x
