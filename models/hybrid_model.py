import torch
import torch.nn as nn
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from .classical_cnn import ClassicalCNN
from .quantum_circuit import create_quantum_circuit

import os

class HybridQNN(nn.Module):
    def __init__(self, n_qubits=4, n_layers=1, pretrained_cnn_path=None):
        super().__init__()
        
        # 1. Classical CNN Feature Extractor
        self.cnn = ClassicalCNN(out_features=n_qubits)
        if pretrained_cnn_path and os.path.exists(pretrained_cnn_path):
            try:
                self.cnn.load_state_dict(torch.load(pretrained_cnn_path, map_location='cpu'))
                print(f"Loaded pretrained CNN weights from {pretrained_cnn_path}")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")
        
        # 2. Quantum Layer
        qc, input_params, weight_params = create_quantum_circuit(n_qubits=n_qubits, n_layers=n_layers)
        
        # Define observables: Z measurements on each qubit separately
        # This allows the QNN to output (Batch, n_qubits) features instead of (Batch, 1)
        observables = []
        for i in range(n_qubits):
            # Create Pauli string like "IIIZ", "IIZI", etc. (Qiskit order is q_n-1 ... q_0)
            # We want Z on the i-th qubit.
            pauli_str = ["I"] * n_qubits
            pauli_str[n_qubits - 1 - i] = "Z"
            observables.append(SparsePauliOp("".join(pauli_str)))
        
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=input_params,
            weight_params=weight_params,
            observables=observables
        )
        
        # Wrap QNN for PyTorch
        self.quantum_layer = TorchConnector(qnn)
        
        # 3. Post-processing
        # Now the QNN outputs 4 features, so fc layer connects 4 -> 10
        self.fc = nn.Linear(n_qubits, 10) 

    def forward(self, x):
        # 1. Classical Features
        x = self.cnn(x) # (Batch, 4)
        
        # 2. Quantum Processing
        # TorchConnector expects (Batch, Input_Dim)
        x = self.quantum_layer(x) # (Batch, 4)
        
        # 3. Output
        x = self.fc(x) # (Batch, 10)
        return x
