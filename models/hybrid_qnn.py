import torch
import torch.nn as nn
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import Estimator

from .quantum_circuit import create_qnn

class HybridQNN(nn.Module):
    def __init__(self):
        super().__init__()

        qc, params = create_qnn()

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=params,
            weight_params=params,
            estimator=Estimator()
        )

        self.q_layer = TorchConnector(qnn)
        self.fc = nn.Linear(1, 10)

    def forward(self, x):
        x = self.q_layer(x)
        x = self.fc(x)
        return x
