from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np

def create_qnn(num_qubits=4):
    qc = QuantumCircuit(num_qubits)

    params = [Parameter(f"θ{i}") for i in range(num_qubits)]

    # Angle Encoding
    for i in range(num_qubits):
        qc.ry(params[i], i)

    # Entanglement
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    return qc, params
