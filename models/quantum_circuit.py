from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np

def create_quantum_circuit(n_qubits=4, n_layers=1):
    """
    Creates a Parameterized Quantum Circuit (PQC).
    
    Structure:
    1. Data Encoding Layer (Input from CNN)
    2. Repeated Variational Layers (Trainable Weights)
    3. Measurement
    """
    qc = QuantumCircuit(n_qubits)
    
    # 1. Input Parameters (Data Encoding)
    # These will be mapped to the 4 features from CNN
    input_params = ParameterVector("x", n_qubits)
    
    # 2. Trainable Parameters (Variational Weights)
    # These will be learned by the optimizer
    weight_params = ParameterVector("θ", n_qubits * n_layers)
    
    # --- Step 1: Angle Encoding ---
    # Encode classical features into qubit rotations
    for i in range(n_qubits):
        qc.ry(input_params[i], i)
    
    qc.barrier() # Visual separator
    
    # --- Step 2: Variational Layers ---
    for layer in range(n_layers):
        # Trainable rotations
        for i in range(n_qubits):
            qc.ry(weight_params[layer * n_qubits + i], i)
        
        # Entanglement (Circular CNOTs)
        if n_qubits > 1:
            for i in range(n_qubits):
                qc.cx(i, (i + 1) % n_qubits)
        
        qc.barrier()

    return qc, input_params, weight_params

if __name__ == "__main__":
    # Quick test
    qc, inputs, weights = create_quantum_circuit(n_qubits=4, n_layers=1)
    print("Circuit Created Successfully")
    print(f"Inputs: {inputs}")
    print(f"Weights: {weights}")
    print(qc.draw(output='text'))
