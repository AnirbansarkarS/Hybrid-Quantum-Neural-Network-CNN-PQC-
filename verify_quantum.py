import matplotlib.pyplot as plt
import numpy as np
from qiskit.visualization import circuit_drawer
from models.quantum_circuit import create_quantum_circuit
import os

def main():
    print("--- Phase 2: Quantum Circuit Verification ---")
    
    n_qubits = 4
    n_layers = 2 # Let's try 2 layers for more depth
    
    qc, inputs, weights = create_quantum_circuit(n_qubits=n_qubits, n_layers=n_layers)
    
    print(f"Number of Qubits: {n_qubits}")
    print(f"Number of Input Parameters: {len(inputs)}")
    print(f"Number of Trainable Weights: {len(weights)}")
    
    # Draw the circuit
    print("\nCircuit Diagram:")
    print(qc.draw(output='text'))
    
    # Save visualization to a file
    try:
        # Check if we can use matplotlib for nicer drawing
        fig = qc.draw(output='mpl')
        plt.title(f"Hybrid QNN Circuit ({n_qubits} Qubits, {n_layers} Layers)")
        save_path = "models/quantum_circuit_viz.png"
        fig.savefig(save_path)
        print(f"\n[SUCCESS] Circuit visualization saved to: {save_path}")
    except Exception as e:
        print(f"\n[NOTE] Matplotlib drawing failed (likely missing dependencies), text drawing used above. Error: {e}")

    print("\nTheory Check:")
    print("1. Angle Encoding: Features mapped to Ry rotations.")
    print("2. Trainable Weights: Parametric Ry gates follow encoding.")
    print("3. Entanglement: CNOT gates link qubits for non-linear correlation.")
    
if __name__ == "__main__":
    main()
