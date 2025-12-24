import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.quantum_circuit import create_quantum_circuit

def generate_circuit_image():
    print("Generating Quantum Circuit Diagram...")
    qc, _, _ = create_quantum_circuit(n_qubits=4, n_layers=1)
    
    # Use matplotlib output for a nice visualization
    # Note: Requires pylatexenc for some features, but basic mpl should work
    fig = qc.draw(output='mpl', style='iqp')
    plt.title("Hybrid QNN: 4-Qubit Variational Circuit Architecture", fontsize=14, pad=20)
    
    save_path = "circuit_diagram.png"
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved circuit diagram to {save_path}")

if __name__ == "__main__":
    generate_circuit_image()
