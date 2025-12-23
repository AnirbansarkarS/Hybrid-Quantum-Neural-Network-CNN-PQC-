# Phase 2: Theory & Progress Report - Quantum Circuit Design

## ⚛️ Theory: The "Quantum Brain"

In Phase 1, we built a CNN that compresses an image into 4 numbers. Now, we feed those numbers into a **Quantum Intelligence** layer.

### 1. Data Encoding (Angle Encoding)
We can't just "plug in" a number into a qubit. We use **Angle Encoding**. 
- If our CNN outputs a value of `1.57` ($\pi/2$), we rotate the qubit by 90 degrees.
- This maps classical data into the "Hilbert Space" (the mathematical home of quantum states).

### 2. Parameterized Quantum Circuit (PQC)
Unlike a fixed gate (like an X or H gate), a PQC has "knobs" (parameters $\theta$) that we can turn.
- The **Weights ($\theta$)** are what the Adam optimizer will adjust during training.
- By rotating qubits differently, the circuit learns to separate different classes of images (e.g., distinguishing a '0' from a '1').

### 3. Entanglement (The "Connectivity")
In classical networks, we have "Fully Connected" layers. In quantum, we use **Entanglement** (CNOT gates).
- This makes qubits depend on each other. A change in Qubit 0 affects the state of Qubit 1.
- This allows the model to capture complex relationships between the 4 features extracted by the CNN.

---

## ✅ Progress: Phase 2

### 1. Refined `models/quantum_circuit.py`
- Implemented `create_quantum_circuit` with:
    - `input_params` (ParameterVector "x"): For CNN features.
    - `weight_params` (ParameterVector "θ"): For trainable weights.
    - Layers of Ry rotations and circular CNOT entanglement.

### 2. Visualization & Verification
- Created `verify_quantum.py`.
- This script generates the circuit, counts parameters, and saves a visual diagram.
- **Why?** It's critical to see the circuit architecture to debug the data flow before training.

## 🚀 Next Steps (Phase 3)
Now comes the "Magic": **Hybrid Training**.
1. Wrap this circuit in a `qiskit_machine_learning.connectors.TorchConnector`.
2. Stack it on top of our Phase 1 CNN.
3. Train them together so the CNN learns to extract features that the Quantum Circuit is "good" at classifying.
