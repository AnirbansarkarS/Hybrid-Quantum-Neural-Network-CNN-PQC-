# Phase 4: Theory & Research Report - Evaluation

## 📊 Theory: The Evaluation Framework

In Phase 4, we analyze whether the **Quantum Layer** provides any measurable difference compared to the standard Classical layer.

### 1. Accuracy vs. Expressivity
- **The Question:** Does a Quantum Circuit with only 12 parameters (8 weights + 4 inputs) perform as well as a Classical layer with hundreds of parameters?
- **Quantum Advantage:** Quantum circuits can cover a larger "Hilbert Space" with fewer parameters. This is known as high **expressivity**.
- **Observation:** In small datasets like MNIST, Hybrid models often converge faster or achieve similar accuracy to deeper classical models despite having fewer weights.

### 2. Time Complexity (Simulation vs. Real HW)
- **Classical:** Runs in milliseconds on CPU/GPU.
- **Quantum Simulation:** Every gradient step requires $2 \times N_{params}$ circuit runs. This is the main bottleneck today.
- **On Real Hardware:** We gain advantage in parallelism once the qubit count scales, but overhead remains high in the NISQ era.

### 3. Noise Simulation
- Real Quantum Computers are "noisy". Environmental heat or radiation can flip qubits.
- In research, we often add **Noise Models** (e.g., Thermal Relaxation, Depolarizing Noise) to see if our Hybrid model still works in "real-world" quantum conditions.
- **Robustness:** Hybrid models are often more robust to noise than pure quantum models because the Classical CNN acts as a noise-filtering pre-processor.

---

## ✅ Progress: Phase 4 (Evaluation)

### 1. Metrics Implementation
- Created `utils/metrics.py` to generate professional comparison plots.
- Added research tracking for:
    - Training time per epoch.
    - Parameter count (Classical vs Quantum).
    - Final Accuracy.

### 2. Comparative Analysis
- We observe that the **Hybrid Model** manages to classify digits using only the expectation values of 4 qubits. 
- This proves that **Quantum Feature Selection** is viable for image data.

## 🏁 Project Conclusion
We have successfully built a full pipeline:
1. **Phase 1:** Image Compression (CNN).
2. **Phase 2:** Quantum Mapping (Angle Encoding).
3. **Phase 3:** Hybrid Learning (Parameter Shift Rule).
4. **Phase 4:** Research Verification (Metrics & Analysis).

**This architecture is the foundation for Quantum Machine Learning (QML) research.**
