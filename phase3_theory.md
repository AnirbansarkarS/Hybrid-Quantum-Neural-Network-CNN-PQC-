# Phase 3: Theory & Progress Report - Hybrid Training

## 🧬 Theory: How Classical and Quantum Converge

In Phase 3, we perform **End-to-End Hybrid Training**. This means the error calculated at the output (the classification guess) flows backward through the Quantum Circuit and then into the CNN.

### 1. The Bridge: `TorchConnector`
PyTorch works by building a computational graph of tensors. A Quantum Circuit, however, is a series of hardware instructions (or simulations).
- `TorchConnector` allows the Qiskit Quantum Circuit to appear as a standard `nn.Module`.
- It handles the conversion between PyTorch Tensors and Qiskit Parameter values automatically.

### 2. Quantum Backpropagation (The Hard Part)
Standard Neural Networks use **Automatic Differentiation (Autograd)**. You can't "differentiate" a physical quantum circuit in the same way.
- Instead, we use the **Parameter Shift Rule**.
- To find the gradient of a parameter $\theta$, we run the circuit twice: once at $\theta + \pi/2$ and once at $\theta - \pi/2$.
- The difference between these results tells us the "slope" (gradient) of the error.
- **This is why Quantum Training is slow:** Every update requires multiple circuit executions!

### 3. Gradient Flow
1. **Forward Pass:** Image -> CNN -> 4 Features -> Qubits -> Measure Expectation -> Linear Layer -> Guess.
2. **Loss:** Calculate Cross Entropy.
3. **Backward Pass:**
   - Linear layer gradients (Classical).
   - Quantum layer gradients (Parameter Shift Rule).
   - CNN gradients (Classical, but using the error "passed through" the quantum layer).

---

## ✅ Progress: Phase 3

### 1. `models/hybrid_model.py`
- Created `HybridQNN` class.
- Integrated `ClassicalCNN` as the feature extractor.
- Used `EstimatorQNN` for calculating expectation values (physical observables).
- Wrapped with `TorchConnector`.

### 2. `run_phase3.py`
- Implemented a hybrid training loop.
- Added **Batch Limiting**: Because quantum simulation is computationally expensive, we limit training to a subset of data to verify the logic without waiting hours.
- Verified that gradients flow through the entire stack.

## 🚀 Next Steps (Phase 4)
We will now evaluate our creation:
1. Compare accuracy of "4-feature Classical" vs "4-feature Quantum".
2. Analyze the training time overhead.
3. Discuss the research implications of Quantum Advantage in feature processing.
