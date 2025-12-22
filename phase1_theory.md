# Phase 1: Theory & Progress Report

## 🧠 Theory Overview

### why reduce dimensions?
Quantum computers (Noisy Intermediate-Scale Quantum - NISQ) currently have a limited number of stable qubits. 
- **The Problem:** A standard MNIST image is 28x28 = 784 pixels. We cannot easily map 784 pixels to 784 qubits (too noisy, too expensive).
- **The Solution:** We use a **classical Neural Network (CNN)** as a compressor. It learns to "see" the image and summarize it into just **4 numbers** (features).
- **The Result:** These 4 numbers can be easily encoded into **4 qubits** using angle encoding.

### The Baseline Approach
Before adding the quantum magic, we **MUST** prove that these 4 features contain enough information to distinguish the digits.
If a classical network can't classify digits using only 4 features, a quantum network won't be able to either. Garbage in, garbage out.

**Architecture for Phase 1:**
1.  **Input:** Image (28x28)
2.  **CNN (Feature Extractor):** Convolutions + MaxPool -> ... -> **4 values**.
3.  **Classical Head (The "Judge"):** A simple Linear layer taking 4 values -> 10 classes.

We train this entire stack. Once trained, we throw away the "Classical Head" and replace it with the **Quantum Circuit** (Phase 2/3). The CNN weights we saved will be reused.

---

## ✅ Progress: Phase 0 & 1

### 1. Environment Setup (Phase 0)
- Defined dependencies in `requirements.txt`: `torch`, `torchvision`, `qiskit` etc.
- Created `utils/config.py` to centralize settings (Batch size 64, LR 0.001, 4 Features).

### 2. Dataset Implementation
- Created `utils/data.py` to automatically download and load **MNIST**.
- Applied normalization `(0.1307, 0.3081)` which is standard for MNIST.

### 3. Model Implementation
- **File:** `models/classical_cnn.py`
- Implemented `ClassicalCNN`:
    - Conv1 -> Relu -> MaxPool
    - Conv2 -> Relu -> MaxPool
    - FC1 -> Relu -> **FC2 (Output 4)**
- Implemented `ClassicalClassifier` wrapper:
    - Wraps the CNN.
    - Adds a final Linear Layer (4 -> 10) so we can compute CrossEntropyLoss and train it.

### 4. Execution Script
- **File:** `run_phase1.py`
- Loads data.
- Initializes the model stack.
- Trains for 5 epochs using Adam optimizer.
- Evaluates accuracy on Test set.
- **Saves the trained CNN weights** to `models/classical_cnn_weights.pth`.

## 🚀 Next Steps (Phase 2)
Now that we have confirmed our CNN can compress images into 4 meaningful numbers, we will:
1.  **Design the Quantum Circuit:** It will take those 4 numbers as rotation angles for 4 qubits.
2.  **Entangle them:** CNOT gates to make the qubits talk to each other.
3.  **Measure:** Get the "Quantum Plan" of the class.
