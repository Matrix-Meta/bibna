# bibna (Brain-Inspired BitNet Architecture)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![SYCL](https://img.shields.io/badge/Framework-SYCL%20(DPC%2B%2B)-orange.svg)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/data-parallel-c-plus-plus.html)

> **Note:** This project is highly experimental and is currently in the active prototyping phase.

[繁體中文 (Traditional Chinese)](README_zh-TW.md)

---

## 📖 Overview

**bibna** is a cutting-edge research framework implemented in C++ using SYCL. It aims to bridge the gap between the extreme efficiency of **BitNet b1.58** (1.58-bit Large Language Models) and the biological plausibility of **Spiking Neural Networks (SNNs)** and **Hebbian Learning**.

Unlike standard deep learning frameworks that treat neural networks as static graphs of floating-point matrix multiplications, `bibna` builds a computational stack from the ground up to support:
1.  **Ternary Weights (`{-1, 0, +1}`)**: Minimizing memory bandwidth and compute energy.
2.  **Event-Driven Computation**: Using spiking neurons with internal states (membrane potential).
3.  **Dynamic Memory**: Implementing "Fast Weights" that update during inference to provide a working memory context.

## 🧠 Core Philosophy

The architecture is driven by the hypothesis that general intelligence requires more than just static pattern matching. It needs:
*   **Local Reasoning**: Handled by `MicroCircuit`s—small, recurrent clusters of neurons that process information temporally.
*   **Global Integration**: Managed by `CorticalHub`s that integrate local signals and modulate them via attention mechanisms.
*   **Self-Correction**: An **ACC (Anterior Cingulate Cortex)** module to detect conflicts and uncertainty, triggering "reflection" loops before outputting a response.

## 🏗️ Architecture & Components

The project follows a strict bottom-up hierarchical design:

### 1. Low-Level Primitives (`src/kernels.hpp`, `src/bitlinear.hpp`)
*   **BitLinear**: The fundamental dense layer. It uses on-the-fly quantization to project FP32 activations against ternary weights.
*   **Straight-Through Estimator (STE)**: Allows gradients to flow through the non-differentiable quantization steps during training.

### 2. Neuronal Model (`src/spiking_unit.hpp`)
*   **SpikingUnit**: Replaces ReLU/GELU. Neurons accumulate "voltage" over time and fire discrete spikes when a threshold is crossed.
*   **Surrogate Gradients**: Enables backpropagation through time (BPTT) for spiking neurons.

### 3. Micro-Architecture (`src/micro_circuit.hpp`)
*   **The MicroCircuit**: A recurrent block containing $N$ spiking neurons.
*   It features internal recurrent connections (`W_rec`) and residual pathways, acting as a "cortical column" capable of maintaining local temporal state.

### 4. Macro-Architecture (`src/cortical_hub.hpp`, `src/brain_block.hpp`)
*   **CorticalHub**: The central router. It implements **Fast Weights ($M_t$)**, a short-term memory matrix updated via Hebbian rules ( $M_{t+1} \leftarrow \lambda M_t + \eta (k \otimes v)$ ).
*   **BrainBlock**: The high-level layer that runs multiple `MicroCircuit`s in parallel and fuses their outputs via the Hub.

## 🗺️ Roadmap

The project development is divided into 6 phases (see `docs/brain_inspired_bitnet_design_v1.md` for details):

- [x] **Phase 0: Environment Setup**
    - CMake build system and SYCL (Intel DPC++) environment configuration.
- [x] **Phase 1: Basic Operators**
    - Implementation of `BitLinear` and `SpikingUnit` with gradient verification.
- [x] **Phase 2: Micro-Structures**
    - `MicroCircuit` with recurrence and `CorticalHub` with basic Fast Weights.
- [x] **Phase 3: BrainBlock Integration**
    - Orchestrating parallel circuits and pooling outputs.
- [ ] **Phase 4: Advanced Fast Weights & Decoding**
    - Full integration of attention-over-memory and anti-collapse decoding strategies.
- [ ] **Phase 5: Meta-Controller (ACC)**
    - Implementing the conflict monitoring system for dynamic inference depth.
- [ ] **Phase 6: Continuous Learning**
    - Episodic buffers and adapter-based plasticity.

## 🚀 Getting Started

### Prerequisites
*   **C++ Compiler**: C++17 standard compliant.
*   **SYCL SDK**: Intel oneAPI Base Toolkit (DPC++) is recommended for broader hardware support (CPU, GPU, FPGA).
*   **CMake**: Version 3.10+.

### Build Instructions

```bash
# 1. Clone the repository
git clone https://github.com/your-username/bibna.git
cd bibna

# 2. Create build directory
mkdir build && cd build

# 3. Configure (Ensure Intel oneAPI variables are sourced)
cmake ..

# 4. Build the project
make -j$(nproc)
```

### Running Tests

The compiled binary `bibna_main` runs a suite of validation tests ranging from unit tests to a small-scale generation demo.

```bash
./bibna_main
```

## 📂 Project Structure

```
bibna/
├── CMakeLists.txt          # Build configuration
├── docs/                   # Design documents
│   └── brain_inspired...   # The "Bible" of this architecture
├── main.cpp                # Entry point & Test Suite
└── src/                    # Source code
    ├── acc.hpp             # Conflict monitoring (Partial)
    ├── bitlinear.hpp       # 1.58-bit Linear Layer
    ├── brain_block.hpp     # Parallel Circuit Manager
    ├── cortical_hub.hpp    # Fast Weights & Attention
    ├── kernels.hpp         # SYCL Kernels (The "Metal")
    ├── micro_circuit.hpp   # Recurrent Unit
    └── spiking_unit.hpp    # Neuron Model
```

## 🤝 Contributing

Contributions are welcome! Please read the design document in `docs/` thoroughly before submitting Pull Requests. Since this is an experimental architecture, we value **clean, documented code** over raw feature speed.

## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See the [LICENSE](LICENSE) file for details.