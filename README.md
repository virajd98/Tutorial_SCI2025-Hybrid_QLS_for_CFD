# Variational Quantum Linear Solver (VQLS) for CFD

A hybrid quantum-classical implementation of **Variational Quantum Linear Solvers** for solving systems of linear equations on quantum computers, with applications to Computational Fluid Dynamics (CFD).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

The **Variational Quantum Linear Solver (VQLS)** is designed to solve linear systems of the form:

$$
A \cdot x = b
$$

where $A$ is a square matrix (typically symmetric/Hermitian) and $x$ is the solution we seek to find. This implementation uses parameterized quantum circuits optimized through classical gradient descent—a paradigm known as **quantum-classical hybrid computing**.

## Installation

### Prerequisites

- Python 3.10+

### Quick Start

```bash
# Clone the repository
git clone https://github.com/virajd98/Tutorial_SCI2025-Hybrid_QLS_for_CFD.git
cd Tutorial_SCI2025-Hybrid_QLS_for_CFD

# Install dependencies
pip install - r requirements.txt
```

## Architecture

```
vqls_prototype/
├── solver/                          # Core solvers
│   ├── base_solver.py              # Abstract base class
│   ├── vqls.py                     # Standard VQLS solver
│   ├── log.py                      # Optimization logging
│   └── validation.py               # Parameter validation
│
├── matrix_decomposition/            # Matrix → Quantum Circuit conversion
│   ├── matrix_decomposition.py     # Base decomposition classes
│   │   ├── MatrixDecomposition
│   │   ├── SymmetricDecomposition
│   │   └── PauliDecomposition
│   └── optimized_matrix_decomposition.py
│       ├── ContractedPauliDecomposition
│       └── OptimizedPauliDecomposition
│
├── hadamard_test/                   # Hadamard test circuits
│   ├── hadamard_test.py            # Standard Hadamard test
│   ├── hadamard_overlap_test.py    # Overlap-based test
│   └── direct_hadamard_test.py     # Direct measurement test
│
│
└── primitives_run_builder/          # Backend abstraction
    ├── base_run_builder.py         # Base builder pattern
    ├── estimator_run_builder.py    # Estimator routing
    └── sampler_run_builder.py      # Sampler routing
```

## Usage Scenarios

### Scenario 1: Standard VQLS for Symmetric Matrices

Use this for most PDE discretizations (Laplacian, diffusion equations):

```python
from vqls_prototype.solver.vqls import VQLS

vqls = VQLS(estimator, ansatz, optimizer, sampler,
            options={"matrix_decomposition": "symmetric"})
result = vqls.solve(A_symmetric, b)
```

### Scenario 2: Hybrid QST for Better State Extraction

Use when final state quality is critical:

```python
from vqls_prototype.solver.hybrid_qst_vqls import Hybrid_QST_VQLS

hybrid_vqls = Hybrid_QST_VQLS(
    estimator, ansatz, optimizer, sampler,
    tomography_method="qst"  # Options: "qst", "simulator", "shadow", "htree"
)
result = hybrid_vqls.solve(A, b)
```

### Scenario 3: Pauli Decomposition for General Matrices

Use for non-symmetric or complex-valued matrices:

```python
vqls = VQLS(estimator, ansatz, optimizer, sampler,
            options={"matrix_decomposition": "pauli"})
result = vqls.solve(A_general, b)
```

## Solver Comparison

| Aspect | VQLS | Hybrid_QST_VQLS | QST_VQLS |
|--------|------|-----------------|----------|
| **Optimization** | Hadamard test | Direct Hadamard | QST-based |
| **State Extraction** | Implicit (via optimization) | Explicit tomography | Full tomography |
| **Measurement Overhead** | Low | Medium | High |
| **Robustness** | Standard | Better for small systems | Experimental |
| **Use Case** | General PDE/CFD | When state quality matters | Research/Legacy |

## Notebooks & Examples

- **`Tut_final.ipynb`**: Interactive tutorial covering VQLS concepts, Ising systems, and complete workflows
- **`blocks_demo.ipynb`**: Demonstration of blocked circuit operations
- **`run.py`**: Standalone Python script solving a 2×2 Laplacian system with convergence visualization

## Key Concepts

### Variational Ansatz
The solver uses parameterized quantum circuits (default: `RealAmplitudes`) to represent candidate solutions:

$$|\psi(\theta)\rangle = V(\theta) |0\rangle$$

Parameters $\theta$ are optimized classically to minimize the cost function.

### Cost Function
The VQLS minimizes:

$$C(\theta) = \langle \psi(\theta) | A^\dagger (\mathbb{I} - |b\rangle\langle b|) A | \psi(\theta) \rangle$$

### Matrix Decomposition
Classical matrices are decomposed into Pauli/unitary terms for quantum circuit representation:

- **Symmetric decomposition** (more efficient): For $A = A^\dagger$
- **Pauli decomposition** (general): Works for any matrix

### State Tomography
After optimization, the final quantum state can be reconstructed via:
- **Full QST**: Complete characterization (high measurement cost)
- **Shadow QST**: Classical shadow protocol (efficient)
- **Simulator QST**: Direct statevector on simulators

## Configuration Options

When creating a solver, customize behavior via the `options` dictionary:

```python
vqls = VQLS(
    estimator, ansatz, optimizer, sampler,
    options={
        "matrix_decomposition": "symmetric",      # "symmetric" or "pauli"
        "use_overlap_test": False,                # Use overlap-based cost
        "use_local_cost_function": False,         # Term-by-term vs global
        "shots": 1024,                            # Measurement shots
        "verbose": True,                          # Enable logging
    }
)
```

## Accessing Optimization History

Track convergence via the solver's logger:

```python
result = vqls.solve(A, b)

# Plot cost function vs iteration
import matplotlib.pyplot as plt
plt.plot(vqls.logger.values)
plt.xlabel("Iteration")
plt.ylabel("Cost Function")
plt.title("VQLS Convergence")
plt.show()

# Access final parameters
final_params = vqls.logger.parameters[-1]
```

## Important Constraints & Considerations

### Matrix Size
- Must be a **power of 2** (2×2, 4×4, 8×8, etc.) for direct qubit mapping
- Number of qubits: $n = \log_2(\text{matrix size}) + 1$ (includes ancilla)

### Normalization
- Input vector `b` **must be normalized**: $\|b\| = 1$
- Output requires post-processing rescaling

### Ansatz Design
- Qubit count must match system size (see above)
- Default `RealAmplitudes` with full entanglement is recommended for CFD problems
- Higher repetitions (`reps`) improve expressivity but increase barren plateau risk

### Numerical Challenges
1. **Barren plateaus**: Gradient vanishing near random initialization
   - Solution: Use warm-start or bounded parameter initialization
2. **Measurement noise**: Shot limitations on real hardware
   - Solution: Increase shots or use error mitigation
3. **Scaling**: VQLS is limited to small systems (~8 qubits for simulators)

## Validation & Testing

### Fidelity Metric (Primary)
Compares quantum vs classical solutions:

```python
x_quantum = np.real(Statevector(result.state).data)
x_classical = np.linalg.solve(A, b)
x_classical /= np.linalg.norm(x_classical)
fidelity = np.abs(np.dot(np.conj(x_quantum), x_classical))**2
print(f"Fidelity: {fidelity:.4f}")  # Target: > 0.95
```

### Residual Norm
Measures solution quality:

```python
residual = np.linalg.norm(A @ x_quantum - b)
print(f"Residual: {residual:.6f}")
```

## Dependencies

### Core Quantum Framework
- **Qiskit** (≥0.40): Quantum circuit construction and optimization
- **Qiskit Aer**: Quantum simulators
- **Qiskit Algorithms**: Variational algorithms and optimizers
- **Qiskit Experiments**: State tomography implementations

### Numerical Computing
- **NumPy**: Linear algebra
- **SciPy**: Advanced numerical methods
- **Sparse**: Efficient sparse matrix representation (for QST_VQLS)

### Utilities
- **Matplotlib**: Visualization
- **tqdm**: Progress bars

## Publications & References

- **Original Paper**: Bravo-Prieto et al. (2019) "Variational Quantum Linear Solver"  
  arXiv:[1909.05820](https://arxiv.org/abs/1909.05820)

- **Tutorial**: This repository is developed for the SCI 2025 Hybrid VQLS for CFD tutorial

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

**Last Updated**: November 2025  
**Maintained by**: [virajd98](https://github.com/virajd98) and [DenisPoisson](https://github.com/DenisPoisson)