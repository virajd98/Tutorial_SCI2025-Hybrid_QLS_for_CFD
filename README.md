# Hybrid Quantum Linear Solver: A hands on tutorial with Qiskit

A hybrid quantum-classical implementation of **Variational Quantum Linear Solvers** for solving systems of linear equations on quantum computers, with applications to Computational Fluid Dynamics (CFD).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

The **Variational Quantum Linear Solver (VQLS)** is designed to solve linear systems of the form:

$$
A \cdot x = b
$$

where $A$ is a square matrix (typically symmetric/Hermitian) and $x$ is the solution we seek to find. This implementation uses parameterized quantum circuits optimized through classical gradient descent—a paradigm known as **quantum-classical hybrid computing**.

## Installation


### Requirements
```
- Python **3.11+**
```

### Clone the repository
```
git clone https://github.com/virajd98/Tutorial_SCI2025-Hybrid_QLS_for_CFD.git
cd Tutorial_SCI2025-Hybrid_QLS_for_CFD
````

### Setup Environment

**macOS / Linux**

```
chmod +x Run.sh
./Run.sh
```

**Windows (PowerShell)**

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./Run.ps1
```

### Activate the Environment (Manual)

**macOS / Linux**

```
source VQLS_env/bin/activate
```

**Windows (PowerShell)**

```
.\VQLS_env\Scripts\Activate.ps1
```

### Note

When running Jupyter notebooks, make sure to **select the kernel corresponding to the `VQLS_env` environment**.



## Architecture of VQLS Module

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



## Notebooks & Examples

- **`Tut_final.ipynb`**: Interactive tutorial covering VQLS concepts and complete workflow
- **`blocks_demo.ipynb`**: Demonstration of block encoding circuit for a 1D Poisson system




## Configuration Options

When creating a solver, customize behavior via the `options` dictionary:

```
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

## Important Constraints & Considerations

### Matrix Size
- Must be a **power of 2** (2×2, 4×4, 8×8, etc.) for direct qubit mapping

### Normalization
- Input vector `b` **must be normalized**: $\|b\| = 1$
- Output requires post-processing rescaling

### Ansatz Design
- Higher repetitions (`reps`) improve expressivity but increase barren plateau risk

### Fidelity Metric
Compares quantum vs classical solutions:



## Publications & References

- **Original Paper**: Bravo-Prieto et al. (2019) "Variational Quantum Linear Solver"  
  arXiv:[1909.05820](https://arxiv.org/abs/1909.05820)

- **Tutorial**: This repository is developed for the tutorial on Hybrid Quantum Linear Solvers at Super Computing India(SCI), 2025

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

**Last Updated**: December 2025  
**Maintained by**: [virajd98](https://github.com/virajd98) and [DenisPoisson](https://github.com/DenisPoisson)
