# example_run_vqls.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms import optimizers as opt
from qiskit_aer import Aer
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

# Import the solver class from the repo
from vqls_prototype.solver.vqls import VQLS

#Laplacian Matrix
def generate_laplacian_matrix(Nx, Ny, delta_x, delta_y):
    N = Nx * Ny
    A = np.zeros((N, N))

    A0 = -2 / (delta_x**2) - 2 / (delta_y**2) 
    A1 = 1 / (delta_x**2)  
    A2 = 1 / (delta_y**2)  

    for a1 in range(Nx):
        for b1 in range(Ny):
            index = a1 + b1 * Nx
            
            A[index, index] = A0

            #off-diagonal in x-direction
            if a1 > 0:  
                A[index, index - 1] = A1
            if a1 < Nx - 1:  
                A[index, index + 1] = A1

            #off-diagonal in y-direction
            if b1 > 0:  
                A[index, index - Nx] = A2
            if b1 < Ny - 1:  
                A[index, index + Nx] = A2

    return A

# Problem parameters
Nx, Ny = 2,2         # Grid dimensions
delta_x, delta_y = 1.0, 1.0        # Grid spacing

# Derived parameters
system_size = int(np.log2(Nx) + np.log2(Ny))  # Total system qubits

# Generate and scale Laplacian matrix
laplacian_matrix = generate_laplacian_matrix(Nx, Ny, delta_x, delta_y)

# Target vector (uniform superposition)
b_vector = np.ones(2**system_size) / np.sqrt(2**system_size)
normalized_b = np.array(b_vector) / np.linalg.norm(b_vector)

# Step 2: Define ansatz (parameterized quantum circuit)
ansatz = RealAmplitudes(num_qubits=system_size, reps=2, entanglement="full")


print("Laplacian Matrix:\n", laplacian_matrix)
# Step 3: Choose backend and primitives
backend = Aer.get_backend("statevector_simulator")
estimator = Estimator()
sampler = Sampler()

# Step 4: Choose classical optimizer
optimizer = opt.COBYLA(maxiter=2000)

# Step 5: Create and run the VQLS solver
vqls = VQLS(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler,
    options={
        "matrix_decomposition": "symmetric",  # default
        "use_local_cost_function": False,     # global cost
        "use_overlap_test": False,            # standard Hadamard test
        "verbose": True
    }
)

# Solve Ax = b
result = vqls.solve(laplacian_matrix, normalized_b)

# Step 6: Extract results
x_quantum = np.real(Statevector(result.state).data)

unnormalization_factor = 1 / np.sqrt(np.real(np.conj(x_quantum).T @ (laplacian_matrix.T.conj() @ (laplacian_matrix @ x_quantum))))
x_unnormalized = unnormalization_factor * x_quantum

# Now compare with the true classical solution
x_classical = np.linalg.solve(laplacian_matrix, normalized_b)
# print("Quantum solution (rescaled):", x_unnormalized)
# print("Classical:", x_classical)
x_classical /= np.linalg.norm(x_classical)
print("Fidelity between quantum and classical solution:", np.abs(np.dot(np.conj(x_quantum), x_classical))**2)

# Step 7: Optional plotting (if you logged cost)

if hasattr(vqls, "logger"):
    plt.plot(vqls.logger.values)
    plt.xlabel("Iterations")
    plt.ylabel("Cost Function")
    plt.title("VQLS optimization progress")
    plt.show()
