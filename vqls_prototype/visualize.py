import numpy as np

from qiskit_algorithms import optimizers as opt
from qiskit_aer import Aer
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt

# Import the solver class from the repo
from vqls_prototype.solver.vqls import VQLS

import math
import random
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
from scipy.optimize import minimize
from scipy.sparse import kron, identity, csr_matrix

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library.standard_gates import XGate, YGate, ZGate


import numpy as np
from vqls_prototype.matrix_decomposition.matrix_decomposition import PauliDecomposition



def pauli_matrices():
    X = csr_matrix([[0, 1], [1, 0]])
    Z = csr_matrix([[1, 0], [0, -1]])
    I = csr_matrix(np.eye(2))
    return X, Z, I


def construct_ising_matrix(n, J=0, eta=1.0):
    X, Z, I = pauli_matrices()
    transverse_field = csr_matrix((2 ** n, 2 ** n))
    interaction = csr_matrix((2 ** n, 2 ** n))
    for j in range(n):
        term = [I] * n
        term[j] = X
        result = term[0]
        for mat in term[1:]:
            result = kron(result, mat, format='csr')
        transverse_field += result
    for j in range(n - 1):
        term = [I] * n
        term[j] = Z
        term[j + 1] = Z
        result = term[0]
        for mat in term[1:]:
            result = kron(result, mat, format='csr')
        interaction += result
    identity_term = eta * identity(2 ** n, format='csr')
    A = (transverse_field + J * interaction + identity_term)
    A /= np.linalg.norm(A.toarray(), ord=2)
    return A.toarray()

# ------------------------------------------------------------
# Helper: apply a Pauli string to qubits (non-controlled)
# ----------------------------------------------------------
def apply_pauli_string(circ, pauli_string, qubits):
    for i, p in enumerate(pauli_string):
        if p == 'I':
            continue
        elif p == 'X':
            circ.x(qubits[i])
        elif p == 'Y':
            circ.y(qubits[i])
        elif p == 'Z':
            circ.z(qubits[i])
        else:
            raise ValueError(f"Invalid Pauli {p}")

# ------------------------------------------------------------
# Helper: controlled Pauli string for Hadamard test
# ------------------------------------------------------------
def apply_controlled_pauli_string(circ, control, pauli_string, qubits):
    for i, p in enumerate(pauli_string):
        if p == 'I':
            continue
        elif p == 'X':
            circ.cx(control, qubits[i])
        elif p == 'Y':
            circ.cy(control, qubits[i])
        elif p == 'Z':
            circ.cz(control, qubits[i])
        else:
            raise ValueError(f"Invalid Pauli {p}")

# ------------------------------------------------------------
# Hadamard Test for β_{ll'}
# Computes Re⟨0| V† A_l† A_l' V |0⟩
# ------------------------------------------------------------
def hadamard_test_beta(V, Al, Alp):
    n = V.num_qubits
    ctrl = QuantumRegister(1, "anc")
    qr   = QuantumRegister(n, "q")
    cr   = ClassicalRegister(1, "c")
    
    qc = QuantumCircuit(ctrl, qr, cr)

    # H on control
    qc.h(ctrl)

    # Prepare |x(θ)>
    qc.append(V.to_gate(label="V"), qr)

    # Controlled A_l'
    apply_controlled_pauli_string(qc, ctrl[0], Alp, qr)

    # Controlled A_l† (Pauli strings are Hermitian, † = itself)
    apply_controlled_pauli_string(qc, ctrl[0], Al, qr)

    # Apply V†
    qc.append(V.inverse().to_gate(label="V†"), qr)

    # H on control (measure Re part)
    qc.h(ctrl)

    qc.measure(ctrl, cr)

    return qc


# ------------------------------------------------------------
# Hadamard test for γ_{ll'}
# Computes  ⟨0| U† A_l V |0⟩  and similar cross term
# The same Hadamard test circuit as β but use U instead of V†...
# ------------------------------------------------------------

def hadamard_test_gamma_Al(V, U, Al):
    n = V.num_qubits
    ctrl = QuantumRegister(1, "anc")
    qr   = QuantumRegister(n, "q")
    cr   = ClassicalRegister(1, "c")
    
    qc = QuantumCircuit(ctrl, qr, cr)

    # H on control
    qc.h(ctrl)

    #Apply U
    U_ctrl = U.inverse().to_gate(label="U†").control(1)
    qc.append(U_ctrl, [ctrl[0]] + qr[:])
    

    # Controlled A_lp
    apply_controlled_pauli_string(qc, ctrl[0], Al, qr)

    #Apply V_dagger

    Vdg_ctrl = V.to_gate(label="V").control(1)
    qc.append(Vdg_ctrl, [ctrl[0]] + qr[:])


    # Final H
    qc.h(ctrl)

    qc.measure(ctrl, cr)
    return qc



def hadamard_test_gamma_Alp(V, U, Alp):
    n = V.num_qubits
    ctrl = QuantumRegister(1, "anc")
    qr   = QuantumRegister(n, "q")
    cr   = ClassicalRegister(1, "c")
    
    qc = QuantumCircuit(ctrl, qr, cr)

    # H on control
    qc.h(ctrl)


    #Apply V_dagger

    Vdg_ctrl = V.inverse().to_gate(label="V†").control(1)
    qc.append(Vdg_ctrl, [ctrl[0]] + qr[:])

    

    # Controlled A_lp
    apply_controlled_pauli_string(qc, ctrl[0], Alp, qr)

   
    #Apply U
    U_ctrl = U.to_gate(label="U").control(1)
    qc.append(U_ctrl, [ctrl[0]] + qr[:])


    # Final H
    qc.h(ctrl)

    qc.measure(ctrl, cr)
    return qc


def hardware_efficient_ansatz_param(n, layers=2):
    """
    VQLS-compatible fully parameterized ansatz.
    Uses explicit Parameter objects (most robust for VQLS).
    """
    qc = QuantumCircuit(n)
    params = []   # store parameters (optional, not required by VQLS)


    for i in range(n):
        theta = Parameter(f"theta_{i}")
        params.append(theta)
        qc.ry(theta, i)

    qc.barrier()

    for l in range(layers):
        
        # ---- entangling layer ----
        for q in range(0,n,2):
            qc.cz(q, q + 1)

        # ---- rotation layer ----
        for q in range(n):
            theta = Parameter(f"theta_{l}_{q}")
            params.append(theta)
            qc.ry(theta, q)

        # ---- entangling layer ----
        for q in range(1,n-1,2):
            qc.cz(q, q + 1)

        for q in range(1,n-1):
            theta = Parameter(f"theta2_{l}_{q}")
            params.append(theta)
            qc.ry(theta, q)
        qc.barrier()
        


    return qc
