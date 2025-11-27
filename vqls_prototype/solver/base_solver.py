# Base solver class
from typing import Optional, Union, List, Callable, Dict, Tuple
from copy import deepcopy
import logging
import numpy as np


from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimator, BaseSampler
from qiskit_algorithms.variational_algorithm import VariationalAlgorithm
from qiskit_algorithms.utils.validation import validate_min
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
from qiskit_algorithms.optimizers import Minimizer, Optimizer
from qiskit_algorithms.gradients import BaseEstimatorGradient

from .variational_linear_solver import (
    VariationalLinearSolver,
    VariationalLinearSolverResult,
)

from ..matrix_decomposition.matrix_decomposition import MatrixDecomposition
from .log import VQLSLog


class BaseSolver(VariationalAlgorithm, VariationalLinearSolver):
    r"""Systems of linear equations arise naturally in many real-life applications in a wide range
    of areas, such as in the solution of Partial Differential Equations, the calibration of
    financial models, fluid simulation or numerical field calculation. The problem can be defined
    as, given a matrix :math:`A\in\mathbb{C}^{N\times N}` and a vector
    :math:`\vec{b}\in\mathbb{C}^{N}`, find :math:`\vec{x}\in\mathbb{C}^{N}` satisfying
    :math:`A\vec{x}=\vec{b}`.


    References:

        [1] Carlos Bravo-Prieto, Ryan LaRose, M. Cerezo, Yigit Subasi, Lukasz Cincio,
        Patrick J. Coles. Variational Quantum Linear Solver
        `arXiv:1909.05820 <https://arxiv.org/abs/1909.05820>`
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        ansatz: QuantumCircuit,
        optimizer: Union[Optimizer, Minimizer],
        sampler: Union[BaseSampler, None],
        initial_point: Union[np.ndarray, None],
        gradient: Union[BaseEstimatorGradient, Callable, None],
        max_evals_grouped: int,
    ) -> None:
        r"""
        Args:
            estimator: an Estimator primitive to compute the expected values of the
                quantum circuits needed for the cost function
            ansatz: A parameterized circuit used as Ansatz for the wave function.
            optimizer: A classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            sampler: a Sampler primitive to sample the output of some quantum circuits needed to
                compute the cost function. This is only needed if overal Hadammard tests are used.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQLS will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            gradient: An optional gradient function or operator for optimizer.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time. Deprecated if a gradient operator or function is
                given.
        """
        super().__init__()

        validate_min("max_evals_grouped", max_evals_grouped, 1)

        self._num_qubits = None
        self._max_evals_grouped = max_evals_grouped

        self.estimator = estimator
        self.sampler = sampler
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.initial_point = initial_point

        self._gradient = None
        self.gradient = gradient

        self.logger = VQLSLog([], [])
        self.callback = self.logger.update

        self._eval_count = 0

        self.vector_circuit = QuantumCircuit(0)
        self.matrix_circuits: MatrixDecomposition = None  # QuantumCircuit(0)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.info("Estimator : %s", self.estimator.__class__.__module__)

    @property
    def num_qubits(self) -> int:
        """return the numner of qubits"""
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits"""
        self._num_qubits = num_qubits

    @property
    def num_clbits(self) -> int:
        """return the numner of classical bits"""
        return self._num_clbits

    @num_clbits.setter
    def num_clbits(self, num_clbits: int) -> None:
        """Set the number of classical bits"""
        self._num_clbits = num_clbits

    @property
    def ansatz(self) -> QuantumCircuit:
        """Returns the ansatz."""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: Optional[QuantumCircuit]):
        """Sets the ansatz.

        Args:
            ansatz: The parameterized circuit used as an ansatz.
            If None is passed, RealAmplitudes is used by default.

        """
        if ansatz is None:
            ansatz = RealAmplitudes()

        self._ansatz = ansatz
        self.num_qubits = ansatz.num_qubits + 1

    @property
    def initial_point(self) -> Union[np.ndarray, None]:
        """Returns initial point"""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Union[np.ndarray, None]):
        """Sets initial point"""
        self._initial_point = initial_point

    @property
    def max_evals_grouped(self) -> int:
        """Returns max_evals_grouped"""
        return self._max_evals_grouped

    @max_evals_grouped.setter
    def max_evals_grouped(self, max_evals_grouped: int):
        """Sets max_evals_grouped"""
        self._max_evals_grouped = max_evals_grouped
        self.optimizer.set_max_evals_grouped(max_evals_grouped)

    @property
    def callback(self) -> Callable[[int, float, np.ndarray], None]:
        """Returns callback"""
        return self._callback

    @callback.setter
    def callback(self, callback: Callable[[int, float, np.ndarray], None]):
        """Sets callback"""
        self._callback = callback

    @property
    def optimizer(self) -> Optimizer:
        """Returns optimizer"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optional[Optimizer]):
        """Sets the optimizer attribute.

        Args:
            optimizer: The optimizer to be used.

        """

        if isinstance(optimizer, Optimizer):
            optimizer.set_max_evals_grouped(self.max_evals_grouped)

        self._optimizer = optimizer

    def construct_circuit(
        self,
        matrix: Union[np.ndarray, QuantumCircuit, List],
        vector: Union[np.ndarray, QuantumCircuit],
    ) -> Tuple[List[QuantumCircuit], List[QuantumCircuit]]:
        raise NotImplementedError("Please implement construc_circuit")

    def _assemble_cost_function(
        self,
        hdmr_values_norm: np.ndarray,
        hdmr_values_overlap: np.ndarray,
        coefficient_matrix: np.ndarray,
    ) -> float:
        raise NotImplementedError("Please implement assemble_cost function")

    def get_cost_evaluation_function(
        self,
        hdmr_tests_norm: List,
        hdmr_tests_overlap: List,
        coefficient_matrix: np.ndarray,
    ) -> Callable[[np.ndarray], Union[float, List[float]]]:
        raise NotImplementedError("Please implement get_cost_evaluation_function")

    def _validate_solve_options(self, options: Union[Dict, None]) -> Dict:
        """validate the options used for the solve methods

        Args:
            options (Union[Dict, None]): options
        """
        raise NotImplementedError("Please implement _validate_solve_options")

    def _solve(
        self,
        matrix: Union[np.ndarray, QuantumCircuit, List[QuantumCircuit]],
        vector: Union[np.ndarray, QuantumCircuit],
    ) -> VariationalLinearSolverResult:
        raise NotImplementedError("Please implement _solve method")

    @staticmethod
    def get_coefficient_matrix(coeffs) -> np.ndarray:
        """Compute all the vi* vj terms

        Args:
            coeffs (np.ndarray): list of complex coefficients
        """
        return coeffs[:, None].conj() @ coeffs[None, :]

    def _compute_normalization_term(
        self,
        coeff_matrix: np.ndarray,
        hdmr_values: np.ndarray,
    ) -> float:
        """Compute <phi|phi>

        .. math::
            \\langle\\Phi|\\Phi\\rangle = \\sum_{nm} c_n^*c_m \\langle 0|V^* U_n^* U_m V|0\\rangle

        Args:
            coeff_matrix (List): the matrix values of the c_n^* c_m coefficients
            hdmr_values (List): the values of the circuits output

        Returns:
            float: value of the sum
        """

        # compute all the terms in <\phi|\phi> = \sum c_i* cj <0|V Ai* Aj V|0>
        # hdrm_values here contains the values of the <0|V Ai* Aj V|0>  with j>i
        # out = np.copy(hdmr_values)
        out = hdmr_values

        # we multiply hdmr values by the triup coeff matrix and sum
        out *= coeff_matrix[np.triu_indices_from(coeff_matrix, k=1)]
        out = out.sum()

        # add the conj that corresponds to the tri down matrix
        out += out.conj()

        # add the diagonal terms
        # since <0|V Ai* Aj V|0> = 1 we simply
        # add the sum of the cici coeffs
        out += np.trace(coeff_matrix)

        return out.item()

    def _compute_global_terms(
        self,
        coeff_matrix: np.ndarray,
        hdmr_values: np.ndarray,
    ) -> float:
        """Compute |<b|phi>|^2

        .. math::
            |\\langle b|\\Phi\\rangle|^2 = \\sum_{nm} c_n^*c_m \\gamma_{nm}

        with

        .. math::

            \\gamma_nm = \\langle 0|V^* U_n^* U_b |0 \\rangle \\langle 0|U_b^* U_m V |0\\rangle

        Args:
            coeff_matrix (np.ndarray): the matrix values of the c_n^* c_m coefficients
            hdmr_values (List): values of the circuit outputs

        Returns:
            float: value of the sum
        """

        if self.options["use_overlap_test"]:
            # hdmr_values here contains the values of <0|V* Ai* U|0><0|V Aj U|0> for j>=i
            # we first insert these values in a tri up matrix
            size = len(self.matrix_circuits)
            hdmr_matrix = np.zeros((size, size)).astype("complex128")
            hdmr_matrix[np.tril_indices(size)] = hdmr_values

            # add the conj that correspond to the tri low part of the matrix
            # warning the diagonal is also contained in out and we only
            # want to add the conj of the tri up excluding the diag
            hdmr_matrix[np.triu_indices_from(hdmr_matrix, k=1)] = hdmr_matrix[
                np.tril_indices_from(hdmr_matrix, k=-1)
            ].conj()

            # multiply by the coefficent matrix and sum the values
            out_matrix = coeff_matrix * hdmr_matrix
            out = out_matrix.sum()

        else:
            # hdmr_values here contains the values of <0|V* Ai* U|0>
            # compute the matrix of the <0|V* Ai* U|0> <0|V Aj U*|0> values
            hdmr_matrix = self.get_coefficient_matrix(hdmr_values)
            out = (coeff_matrix * hdmr_matrix).sum()

        return out

    def _compute_local_terms(
        self, coeff_matrix: np.ndarray, hdmr_values: np.ndarray, norm: float
    ) -> float:
        """Compute the term of the local cost function given by

        .. math::
            \\sum c_i^* c_j \\frac{1}{n} \\sum_n \\langle 0|V^* A_i U Z_n U^* A_j^* V|0\\rangle

        Args:
            coeff_matrix (np.ndarray): the matrix values of the c_n^* c_m coefficients
            hdmr_values (List): values of the circuit outputs
            norm (float): value of the norm term

        Returns:
            float: value of the sum
        """

        # add all the hadamard test values corresponding to the insertion
        # of Z gates on the same cicuit
        # b_ij = \sum_n \\frac{1}{n} \\sum_n \\langle 0|V^* A_i U Z_n U^* A_j^* V|0\\rangle
        num_zgate = self.matrix_circuits[0].circuit.num_qubits
        hdmr_values = hdmr_values.reshape(-1, num_zgate).mean(1)

        # hdmr_values then contains the values of <0|V* Ai* U|0><0|V Aj U|0> for j>=i
        # we first insert these values in a tri up matrix
        size = len(self.matrix_circuits)
        hdmr_matrix = np.zeros((size, size)).astype("complex128")
        hdmr_matrix[np.triu_indices(size)] = hdmr_values

        # add the conj that correspond to the tri low part of the matrix
        # warning the diagonal is also contained in out and we only
        # want to add the conj of the tri up excluding the diag
        hdmr_matrix[np.tril_indices_from(hdmr_matrix, k=-1)] = hdmr_matrix[
            np.triu_indices_from(hdmr_matrix, k=1)
        ].conj()

        # multiply by the coefficent matrix and sum the values
        out_matrix = coeff_matrix * hdmr_matrix
        out = (out_matrix).sum()

        # add \sum c_i* cj <0|V Ai* Aj V|0>
        out += norm

        # factor two coming from |0><0| = 1/2(I+Z)
        out /= 2

        return out

    def solve(
        self,
        matrix: Union[np.ndarray, QuantumCircuit, List[QuantumCircuit]],
        vector: Union[np.ndarray, QuantumCircuit],
    ) -> VariationalLinearSolverResult:
        """_summary_

        Args:
            matrix (Union[np.ndarray, QuantumCircuit, List[QuantumCircuit]]): _description_
            vector (Union[np.ndarray, QuantumCircuit]): _description_

        Returns:
            VariationalLinearSolverResult: _description_
        """

        # make a copy of the optimizers
        optimizers_list = deepcopy(self.optimizer)

        if not isinstance(optimizers_list, List):
            optimizers_list = [optimizers_list]

        for opt in optimizers_list:
            self.optimizer = opt
            solution = self._solve(matrix, vector)
            self.initial_point = solution.optimal_point

        # reset the optimizer to its original value
        self.optimizer = optimizers_list

        return solution
