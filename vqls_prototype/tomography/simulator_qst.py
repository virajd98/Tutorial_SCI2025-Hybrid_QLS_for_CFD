import numpy as np
from qiskit.quantum_info import Statevector


class SimulatorQST:
    def __init__(self, circuit):
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits

    def get_relative_amplitude_sign(self, parameters):
        """_summary_

        Args:
            circuit (_type_): _description_
            parameters (_type_): _description_
            backend (_type_): _description_
        """
        state_vector = (
            Statevector(self.circuit.assign_parameters(parameters))
        ).data.real
        return np.sign(state_vector)

    def get_statevector(self, parameters, **kwargs):  # pylint: disable=unused-argument
        """_summary_

        Args:
            circuit (_type_): _description_
            parameters (_type_): _description_
            backend (_type_): _description_
        """
        return (Statevector(self.circuit.assign_parameters(parameters))).data.real
