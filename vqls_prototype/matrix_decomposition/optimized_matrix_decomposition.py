"""Methods to decompose a matrix into quantum circuits"""

import itertools
from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Union, List, TypeVar
from qiskit.quantum_info import SparsePauliOp
import numpy as np
import numpy.typing as npt
from qiskit.circuit import QuantumCircuit
import networkx as nx
from tqdm import tqdm

from .matrix_decomposition import PauliDecomposition


complex_type = TypeVar("complex_type", float, complex)
complex_array_type = npt.NDArray[np.cdouble]


class ContractedPauliDecomposition(PauliDecomposition):
    """A class that represents the Pauli decomposition of a matrix with added
    attributes representing the simplification of the Al.T Am terms.

    We first contract the Al.T Am terms in a single Pauli string and
    indentify unique pauli strings. This leads to a first considerable reduction
    of the number of gates

    We then replace the hadammard test by direct measurement on the unique pauli strings.
    Since some of the unique pauli strings are qubit wise commutatives
    we can measure a reduced number of circuits
    """

    contraction_dict = {
        "II": ("I", 1),
        "IX": ("X", 1),
        "IY": ("Y", 1),
        "IZ": ("Z", 1),
        "XI": ("X", 1),
        "YI": ("Y", -1),
        "ZI": ("Z", 1),
        "XX": ("I", 1),
        "YY": ("I", -1),
        "ZZ": ("I", 1),
        "XY": ("Z", 1.0j),
        "YX": ("Z", 1.0j),
        "XZ": ("Y", -1.0j),
        "ZX": ("Y", 1.0j),
        "YZ": ("X", -1.0j),
        "ZY": ("X", -1.0j),
    }

    inverse_contraction_dict = {
        "I": [["I", "X", "Y", " Z"], ["I", "X", "Y", " Z"], [1, 1, -1, 1]],
        "X": [["X", "I", "Y", "Z"], ["I", "X", "Z", "Z"], [1, 1, -1.0j, -1.0j]],
        "Y": [["Y", "I", "X", "Z"], ["I", "Y", "Z", "X"], [-1, 1, -1.0j, 1.0j]],
        "Z": [["Z", "I", "X", "Y"], ["I", "Z", "Y", "Z"], [1, 1, 1.0j, 1.0j]],
    }

    def __init__(
        self,
        matrix: Optional[npt.NDArray] = None,
        circuits: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        coefficients: Optional[
            Union[float, complex, List[float], List[complex]]
        ] = None,
        load: Optional[str] = None,
        sparse: Optional[bool] = False,
    ):
        super().__init__(matrix, circuits, coefficients, load, sparse)
        self.contract_pauli_terms(build_circuits=True)

    def contract_pauli_terms(self, build_circuits=True) -> None:
        """Compute the contractions of the Pauli Strings.

        Returns:
            Tuple[complex_array_type, List[complex_array_type]]:
                A tuple containing the list of coefficients and
                the numpy matrix of the decomposition.
        """
        self.contraction_map = []
        self.contraction_coefficient = []
        self.contracted_circuits = []
        self.contraction_index_mapping = []

        self.unique_pauli_strings = []
        number_existing_circuits = 0
        nstrings = len(self.strings)
        str_len = range(self.num_qubits)
        index_contracted_pauli = {}

        # loop over combination of gates
        for i1 in tqdm(range(nstrings)):
            for i2 in range(i1 + 1, nstrings):
                # extract pauli strings
                pauli_string_1, pauli_string_2 = self.strings[i1], self.strings[i2]
                contracted_pauli_string, contracted_coefficient = "", 1.0 + 0.0j

                # contract pauli gates qubit wise
                for ip in str_len:
                    pauli1, pauli2 = pauli_string_1[ip], pauli_string_2[ip]
                    pauli, coefficient = self.contraction_dict[pauli1 + pauli2]
                    contracted_pauli_string += pauli
                    contracted_coefficient *= coefficient

                # contraction map -> not  needed
                self.contraction_map.append(
                    [
                        (pauli_string_1, pauli_string_2),
                        (contracted_pauli_string, contracted_coefficient),
                    ]
                )

                # store circuits if we haven't done that yet
                if contracted_pauli_string not in index_contracted_pauli:
                    self.unique_pauli_strings.append(contracted_pauli_string)
                    if build_circuits:
                        self.contracted_circuits.append(
                            self._create_circuit(contracted_pauli_string)
                        )
                    self.contraction_index_mapping.append(number_existing_circuits)
                    index_contracted_pauli[
                        contracted_pauli_string
                    ] = number_existing_circuits
                    number_existing_circuits += 1

                # otherwise find reference of existing circuit
                else:
                    self.contraction_index_mapping.append(
                        # self.unique_pauli_strings.index(contracted_pauli_string)
                        index_contracted_pauli[contracted_pauli_string]
                    )

                # store the contraction coefficient
                self.contraction_coefficient.append(contracted_coefficient)

    def _find_contracted_pairs(self, pauli):
        """_summary_

        Args:
            pauli (_type_): _description_
        """
        first_pauli, second_pauli, coefficients = [], [], []
        for p in pauli:
            p1, p2, c = self.inverse_contraction_dict[p]
            first_pauli.append(p1)
            second_pauli.append(p2)
            coefficients.append(c)

        return (
            itertools.product(*first_pauli),
            itertools.product(*second_pauli),
            itertools.product(*coefficients),
        )

    def post_process_contracted_norm_values(self, hdmr_values_norm):
        """Post process the measurement obtained with the direct

        Args:
            hdmr_values_norm (list): list of measrurement values
        """

        # map the values onto the index of the  Al.T Am terms
        hdmr_values_norm = hdmr_values_norm[self.contraction_index_mapping] * np.array(
            self.contraction_coefficient
        )

        return hdmr_values_norm


@dataclass
class OptimizationMeasurementGroup:
    cluster: OrderedDict
    eigenvalues: List
    index_mapping: List
    shared_basis_string: List
    shared_basis_transformation: List


class OptimizedPauliDecomposition(ContractedPauliDecomposition):
    def __init__(
        self,
        matrix: Optional[npt.NDArray] = None,
        vector: Optional[npt.NDArray] = None,
        circuits: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        coefficients: Optional[
            Union[float, complex, List[float], List[complex]]
        ] = None,
    ):
        super().__init__(matrix, circuits, coefficients)

        # create the sparse pauli matrices of the single terms
        if vector is not None:
            self.vector_pauli_product = self.get_vector_pauli_product(vector)

        # add the single pauli terms
        self.num_unique_norm_terms = len(self.unique_pauli_strings)
        self.num_unique_overlap_terms = len(self.strings)

        # compute the measurement optimized mapping
        self.optimized_measurement = self.group_contracted_terms()

    def get_vector_pauli_product(self, vector):
        """get the sparese representation of the pauli matrices

        Returns:
            _type_: _description_
        """
        return [
            SparsePauliOp(pauli).to_matrix(sparse=True) @ vector
            for pauli in self.strings
        ]

    @staticmethod
    def _string_qw_commutator(pauli1, pauli2):
        """assesses if two pauli string qubit-wise commutes or not

        Args:
            pauli1 (str): first puali string
            pauli2 (str): 2nd pauli string
        """

        return np.all(
            [(p1 == p2) | (p1 == "I") | (p2 == "I") for p1, p2 in zip(pauli1, pauli2)]
        )

    @staticmethod
    def _get_eigenvalues(pauli_string):
        """Compute the eigenvalue of the string"""
        ev_dict = {"X": [1, -1], "Y": [1, -1], "Z": [1, -1], "I": [1, 1]}
        pauli_string = pauli_string[::-1]
        evs = ev_dict[pauli_string[0]]
        for p in pauli_string[1:]:
            evs = np.kron(ev_dict[p], evs)
        return evs

    @staticmethod
    def _determine_shared_basis_string(pauli_strings):
        """determine the shared basis string for a list of pauli strings

        Args:
            pauli_strings (List): list of pauli string

        Returns:
            string: the shared basis pauli string
        """
        shared_basis = np.array(["I"] * len(pauli_strings[0]))
        for pstr in pauli_strings:
            for ig, pgate in enumerate(list(pstr)):
                if pgate != "I":
                    shared_basis[ig] = pgate

            if not np.any(shared_basis == "I"):
                break
        return shared_basis

    @staticmethod
    def _create_shared_basis_circuit(pauli_string):
        """create the circuit needed to rotate the qubits in the shared eigenbasis"""

        num_qubits = len(pauli_string)
        circuit = QuantumCircuit(num_qubits)
        for iqbit in range(num_qubits):
            op = pauli_string[iqbit]
            if op == "X":
                circuit.ry(-np.pi / 2, iqbit)
            if op == "Y":
                circuit.rx(np.pi / 2, iqbit)

        return circuit

    def _create_qwc_graph(self):
        """Creates a nx graph representing the qwc map"""

        # create qwc edges
        nstrings = len(self.unique_pauli_strings)
        qwc_graph_edges = []
        for i1 in tqdm(range(nstrings)):
            for i2 in range(i1 + 1, nstrings):
                pauli_string_1, pauli_string_2 = (
                    self.unique_pauli_strings[i1],
                    self.unique_pauli_strings[i2],
                )
                if self._string_qw_commutator(pauli_string_1, pauli_string_2):
                    qwc_graph_edges.append([pauli_string_1, pauli_string_2])

        # create graph
        qwc_graph = nx.Graph()
        qwc_graph.add_nodes_from(self.unique_pauli_strings)
        qwc_graph.add_edges_from(qwc_graph_edges)

        return qwc_graph

    @staticmethod
    def _get_commuting_strings(pauli_string):
        """Returns all the strings that commute with string

        Args:
            string (_type_): _description_
        """
        all_gates = ["I", "X", "Y", "Z"]
        basis = []
        for p in pauli_string:
            if p == "I":
                basis.append(all_gates)
            if p != "I":
                basis.append(["I", p])
        return ["".join(s) for s in itertools.product(*basis)]

    def _create_qwc_graph_fast(self):
        """Creates a nx graph representing the qwc map"""

        # create qwc edges
        qwc_graph_edges = []
        qwc_graph = nx.Graph()
        qwc_graph.add_nodes_from(self.unique_pauli_strings)

        for pauli_string in tqdm(self.unique_pauli_strings):
            commuting_strings = self._get_commuting_strings(pauli_string)
            for cs in commuting_strings:
                if cs in qwc_graph.nodes:
                    if cs != pauli_string:
                        qwc_graph_edges.append([pauli_string, cs])

        # add edges
        qwc_graph.add_edges_from(qwc_graph_edges)

        return qwc_graph

    def _cluster_graph(self, qwc_complement_graph, strategy="largest_first"):
        """Cluster the qwc graph"""

        # greedy clustering
        qwc_groups_flat = nx.coloring.greedy_color(
            qwc_complement_graph, strategy=strategy
        )
        return qwc_groups_flat

    def _cluster_graph_fast(self):
        processed_strings = OrderedDict()
        qwc_cluster = OrderedDict()
        icluster = 0
        for pauli_string in tqdm(self.unique_pauli_strings):
            if "I" in pauli_string:
                continue
            if pauli_string in processed_strings:
                continue
            commuting_strings = self._get_commuting_strings(pauli_string)
            commuting_strings = [
                cs
                for cs in commuting_strings
                if (cs not in processed_strings) and (cs in self.unique_pauli_strings)
            ]
            qwc_cluster[icluster] = commuting_strings
            for cs in commuting_strings:
                processed_strings[cs] = icluster
            icluster += 1
        if len(processed_strings.keys()) != len(self.unique_pauli_strings):
            print("Fast Clustering failed, default to full method")
            return None
        return processed_strings

    def group_contracted_terms(self, return_group=False):
        """Finds the qubit wise commutating operator to further
        optimize the number of measurements."""

        qwc_cluster = self._cluster_graph_fast()

        if qwc_cluster is None:
            # compute the complement of the qwc graph
            qwc_complement_graph = nx.complement(self._create_qwc_graph_fast())

            # determine the cluster
            qwc_cluster = self._cluster_graph(qwc_complement_graph)

        if return_group:
            return np.unique([v for _, v in qwc_cluster.items()])

        # organize the groups
        nstrings = len(self.unique_pauli_strings)
        optimized_measurement = OptimizationMeasurementGroup(
            OrderedDict(), [None] * nstrings, [None] * nstrings, [], []
        )

        # loop over the qwc cluster from nx
        for pauli, group_id in qwc_cluster.items():
            # populate cluster data in optimized_measurement
            if group_id not in optimized_measurement.cluster:
                optimized_measurement.cluster[group_id] = []
            optimized_measurement.cluster[group_id].append(pauli)

            # populate the eigenvalue data in optimized_measurement
            optimized_measurement.eigenvalues[
                self.unique_pauli_strings.index(pauli)
            ] = self._get_eigenvalues(pauli)

            # populate the index mapping in optimized_measurement
            optimized_measurement.index_mapping[
                self.unique_pauli_strings.index(pauli)
            ] = group_id

        # determine the shared eigenbasis
        optimized_measurement.shared_basis_transformation = []
        optimized_measurement.shared_basis_string = []
        for group_id, pauli_strings in optimized_measurement.cluster.items():
            shared_basis = self._determine_shared_basis_string(pauli_strings)

            optimized_measurement.shared_basis_string.append(list(shared_basis[::-1]))
            optimized_measurement.shared_basis_transformation.append(
                self._create_shared_basis_circuit(shared_basis[::-1])
            )
        optimized_measurement.shared_basis_string = np.array(
            optimized_measurement.shared_basis_string
        )
        return optimized_measurement

    def get_norm_values(self, samples):
        """Post process the measurement obtained with the direct hadamard test
        to form the norm values

        Args:
            samples (list): list of sample circuit output values
        """

        # map the sampled values to all the group members (unique pauli strings)
        index = self.optimized_measurement.index_mapping
        out = samples[index]

        # get the eigenvalues of the pauli string
        eigenvalues = self.optimized_measurement.eigenvalues

        # mulitpliy the sampled values with the eigenvalues
        # of each transformation
        out = np.array([np.dot(ev, val) for ev, val in zip(eigenvalues, out)])

        # map the values onto the index of the  Al.T Am terms
        out = out[self.contraction_index_mapping] * np.array(
            self.contraction_coefficient
        )

        return out

    def get_overlap_values(self, samples, sign_ansatz):
        """Post process the measurement obtained with the direct hadamard test
        to form the overlap values

        Args:
            samples (list): list of sample circuit output values
            ansatz_sign (np.array): sign of the amplitude of the ansatz
        """
        output = []
        for ipaulis in range(len(self.vector_pauli_product)):
            output.append(
                np.dot(
                    sign_ansatz * np.sqrt(samples),
                    self.vector_pauli_product[ipaulis],
                )
            )

        return np.array(output).flatten()
