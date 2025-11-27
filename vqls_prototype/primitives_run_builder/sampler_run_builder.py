"""This module defines the sampler run builder class."""

from typing import Union, List, Dict, Any
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import Sampler, PrimitiveJob
from qiskit_aer.primitives import Sampler as aer_Sampler
from qiskit_aer.primitives import SamplerV2 as aer_SamplerV2
from qiskit_ibm_runtime import Sampler as ibm_runtime_Sampler
from qiskit_ibm_runtime import SamplerV2 as ibm_runtime_SamplerV2
from qiskit_ibm_runtime import RuntimeJobV2

from .base_run_builder import BasePrimitiveRunBuilder

SamplerValidType = Union[
    Sampler,
    aer_Sampler,
    aer_SamplerV2,
    ibm_runtime_Sampler,
    ibm_runtime_SamplerV2,
]


class SamplerRunBuilder(BasePrimitiveRunBuilder):  # pylint: disable=abstract-method
    """
    A class to build and configure sampler runs based on their provenance and options.

    Attributes:
        sampler (SamplerValidType): The quantum sampler instance.
        circuits (List[QuantumCircuit]): List of quantum circuits.
        parameter_sets (List[List[float]]): List of parameter sets.
    """

    def __init__(
        self,
        sampler: SamplerValidType,
        circuits: List[QuantumCircuit],
        parameter_sets: List[List[float]],
        options: Dict[str, Any],
    ):
        super().__init__(sampler, circuits, parameter_sets, options)

    def _select_run_builder(self) -> Union[PrimitiveJob, RuntimeJobV2]:
        builders = {
            ("qiskit", "Sampler"): self._build_native_qiskit_run,
            ("qiskit_aer", "SamplerV2"): self._build_v2_run,
            ("qiskit_aer", "Sampler"): self._build_v1_run,
            ("qiskit_ibm_runtime", "SamplerV2"): self._build_v2_run,
            ("qiskit_ibm_runtime", "SamplerV1"): self._build_v1_run,
        }
        try:
            return builders[self.provenance]
        except KeyError as err:
            raise NotImplementedError(
                f"{self.__class__.__name__} not compatible with {self.provenance}."
            ) from err

    def _build_native_qiskit_run(self) -> PrimitiveJob:
        """Builds a run function for a standard qiskit Sampler."""
        return self.primitive.run(
            self.circuits,
            self.parameter_sets,
            shots=self.shots,
            seed=self.seed,
        )

    def _build_v2_run(self) -> Union[PrimitiveJob, RuntimeJobV2]:
        """Builds a run function for qiskit-aer and qiskit-ibm-runtime SamplerV2."""
        backend = self.primitive._backend  # pylint: disable=protected-access
        optimization_level = 1
        pm = generate_preset_pass_manager(optimization_level, backend)
        pubs = []
        for qc, param in zip(self.circuits, self.parameter_sets):
            isa_circuit = pm.run(qc)
            pubs.append((isa_circuit, param))
        return self.primitive.run(pubs)
