"""This module defines a base class for primitive run builders."""

from typing import Union, List, Tuple, Dict, Any
from qiskit import QuantumCircuit
from qiskit.primitives import PrimitiveJob
from qiskit_ibm_runtime import RuntimeJobV2


class BasePrimitiveRunBuilder:
    """
    Base class for building and configuring primitive runs based on their provenance and options.
    """

    def __init__(
        self,
        primitive,
        circuits: List[QuantumCircuit],
        parameter_sets: List[List[float]],
        options: Dict[str, Any],
    ):
        """
        Initializes BasePrimitiveRunBuilder for given primitive, circuits, parameters, and options.

        Args:
            primitive (Union[SamplerValidType, EstimatorValidType]): The primitive to use for runs.
            circuits (List[QuantumCircuit]): The quantum circuits to run.
            parameter_sets (List[List[float]]): The parameters to vary in the circuits.
            options (Dict[str, Any]): Configuration options such as number of shots.
        """
        self.primitive = primitive
        self.circuits = circuits
        self.parameter_sets = parameter_sets
        self.shots = options.pop("shots", None)
        self.seed = options.pop("seed", None)
        self.provenance = self.find_provenance()

    def find_provenance(self) -> Tuple[str, str]:
        """Determines the provenance of the primitive based on its class and module."""
        return (
            self.primitive.__class__.__module__.split(".")[0],
            self.primitive.__class__.__name__,
        )

    def build_run(self) -> Union[PrimitiveJob, RuntimeJobV2]:
        """
        Configures and returns primitive runs based on its provenance.

        Raises:
            NotImplementedError: If the primitive's provenance is not supported.

        Returns:
            Union[PrimitiveJob, RuntimeJobV2]: A primitive job.
        """
        primitive_job = self._select_run_builder()
        return primitive_job()

    def _select_run_builder(self) -> Union[PrimitiveJob, RuntimeJobV2]:
        """Selects the appropriate builder function based on the primitive's provenance."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _build_native_qiskit_run(self) -> PrimitiveJob:
        """Builds a run function for a standard qiskit primitive."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _build_v2_run(self) -> Union[PrimitiveJob, RuntimeJobV2]:
        """Builds a run function for qiskit-aer and qiskit-ibm-runtime V2 primitives."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _build_v1_run(self):
        """
        Attempts to build a run function for primitives V1, which will be soon deprecated.

        Raises:
            NotImplementedError: Indicates that V1 will be soon deprecated.
        """
        raise NotImplementedError(
            "Primitives V1 will be soon deprecated. Please, use V2 implementation."
        )
