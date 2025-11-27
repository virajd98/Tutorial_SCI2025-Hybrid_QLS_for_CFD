from __future__ import annotations

from collections.abc import Sequence
from typing import Union, Any
import numpy as np
from qiskit.circuit import QuantumCircuit


def validate_initial_point(
    point: Union[Sequence[float], np.ndarray] | None, circuit: QuantumCircuit
) -> Union[Sequence[float], Any, np.ndarray[Any, Any]]:
    r"""
    Validate a choice of initial point against a choice of circuit. If no point is provided, a
    random point will be generated within certain parameter bounds. It will first look to the
    circuit for these bounds. If the circuit does not specify bounds, bounds of :math:`-2\pi`,
    :math:`2\pi` will be used.

    Args:
        point: An initial point.
        circuit: A parameterized quantum circuit.

    Returns:
        A validated initial point.

    Raises:
        ValueError: If the dimension of the initial point does not match the number of circuit
        parameters.
    """
    expected_size = circuit.num_parameters

    if point is None:
        # get bounds if circuit has them set, otherwise use [-2pi, 2pi] for each parameter
        bounds = getattr(circuit, "parameter_bounds", None)
        if bounds is None:
            bounds = [(-2 * np.pi, 2 * np.pi)] * expected_size

        # replace all Nones by [-2pi, 2pi]
        lower_bounds = []
        upper_bounds = []
        for lower, upper in bounds:
            lower_bounds.append(lower if lower is not None else -2 * np.pi)
            upper_bounds.append(upper if upper is not None else 2 * np.pi)

        # sample from within bounds
        point = np.random.uniform(low=lower_bounds, high=upper_bounds).tolist()

    elif len(point) != expected_size:
        raise ValueError(
            f"The dimension of the initial point ({len(point)}) does not match the "
            f"number of parameters in the circuit ({expected_size})."
        )

    return point
