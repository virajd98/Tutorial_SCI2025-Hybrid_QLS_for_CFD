from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class VQLSLog:
    values: List
    parameters: List

    def update(  # pylint: disable=unused-argument
        self, count: int, cost: float, parameters: np.ndarray
    ) -> None:
        self.values.append(cost)
        self.parameters.append(parameters)
