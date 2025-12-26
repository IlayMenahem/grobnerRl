import numpy as np
from jaxtyping import Array

Ideal = list[np.ndarray] | list[Array]
SelectablePairs = list[tuple[int, int]]
Observation = tuple[Ideal, SelectablePairs]
Action = int
