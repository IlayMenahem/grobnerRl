import numpy as np
from jaxtyping import ArrayLike

Ideal = list[np.ndarray] | list[ArrayLike]
SelectablePairs = list[tuple[int, int]]
Observation = tuple[Ideal, SelectablePairs]
Action = int
