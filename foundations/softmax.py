import numpy as np
from numpy.typing import NDArray

class Solution:
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Subtract max(z) for numerical stability before computing exp
        exp_z = np.exp(z - np.max(z))
        
        # Divide by the sum of exponentials and round to 4 decimal places
        return np.round(exp_z / np.sum(exp_z), 4)