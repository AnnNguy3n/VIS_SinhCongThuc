import numpy as np
from numba import njit


@njit
def get_valid_op(struct, idx, start):
    valid_op = np.full(2, 0)
    valid_op[start-2:] = 1

    if idx // 2 <= struct[0,1] // 2:
        valid_op[1] = 0

    return np.where(valid_op == 1)[0] + 2
