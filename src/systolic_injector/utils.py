import numpy as np

"""
Definition of some utils functions
"""

# ###########Variables and constants

# indexes for the shapes
input_indexes = {
    "CHANNELS": 0,
    "HEIGHT": 1,
    "WIDTH": 2,
}


# ########Functions
def print_matrix_in_index(mat, index):
    """
    This function prints a 3D tensor, iterating over the dimension specified by index
    """

    assert len(mat.shape) == 3

    a, b, c = mat.shape

    if index == 0:
        for i in range(a):
            print(mat[i, :, :])
    elif index == 1:
        for i in range(b):
            print(mat[:, i, :])
    elif index == 2:
        for i in range(c):
            print(mat[:, :, i])
    else:
        assert False, "index cannot have value " + str(index)


def space_time_equation(nu: np.ndarray, t: np.ndarray):
    """
        Args:
            nu -> 3D iteration vector
            T -> space-time projection matrix
        Returns:
            eps -> a vector composed by [x, y, t] describing the position in space (x,y) and in time (t)
            of the operation corresponding to the given iteration vector
    """

    return t @ nu


def inverse_space_time_equation(eps: np.ndarray, t: np.ndarray):
    """
        Args:
            eps -> space-time vector containing [x,y,t]
            T -> space-time projection matrix
        Returns:
            nu -> 3D iteration vector composed by [i, j, k]
    """

    t_inv = np.linalg.inv(t)
    nu = t_inv @ eps
    return nu
