import numpy as np
from enum import Enum

"""
Definition of some utils functions
"""

# ########### Constants and types

# indexes for the shapes depending on the LowLif type
input_indexes_mapping = [
    # For more information on these indexes, check the class LowLif
    {  # Single LowLif Type
        "HEIGHT": 0,
        "WIDTH": 1,
    },
    {  # Channeled LowLif Type
        "CHANNEL": 0,
        "HEIGHT": 1,
        "WIDTH": 2,
    },
    {  # Parallel LowLif Type
        "BATCH": 0,
        "CHANNEL": 1,
        "HEIGHT": 2,
        "WIDTH": 3,
    },
]

# LineType = Enum("LineType", ["a", "b", "c"])
LineType = Enum("LineType", ["b", "a", "c"])

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
            eps -> space-time vector containing [x, y, t]
            T -> space-time projection matrix
        Returns:
            nu -> 3D iteration vector composed by [i, j, k]
    """

    t_inv = np.linalg.inv(t)
    nu = t_inv @ eps
    return nu


def is_comprised(
        nu: list[int, int, int],
        nu_start: list[int, int, int],
        nu_stop: list[int, int, int]) -> bool:
    """
    Checks whether nu is comprised between nu_start and nu_stop, i.e. for each component i the following
    relationship holds:

    nu_start(i) <= nu(i) <= nu_stop(i)

        Args:
            nu -> iteration vector to check
            nu_start -> lower bound (included)
            nu_stop -> upper bound (included)

        Returns:
            o -> it is True when nu_start <= nu <= nu_stop, otherwise False
    """

    for i in range(len(nu)):
        if nu[i] < nu_start[i] or nu[i] > nu_stop[i]:
            return False
    return True


def check_improvement(self, a: np.ndarray, b: np.ndarray, improving_direction: np.ndarray):
    """
    This function returns true if b is greater than a in direction improving_direcition. This means that b is better
    than a if b-a is parallel to improving_direction AND the distance of b to the origin is greater than the
    distance of a to the origin.
    Obviously, a and b be should be parallel to improving_direcion:
        a = alpha * improving_direction, for some scalar alpha
        b = beta * improving_direction, for some scalar beta
    a and b are both parallel to improving_direction iff b-a is parallel to improving_direction. (The proof is left
    to the reader as an exercise).

    Parameters
    ---
    a -> best for now
    b -> potential improvement over a
    improving_direction -> direction in which we seek the improvment

    Returns
    ---
    res -> true if b is better than a in improving_direction, false otherwise
    """

    alpha = np.zeros((2, 1))
    d = a - b  # we compute b-a (or a-b, so we can check for a positive sign instead of a negative one, later on)
    for i in range(2):
        alpha[i] = d[i] / improving_direction[i]

    # d is parallel to improving direction if d = alpha * improving_direction for some scalar alpha. This means
    # that exist 1/alpha such that d * 1/alpha = improving_direction. alpha exists if we can divide d and
    # improving_direction component-wise and the result have the same values for both components (i.e. the division
    # actually corresponds to a scalar :D
    if alpha[0] == alpha[1] and alpha[0] > 0:
        return True

    return False
