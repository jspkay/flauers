import numpy as np

"""
Defined based on the spatial projection orientation and with det(T) > 0
"""

# c is the stationary variable
output_stationary = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 1]
    ]
)
output_stationary_eq = np.array(
    [
        [-1,  0, 0],
        [ 0, -1, 0],
        [ 1,  1, 1]
    ]
)

# a is the stationary variable
row_stationary = np.array(
    [
        [1, 0,  0],
        [0, 0, -1],
        [1, 1,  1]
    ]
)
row_stationary_eq = np.array(
    [
        [-1, 0, 0],
        [ 0, 0, 1],
        [ 1, 1, 1]
    ]
)

# b is the stationary variable
col_stationary = np.array(
    [
        [0,  0, 1],
        [0, -1, 0],
        [1,  1, 1]
    ]
)
col_stationary_eq = np.array(
    [
        [0, 0, -1],
        [0, 1,  0],
        [1, 1,  1]
    ]
)


# No stationary variable
no_local_reuse = np.array( # det is +3
    [
        [-1,  1,  0],
        [ 0, -1,  1],
        [ 1,  1,  1]
    ]
)

no_local_reuse_eq = np.array( # det is +3
    [
        [ 0,  1,  -1],
        [-1,  1,  0],
        [ 1,  1,  1]
    ]
)
