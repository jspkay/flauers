import numpy as np

output_stationary = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 1]
    ]
)

row_stationary = np.array(
    [
        [1, 0,  0],
        [0,  0, 1],
        [1,  1, 1]
    ]
)

col_stationary = np.array(
    [
        [0, 0,  1],
        [1,  0, 0],
        [1,  1, 1]
    ]
)


no_local_reuse = np.array(
    [
        [-1,  1,  0],
        [ 0, -1,  1],
        [ 1,  1,  1]
    ]
)

no_local_reuse_equivalent = np.array(
    [
        [ 0, -1,  1],
        [-1,  1,  0],
        [ 1,  1,  1]
    ]
)