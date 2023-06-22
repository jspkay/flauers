import numpy as np
from . import utils
from . import lowerings
from . import systolic_array
from . import projection_matrices

SystolicArray = systolic_array.SystolicArray
ProjectionMatrices = projection_matrices


def convolve(A: np.ndarray, B: np.ndarray, history: list = [],
             lowering: lowerings.LoweringLifting = lowerings.SlimKernel,
             N1=-1,
             N2=-1,
             N3=-1,  # TODO: maybe projection_matrix can have its own class 🤷
             projection_matrix: np.ndarray = ProjectionMatrices.output_stationary) -> np.ndarray:
    """
    Perform convolution between two matrices a and b using a systolic array, such that C = A * B

    Parameters
    ---
    A : input matrix
    B : filter on the input matrix
    history : this parameter is used to report the iterations over i, j and k of the multiplication
    lowering : this object defines the lowering/lifting strategy to implement the convolution
    N1 : rows of the first matrix A (corresponding to the rows of C)
    N2 : columns of the matrix B (corresponding to the columns of C)
    N3 : columns of the matrix A (corresponding to the rows of the matrix B)
    projection_matrix

    Returns
    ---
    O : systolic-array wise injected convolution
    """

    assert A.shape[1] == A.shape[2], "For now, only square matrices are available!"
    assert B.shape[1] == B.shape[2], "For now, only square matrices are available!"

    n = A.shape[1]
    m = B.shape[1]
    assert m < n, "Matrix b must be smaller than matrix a, invert the arguments!"

    L = m * m + m
    print("L is ", L)

    transformed = lowering(A, B)

    if N1 == -1:
        N1 = transformed.activation_shape[0] + 1
    if N2 == -1:
        N2 = transformed.kernel_shape[1] + 1
    if N3 == -1:
        N3 = transformed.kernel_shape[0] + 1

    hw = SystolicArray(N1, N2, N3, projection_matrix)
    result = hw.matmul(transformed.get_activation(), transformed.get_kernel(), history)

    return transformed.lift(result)

def matmul(A, B, history: list = [],
           N1=-1,
           N2=-1,
           N3=-1,
           projection_matrix = ProjectionMatrices.output_stationary) -> np.ndarray:

    if N1 == -1:
        N1 = A.shape[0] + 1
    if N2 == -1:
        N2 = B.shape[1] + 1
    if N3 == -1:
        N3 = B.shape[0] + 1

    print(N1)
    print(N2)
    print(N3)

    hw = SystolicArray(N1, N2, N3, projection_matrix)
    return hw.matmul(A, B, history)