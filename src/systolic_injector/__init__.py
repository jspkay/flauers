import numpy as np
from . import utils
from . import lowerings
from . import systolic_array
from . import projection_matrices

SystolicArray = systolic_array.SystolicArray
ProjectionMatrices = projection_matrices


def convolve(A: np.ndarray, B: np.ndarray, history: list = [],
             # TODO: Use typing so that we can specify that lowering is an object from a SUBCLASS of LowLif, otherwise the type-check complains that we cannot instantiate with lowering
             lowering: lowerings.LowLif = lowerings.ExpensiveLowering,
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

    HEIGHT = utils.input_indexes["HEIGHT"]
    WIDTH = utils.input_indexes["WIDTH"]
    CHANNELS = utils.input_indexes["CHANNELS"]

    assert A.shape[HEIGHT] == A.shape[WIDTH], "For now, only square matrices are available!"
    assert B.shape[HEIGHT] == B.shape[WIDTH], "For now, only square matrices are available!"

    n = A.shape[HEIGHT]
    m = B.shape[HEIGHT]
    assert m < n, "Matrix b must be smaller than matrix a, invert the arguments!"

    L = m * m + m
    print("L is ", L)

    transformed = lowering(A.shape, B.shape)

    """
    Very important NOTE: N1, N2 and N3 are parameters for the instantiation of the systolic array design. These params
    correspond to the sizes of the matrices to multiply. Specifically, assuming C = A x B:
        - N1 is the number of ROWS of matrix A (which will be the rows of matrix C),
        - N2 is the number of COLUMNS of matrix B (which will be the columns of matrix C),
        - N3 is the number of COLUMNS of matrix A and the number of ROWS of matrix B.
    That is because we are gonna use the systolic array to perform matrix multiplication! 
    Another minor thing is: the shape of the transformed lowerings are bi-dimensional, since we are doing matrix
    multiplication, so we use the indexes defined in lowerings.py instead of the indexes defined in utils
    """
    out_height, out_width = lowerings.lowering_indexes.values()
    if N1 == -1:
        N1 = transformed.lowered_activation_shape[ out_height ] + 1
    if N2 == -1:
        N2 = transformed.lowered_kernel_shape[ out_width ] + 1
    if N3 == -1:
        N3 = transformed.lowered_kernel_shape[ out_height ] + 1

    hw = SystolicArray(N1, N2, N3, projection_matrix)
    result = hw.matmul(transformed.lower_activation(A), transformed.lower_kernel(B), history)

    return transformed.lift(result)


def matmul(A, B, history: list = [],
           N1=-1,
           N2=-1,
           N3=-1,
           projection_matrix=ProjectionMatrices.output_stationary) -> np.ndarray:
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
