from . import utils
from . import lowerings
from . import tilings
from . import systolic_array
from . import projection_matrices
from . import fault_models
from . import exceptions

import numpy as np
import logging

SystolicArray = systolic_array.SystolicArray
ProjectionMatrices = projection_matrices

__version__ = "0.9.9"

logger = logging.getLogger(__name__)

def convolve_with_array(A: np.ndarray, B: np.ndarray,
                        array: SystolicArray,
                        lowering: lowerings.LoLif = lowerings.S_Im2Col,
                        tiling: bool = False,
                        ) -> np.ndarray:
    """
    Perform convolution between two matrices a and b using a systolic array, such that C = A * B

    Parameters
    ---
    A : input matrix
    B : filter on the input matrix
    array: SystolicArray object to be used for the convolution
    lowering : this object defines the lowering/lifting strategy to implement the convolution

    Returns
    ---
    O : systolic-array wise injected convolution
    """

    out_height, out_width = lowerings.lowered_indices.values()

    transformed = lowering(A.shape, B.shape)

    if use_gpu:
        low_A = transformed.lower_activation_cuda(A)
        low_B = transformed.lower_kernel_cuda(B)
        lowered_result = array.matmul_cuda(low_A, low_B, tiling = tiling)
        result = transformed.lift_cuda(lowered_result)
    else:
        low_A = transformed.lower_activation(A)
        low_B = transformed.lower_kernel(B)
        lowered_result = array.matmul(low_A, low_B, tiling = tiling)
        result = transformed.lift(lowered_result)

    return result


def convolve(A: np.ndarray, B: np.ndarray,
             # TODO: Use typing so that we can specify that lowering is an object from a SUBCLASS of LowLif, otherwise the type-check complains that we cannot instantiate with lowering
             lowering: lowerings.LoLif = lowerings.S_Im2Col,
             N1=-1,
             N2=-1,
             N3=-1,  # TODO: maybe projection_matrix can have its own class 🤷
             projection_matrix: np.ndarray = ProjectionMatrices.output_stationary,
             tiling: bool = False,
             **kwargs
        ) -> np.ndarray:
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

    logging.info(f"[convolve] the convolution function has started")

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
    out_height, out_width = lowerings.lowered_indices.values()
    if N1 == -1:
        N1 = transformed.lowered_activation_shape[ out_height ] + 1
    if N2 == -1:
        N2 = transformed.lowered_kernel_shape[ out_width ] + 1
    if N3 == -1:
        N3 = transformed.lowered_kernel_shape[ out_height ] + 1

    hw = SystolicArray(N1, N2, N3, projection_matrix, **kwargs)

    low_A = transformed.lower_activation(A)
    low_B = transformed.lower_kernel(B)
    result = hw.matmul(low_A, low_B, tiling=tiling)

    # return low_A @ low_B
    return transformed.lift(result)


def matmul(A, B,
           N1=-1,
           N2=-1,
           N3=-1,
           projection_matrix=ProjectionMatrices.output_stationary,
           tiling: bool = False,
           use_gpu: bool = False,
           **kwargs
           ) -> np.ndarray:
    if N1 == -1:
        N1 = A.shape[0]
    if N2 == -1:
        N2 = B.shape[1]
    if N3 == -1:
        N3 = B.shape[0]

    hw = SystolicArray(N1, N2, N3, projection_matrix, use_gpu=use_gpu, **kwargs)
    return hw.matmul(A, B, tiling=tiling)

