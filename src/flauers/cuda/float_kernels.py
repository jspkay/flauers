import numpy as np
import numba
from numba import cuda
import ctypes
import os

bitflip_float32 = cuda.declare_device("bitflip_float32", "float32(float32, int32)")
sa0_float32     = cuda.declare_device("sa0_float32", "float32(float32, int32)")
sa1_float32     = cuda.declare_device("sa1_float32", "float32(float32, int32)")

basedir = os.path.dirname(os.path.abspath(__file__))
inject_file = os.path.join(basedir, "inject_float.cu")

@cuda.jit(link=[inject_file])
def matmul_float32_bitflip(
        A: np.array,
        B: np.array,
        C: np.array, 
        INJ_A: np.array,
        INJ_B: np.array,
        INJ_C: np.array,
):
    start = cuda.grid(3)
    stride = cuda.gridsize(3)

    n1 = A.shape[0]
    n2 = A.shape[1]
    n3 = B.shape[1]

    for i in range(start[0], n1, stride[0]):
        for j in range(start[1], n2, stride[1]):
            for k in range(start[2], n3, stride[2]):

                a_value = bitflip_float32( A[i, k], INJ_A[i, j, k] )
                b_value = bitflip_float32( B[i, k], INJ_B[i, j, k] )
                c_value = bitflip_float32(a_value*b_value, INJ_C[i, j, k])

                cuda.atomic.add(C, (i, j), c_value)
