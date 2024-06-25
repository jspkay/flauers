from numba import cuda
if cuda.is_available():
    from . import matmuls_int, matmuls_float
    from . import utils

import numpy as np
import numba
from numba import cuda
import ctypes
import os

bitflip_float32 = cuda.declare_device("bitflip_float32", "float32(float32, int32)")
sa0_float32     = cuda.declare_device("sa0_float32", "float32(float32, int32)")
sa1_float32     = cuda.declare_device("sa1_float32", "float32(float32, int32)")

basedir = os.path.dirname(os.path.abspath(__file__))
inject_file = os.path.join(basedir, "inject.cu")

@cuda.jit(link=[inject_file])
def injected_matmul(
        A: np.array,
        B: np.array,
        C: np.array, 
        INJ: np.array,
):
    start = cuda.grid(3)
    stride = cuda.gridsize(3)

    n1 = A.shape[0]
    n2 = A.shape[1]
    n3 = B.shape[1]

    for i in range(start[0], n1, stride[0]):
        for j in range(start[1], n2, stride[1]):
            for k in range(start[2], n3, stride[2]):

                # a_value = ctypes.c_float(A[i, k])
                a_value = A[i, k]
                if INJ[i, j, k] == 0x011:
                    """ pointer = a_value.ctypes.data_as( ctypes.POINTER(ctypes.c_float) )
                    a_value_n = numba.carray(pointer, (1), dtype=np.int32)
                    a_value_n = a_value_n ^ (0x1<<5) """
                    a_value = inject(a_value)
                    # a_value = a_value_n

                b_value = B[k, j]
                if INJ[i, j, k] == 0x01:
                    print("I got in")
                    b_value = 0

                c_value = a_value * b_value

                if INJ[i, j, k] == 0x101:
                    c_value = 0

                cuda.atomic.add(C, (i, j), c_value)
