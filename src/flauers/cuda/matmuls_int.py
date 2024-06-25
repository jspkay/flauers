import numpy as np
import numba
from numba import cuda
import ctypes
import os

"""
inject_int8 = cuda.declare_device("inject_int8", "int8(int8, int8, int8)")
inject_int32 = cuda.declare_device("inject_int32", "int32(int32, int32, int8)")

basedir = os.path.dirname(os.path.abspath(__file__))
inject_file = os.path.join(basedir, "injects.cu")
print(inject_file) """

@cuda.jit("int8(int8, int8, int8)", device=True, inline=True)
def inject_int8(value, bitstring, injection_type):

    if injection_type == 0:
        res = value & ~bitstring
    elif injection_type == 1:
        res = value | bitstring
    elif injection_type == 2:
        res = value ^ bitstring
    else:
        res = bitstring

    return res

@cuda.jit("int32(int32, int32, int8)", device=True, inline=True)
def inject_int32(value, bitstring, injection_type):
    if injection_type == 0:
        res = value & ~bitstring
    elif injection_type == 1:
        res = value | bitstring
    elif injection_type == 2:
        res = value ^ bitstring
    else:
        res = bitstring

    return res

    
@cuda.jit
def injected_matmul_old_int32(
        A, B, C, 
        inject_A, inject_B, inject_C,
        injection_type
):
    start = cuda.grid(2)
    stride = cuda.gridsize(2)

    n1 = A.shape[0]
    n2 = A.shape[1]
    n3 = B.shape[1]

    for i in range(start[0], n1, stride[0]):
        for j in range(start[1], n2, stride[1]):
            c_tmp = 0
            for k in range(n3):

                a_value = inject_int8( A[i, k], inject_A[i, j, k], injection_type)
                b_value = inject_int8( B[k, j], inject_B[i, j, k], injection_type)

                c_tmp += a_value*b_value

                c_tmp = inject_int32(c_tmp, inject_C[i, j, k], injection_type)

            C[i, j] += c_tmp