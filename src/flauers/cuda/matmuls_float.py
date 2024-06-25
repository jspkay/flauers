import numba
from numba import cuda
import os

@cuda.jit("float32(float32, int32, int8)", device=True, inline=True)
def inject_float32(value, bitstring, injection_type):
    # TODO implement this function
    res = value
    return res

@cuda.jit("float64(float64, int64, int8)", device=True, inline=True)
def inject_float64(value, bitstring, injection_type):
    # TODO implement this function
    res = value
    return res

@cuda.jit
def injected_matmul_old_f32(
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

                a_value = inject_float32( A[i, k], inject_A[i, j, k], injection_type )
                b_value = inject_float32( B[k, j], inject_B[i, j, k], injection_type )

                c_tmp += a_value*b_value

                c_tmp = inject_float32(c_tmp, inject_C[i, j, k], injection_type)
            C[i, j] = c_tmp


@cuda.jit(inline=True)
def injected_matmul_old_float64(
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

                a_value = inject_float64( A[i, k], inject_A[i, j, k], injection_type )
                b_value = inject_float64( B[k, j], inject_B[i, j, k], injection_type )

                c_tmp += a_value*b_value

                c_tmp = inject_float64(c_tmp, inject_C[i, j, k], injection_type)
            C[i, j] = c_tmp
