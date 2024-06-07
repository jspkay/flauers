import numba
import numpy as np
from numba import void, float64, float32, int64, int32, int8, prange

from .injects import inject_32, inject_64

########### Single threads #############################################

@numba.njit(void(float32[:,:], float32[:,:], float32[:,:],
                         int32[:,:,:], int32[:,:,:], int32[:,:,:], int8),
            nogil=True, cache=False, parallel=False)
def injected_matmul_old_f32(A, B, C,
                        inject_A, inject_B, inject_C, injection_type):

    N1 = A.shape[0] + 1
    N2 = B.shape[1] + 1
    N3 = A.shape[1] + 1

    for i in range(1, N1):
        for j in range(1, N2):
            c_tmp = np.float32(0)
            for k in range(1, N3):

                a_value = A.view(np.int32)[i-1, k-1]
                a_value = inject_32(
                    a_value,
                    inject_A[i, j, k],
                    injection_type
                )
                a_value = np.array([a_value], dtype=np.int32).view(np.float32)[0]

                b_value = B.view(np.int32)[k-1, j-1]
                b_value = inject_32( # injection
                    b_value,
                    inject_B[i, j, k],
                    injection_type
                )
                b_value = np.array([b_value], dtype=np.int32).view(np.float32)[0]

                # print(c_tmp)
                c_tmp += a_value * b_value  # accumulation
                # print("+", a_value, "*", b_value, "=", c_tmp)

                c_tmp = np.array([c_tmp]).view(np.int32)[0]
                c_tmp = inject_32( # injection
                    c_tmp,
                    inject_C[i, j, k],
                    injection_type,
                )
                c_tmp = np.array([c_tmp], dtype=np.int32).view(np.float32)[0]
                # print("ctmp after inj: ", c_tmp)

            C[i-1, j-1] = c_tmp


@numba.njit(void(float64[:,:], float64[:,:], float64[:,:],
                         int64[:,:,:], int64[:,:,:], int64[:,:,:], int8),
            nogil=True, cache=False, parallel=False)
def injected_matmul_old_f64(A, B, C,
                        inject_A, inject_B, inject_C, injection_type):

    N1 = A.shape[0] + 1
    N2 = B.shape[1] + 1
    N3 = A.shape[1] + 1

    for i in range(1, N1): # prange ?
        for j in range(1, N2): # prange ?
            c_tmp = np.float64(0)
            for k in range(1, N3):

                a_value = A.view(np.int64)[i-1, k-1]
                a_value = inject_64(
                    a_value,
                    inject_A[i, j, k],
                    injection_type
                )
                a_value = np.array([a_value], dtype=np.int64).view(np.float64)[0]

                b_value = B.view(np.int64)[k-1, j-1]
                b_value = inject_64( # injection
                    b_value,
                    inject_B[i, j, k],
                    injection_type
                )
                b_value = np.array([b_value], dtype=np.int64).view(np.float64)[0]

                # print(c_tmp)
                c_tmp += a_value * b_value  # accumulation
                # print("+", a_value, "*", b_value, "=", c_tmp)

                c_tmp = np.array([c_tmp]).view(np.int64)[0]
                c_tmp = inject_64( # injection
                    c_tmp,
                    inject_C[i, j, k],
                    injection_type,
                )
                c_tmp = np.array([c_tmp], dtype=np.int64).view(np.float64)[0]
                # print("ctmp after inj: ", c_tmp)

            C[i-1, j-1] = c_tmp


########### Multiple threads #############################################

@numba.njit(void(float32[:,:], float32[:,:], float32[:,:],
                         int32[:,:,:], int32[:,:,:], int32[:,:,:], int8),
            nogil=True, cache=False, parallel=True)
def injected_matmul_old_f32_parallel(A, B, C,
                        inject_A, inject_B, inject_C, injection_type):

    N1 = A.shape[0] + 1
    N2 = B.shape[1] + 1
    N3 = A.shape[1] + 1

    for i in prange(1, N1):
        for j in prange(1, N2):
            c_tmp = np.float32(0)
            for k in range(1, N3):

                a_value = A.view(np.int32)[i-1, k-1]
                a_value = inject_32(
                    a_value,
                    inject_A[i, j, k],
                    injection_type
                )
                a_value = np.array([a_value], dtype=np.int32).view(np.float32)[0]

                b_value = B.view(np.int32)[k-1, j-1]
                b_value = inject_32( # injection
                    b_value,
                    inject_B[i, j, k],
                    injection_type
                )
                b_value = np.array([b_value], dtype=np.int32).view(np.float32)[0]

                # print(c_tmp)
                c_tmp += a_value * b_value  # accumulation
                # print("+", a_value, "*", b_value, "=", c_tmp)

                c_tmp = np.array([c_tmp]).view(np.int32)[0]
                c_tmp = inject_32( # injection
                    c_tmp,
                    inject_C[i, j, k],
                    injection_type,
                )
                c_tmp = np.array([c_tmp], dtype=np.int32).view(np.float32)[0]
                # print("ctmp after inj: ", c_tmp)

            C[i-1, j-1] = c_tmp


@numba.njit(void(float64[:,:], float64[:,:], float64[:,:],
                         int64[:,:,:], int64[:,:,:], int64[:,:,:], int8),
            nogil=True, cache=False, parallel=True)
def injected_matmul_old_f64_parallel(A, B, C,
                        inject_A, inject_B, inject_C, injection_type):

    N1 = A.shape[0] + 1
    N2 = B.shape[1] + 1
    N3 = A.shape[1] + 1

    for i in prange(1, N1):
        for j in prange(1, N2): 
            c_tmp = np.float64(0)
            for k in range(1, N3):

                a_value = A.view(np.int64)[i-1, k-1]
                a_value = inject_64(
                    a_value,
                    inject_A[i, j, k],
                    injection_type
                )
                a_value = np.array([a_value], dtype=np.int64).view(np.float64)[0]

                b_value = B.view(np.int64)[k-1, j-1]
                b_value = inject_64( # injection
                    b_value,
                    inject_B[i, j, k],
                    injection_type
                )
                b_value = np.array([b_value], dtype=np.int64).view(np.float64)[0]

                # print(c_tmp)
                c_tmp += a_value * b_value  # accumulation
                # print("+", a_value, "*", b_value, "=", c_tmp)

                c_tmp = np.array([c_tmp]).view(np.int64)[0]
                c_tmp = inject_64( # injection
                    c_tmp,
                    inject_C[i, j, k],
                    injection_type,
                )
                c_tmp = np.array([c_tmp], dtype=np.int64).view(np.float64)[0]
                # print("ctmp after inj: ", c_tmp)

            C[i-1, j-1] = c_tmp