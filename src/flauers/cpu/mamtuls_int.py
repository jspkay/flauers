import numba
import numpy as np

from .injects import inject_int

###### Single Threaded #################################################

@numba.njit(nogil=True, cache=False, parallel=False)
def injected_matmul_old_int(A, B, C,
                        inject_A, inject_B, inject_C, injection_type):

    N1 = A.shape[0] + 1
    N2 = B.shape[1] + 1
    N3 = A.shape[1] + 1

    for i in range(1, N1):
        for j in range(1, N2):
            c_tmp = 0
            for k in range(1, N3):

                a_value = inject_int(
                    A[i-1, k-1],
                    inject_A[i, j, k],
                    injection_type
                )

                b_value = inject_int(
                    B[k-1, j-1],
                    inject_B[i, j, k],
                    injection_type
                )

                c_tmp += a_value * b_value  # accumulation

                c_tmp = inject_int( # injection
                    c_tmp,
                    inject_C[i, j, k],
                    injection_type,
                )

            C[i-1, j-1] = c_tmp


###### Single Threaded #################################################

@numba.njit(nogil=True, cache=False, parallel=True)
def injected_matmul_old_int_parallel(A, B, C,
                        inject_A, inject_B, inject_C, injection_type):

    N1 = A.shape[0] + 1
    N2 = B.shape[1] + 1
    N3 = A.shape[1] + 1

    for i in numba.prange(1, N1):
        for j in numba.prange(1, N2):
            c_tmp = 0
            for k in range(1, N3):

                a_value = inject_int(
                    A[i-1, k-1],
                    inject_A[i, j, k],
                    injection_type
                )

                b_value = inject_int(
                    B[k-1, j-1],
                    inject_B[i, j, k],
                    injection_type
                )

                c_tmp += a_value * b_value  # accumulation

                c_tmp = inject_int( # injection
                    c_tmp,
                    inject_C[i, j, k],
                    injection_type,
                )

            C[i-1, j-1] = c_tmp