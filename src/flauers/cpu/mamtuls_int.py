import numba
import numpy as np

from .injects import inject_int

###### Single Threaded #################################################

@numba.njit(nogil=True, cache=False, parallel=False)
def injected_matmul_old_int(A, B, C,
                        inject_A, inject_B, inject_C, injection_type):

    N1 = A.shape[0]
    N2 = B.shape[1]
    N3 = A.shape[1]

    for i in range(0, N1):
        for j in range(0, N2):
            c_tmp = 0
            for k in range(0, N3):

                a_value = inject_int(
                    A[i, k],
                    inject_A[i, j, k],
                    injection_type
                )

                b_value = inject_int(
                    B[k, j],
                    inject_B[i, j, k],
                    injection_type
                )

                c_tmp += a_value * b_value  # accumulation

                c_tmp = inject_int( # injection
                    c_tmp,
                    inject_C[i, j, k],
                    injection_type,
                )

            C[i, j] = c_tmp


###### Multi Threaded #################################################

@numba.njit(nogil=True, cache=False, parallel=True)
def injected_matmul_old_int_parallel(A, B, C,
                        inject_A, inject_B, inject_C, injection_type):

    N1 = A.shape[0]
    N2 = B.shape[1]
    N3 = A.shape[1]

    for i in numba.prange(0, N1):
        for j in numba.prange(0, N2):
            c_tmp = 0
            for k in range(0, N3):

                a_value = inject_int(
                    A[i, k],
                    inject_A[i, j, k],
                    injection_type
                )

                b_value = inject_int(
                    B[k, j],
                    inject_B[i, j, k],
                    injection_type
                )

                c_tmp += a_value * b_value  # accumulation

                c_tmp = inject_int( # injection
                    c_tmp,
                    inject_C[i, j, k],
                    injection_type,
                )

            C[i, j] = c_tmp