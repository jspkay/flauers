import numba
from numba import cuda
import os

inject_float32 = cuda.declare_device("inject_float32", "float32(float32, int32, int8)")

basedir = os.path.dirname(os.path.abspath(__file__))
c_injects_file = os.path.join(basedir, "injects.cu")

@cuda.jit(
    "void(float32[:,:], float32[:,:], float32[:,:],"
            "int32[:,:,:], int32[:,:,:], int32[:,:,:],"
            "int8, boolean )",
    link=[c_injects_file], 
    inline=True)
def injected_matmul_old_float32(
        A, B, C, 
        inject_A, inject_B, inject_C,
        injection_type, additive
):
    start = cuda.grid(2)
    stride = cuda.gridsize(2)

    # thread_id_y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    # thread_id_x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    n1 = A.shape[0]
    n2 = B.shape[1]
    krange = A.shape[1]

    for i in range(start[0], n1, stride[0]):
        for j in range(start[1], n2, stride[1]):
            c_tmp = 0
            for k in range(krange):
                        
                a_value = inject_float32( A[i, k], inject_A[i, j, k], injection_type )
                b_value = inject_float32( B[k, j], inject_B[i, j, k], injection_type )

                c_tmp += a_value*b_value
                # print( "(", thread_id_x, ", ", thread_id_y, "):", a_value, b_value, c_tmp)

                c_tmp = inject_float32(c_tmp, inject_C[i, j, k], injection_type)
            if additive:
                # print( "last_add -> (", thread_id_x, ", ", thread_id_y, "):", C[i, j], c_tmp, C[i, j] + c_tmp)
                C[i, j] += c_tmp
            else:
                C[i, j]  = c_tmp


@cuda.jit(link=[c_injects_file], inline=True)
def injected_matmul_old_float64(
        A, B, C, 
        inject_A, inject_B, inject_C,
        injection_type, additive
):
    start = cuda.grid(2)
    stride = cuda.gridsize(2)

    n1 = A.shape[0]
    n2 = B.shape[1]
    krange = A.shape[1]

    for i in range(start[0], n1, stride[0]):
        for j in range(start[1], n2, stride[1]):
            c_tmp = 0
            for k in range(krange):

                a_value = inject_float64( A[i, k], inject_A[i, j, k], injection_type )
                b_value = inject_float64( B[k, j], inject_B[i, j, k], injection_type )

                c_tmp += a_value*b_value

                c_tmp = inject_float64(c_tmp, inject_C[i, j, k], injection_type)
            if additive:
                C[i, j] += c_tmp
            else:
                C[i, j]  = c_tmp