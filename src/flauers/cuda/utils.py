from numba import cuda

@cuda.jit
def zero_init_matrix(matrix):
    start = cuda.grid(2)
    stride = cuda.gridsize(2)

    n1, n2 = matrix.shape

    for i in range(start[0], n1, stride[0]):
        for j in range(start[1], n2, stride[1]):
            matrix[i, j] = 0

@cuda.jit
def zero_init_3dtensor(matrix):
    start = cuda.grid(3)
    stride = cuda.gridsize(3)

    n1, n2, n3 = matrix.shape

    for i in range(start[0], n1, stride[0]):
        for j in range(start[1], n2, stride[1]):
            for k in range(start[2], n3, stride[2]):
                matrix[i, j, k] = 0