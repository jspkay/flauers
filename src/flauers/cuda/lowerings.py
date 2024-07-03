import numba
from numba import cuda
import os
import numpy

@cuda.jit
def S_Im2Col_lower_activation(result, activation, kernel_size, additive):
    start = cuda.grid(2)
    stride = cuda.gridsize(2)

    for r in range(start[0], result.shape[0], stride[0]):
        for c in range(start[1], result.shape[1], stride[1]):
            kr = c % kernel_size
            kc = c // kernel_size
            if additive:
                result[r, c] += activation[r+kr, kc]
            else:
                result[r, c]  = activation[r+kr, kc]

@cuda.jit
def S_Im2Col_lower_kernel(result, kernel, kernel_size, additive):
    start = cuda.grid(2)
    stride = cuda.gridsize(2)
    
    if additive: 
        for col in range(start[0], result.shape[1], stride[0]): # for each column
            for m in range(start[1], kernel_size**2, stride[1]):
                s = col * kernel_size + m
                kr = m % kernel_size
                kc = m // kernel_size
                result[s, col] += kernel[kr, kc]
    else:
        for col in range(start[0], result.shape[1], stride[0]): # for each column
            s_start = col*kernel_size
            s_stop = s_start + kernel_size**2
            for row in range(start[1], result.shape[0], stride[1]):
                if row >= s_start and row < s_stop:
                    m = row - s_start
                    kr = m % kernel_size
                    kc = m // kernel_size
                    result[row, col] = kernel[kr, kc]
                else:
                    result[row, col] = 0

@cuda.jit
def S_Im2Col_lift(result, lowered, additive):
    start = cuda.grid(2)
    stride = cuda.gridsize(2)
    
    for row in range(start[0], lowered.shape[0], stride[0]):
        for col in range(start[1], lowered.shape[1], stride[1]):
            if additive:
                result[row, col] += lowered[row, col]
            else:
                result[row, col]  = lowered[row, col]