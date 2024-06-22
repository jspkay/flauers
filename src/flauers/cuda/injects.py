import numpy as np
from numba import cuda
from numba import void, int64, int32, int16, int8

# General version for integers
# Note that the dtype is different for the accumulation and for the inputs
# so jit should generate a different version for each (in_dtype, mac_dtype) pair
@cuda.jit
def inject_int(value, bitstring, type):
    
    res = bitstring # bizantine neuron

    if type == 0:  # stuck-at 0
        res = value & ~bitstring
    if type == 1:  # stuck-at 1
        res = value | bitstring
    if type == 2:  # bit-flip
        res = value ^ bitstring

    value = res

# Explicit versions for floats
@cuda.jit( void(int32, int32, int8))
def inject_32(value: np.int32, bitstring: np.int32, type: np.int8):
    
    res = bitstring # bizantine neuron
    	
    if type == 0:  # stuck-at 0
        res = value & ~bitstring
    if type == 1:  # stuck-at 1
        res = value | bitstring
    if type == 2:  # bit-flip
        res = value ^ bitstring

    value = res

@cuda.jit( void(int64, int64, int8))
def inject_64(value: np.int64, bitstring: np.int64, type: np.int8):

    res = bitstring # bizantine neuron

    if type == 0:  # stuck-at 0
        res = value & ~bitstring
    if type == 1:  # stuck-at 1
        res = value | bitstring
    if type == 2:  # bit-flip
        res = value ^ bitstring

    # bizantine neuron
    value = res
