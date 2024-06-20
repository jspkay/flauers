import numpy as np
from numba import njit, int64, int32, int16, int8

# General version for integers
# Note that the dtype is different for the accumulation and for the inputs
# so jit should generate a different version for each (in_dtype, mac_dtype) pair
@njit(nogil=True, cache=False)
def inject_int(value, bitstring, type):
    
    res = bitstring # bizantine neuron

    if type == 0:  # stuck-at 0
        res = value & ~bitstring
    if type == 1:  # stuck-at 1
        res = value | bitstring
    if type == 2:  # bit-flip
        res = value ^ bitstring

    return res

# Explicit versions for floats
@njit( int32(int32, int32, int8), nogil=True, cache=False)
def inject_32(value: np.int32, bitstring: np.int32, type: np.int8):
    if type == 0:  # stuck-at 0
        return value & ~bitstring
    if type == 1:  # stuck-at 1
        return value | bitstring
    if type == 2:  # bit-flip
        return value ^ bitstring

    # bizantine neuron
    return bitstring

@njit( int64(int64, int64, int8), nogil=True, cache=False)
def inject_64(value: np.int64, bitstring: np.int64, type: np.int8):
    if bitstring != 0:
        print("injecting value ", value)

    if type == 0:  # stuck-at 0
        return value & ~bitstring
    if type == 1:  # stuck-at 1
        return value | bitstring
    if type == 2:  # bit-flip
        return value ^ bitstring

    if bitstring != 0:
        print("new value is ", value)

    # bizantine neuron
    return bitstring