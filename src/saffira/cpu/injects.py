import numpy as np
from numba import njit, int64, int32, int16, int8

# General version for integers
@njit(nogil=True, cache=True)
def inject_int(value, bitstring, type):
    if type == 0:  # stuck-at 0
        return value & ~bitstring
    if type == 1:  # stuck-at 1
        return value | bitstring
    if type == 2:  # bit-flip
        return value ^ bitstring

    # bizantine neuron
    return bitstring

# Explicit versions for floats
@njit( int32(int32, int32, int8), nogil=True, cache=True)
def inject_32(value: np.int32, bitstring: np.int32, type: np.int8):
    if type == 0:  # stuck-at 0
        return value & ~bitstring
    if type == 1:  # stuck-at 1
        return value | bitstring
    if type == 2:  # bit-flip
        return value ^ bitstring

    # bizantine neuron
    return bitstring

@njit( int64(int64, int64, int8), nogil=True, cache=True)
def inject_64(value: np.int64, bitstring: np.int64, type: np.int8):
    if type == 0:  # stuck-at 0
        return value & ~bitstring
    if type == 1:  # stuck-at 1
        return value | bitstring
    if type == 2:  # bit-flip
        return value ^ bitstring

    # bizantine neuron
    return bitstring