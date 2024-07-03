from numba import cuda
if cuda.is_available():
    from . import matmuls_int, matmuls_float
    from . import utils
    from . import lowerings

import numpy as np
import numba
from numba import cuda
import ctypes
import os