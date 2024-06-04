import numpy as np

import saffira as si
from saffira import projection_matrices as pm
import torch
import unittest
import logging
import saffira.gpu

# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARNING)

if __name__ == "__main__":
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype = np.float32)
    b = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]], dtype = np.float32)

    hw = si.SystolicArray(10, 10, 10, pm.no_local_reuse)
    f = si.fault_models.StuckAt(line="a", x=0, y=0, bit=2, polarity=1)
    hw.add_fault(f)
    c0 = a @ b
    c = hw.matmul(a, b)
    c1 = np.zeros((3, 3))
    saffira.gpu.injected_matmul[10, 10](a, b, c1, 0x11 * np.ones((3,3,3), dtype=np.int8) )
    print("correct", c0)
    print("old_method", c)
    print("new_method", c1)
    print( (c0 == c1).all() )

