import numpy as np

import saffira as si
from saffira import projection_matrices as pm
import torch
import unittest
import logging

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.WARNING)

if __name__ == "__main__":
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])

    hw = si.SystolicArray(10, 10, 10, pm.no_local_reuse)
    hw1 = si.SystolicArray(10, 10, 10, pm.no_local_reuse, use_old_injection_method=True)
    f = si.fault_models.StuckAt(line="a", x=0, y=0, bit=2, polarity=1)
    hw.add_fault(f)
    hw1.add_fault(f)
    c0 = a @ b
    c = hw.matmul(a, b)
    c1 = hw1.matmul(a, b)
    print("correct", c0)
    print("new_method", c)
    print("old_method", c1)


