from . import *

import unittest
import numpy as np
import random
import torch

class TestWrappings(unittest.TestCase):
    
    def test_matmul(self):
        for _ in range(10):
            N1 = np.random.randint(1, 100)
            N2 = np.random.randint(1, 100)
            N3 = np.random.randint(1, 100)

            A = (np.random.random((N1, N3)) * 100).astype(np.float32)
            B = (np.random.random((N3, N2)) * 100).astype(np.float32)

            Cok = A@B
            C = matmul(A, B, in_dtype=np.float32)

            self.assertTrue(
                np.allclose(C, Cok)
            )

    def test_convolve(self):
        for _ in range(10):
            N1 = np.random.randint(1, 100)
            N2 = np.random.randint(1, 100)
            while N2 >= N1:
                N2 = np.random.randint(1, 100)

            A = (np.random.random((N1, N1)) * 100).astype(np.float32)
            B = (np.random.random((N2, N2)) * 100).astype(np.float32)

            Cok = A@B
            C = matmul(A, B, in_dtype=np.float32)

            self.assertTrue(
                np.allclose(C, Cok)
            )

