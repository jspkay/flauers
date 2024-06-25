from . import tilings

import unittest
import random
import numpy as np

class TestTiling(unittest.TestCase):
    def test_basic_tiling(self):
        A = np.array([ [1,2, 3], [4, 5, 6], [7,8,9]])
        B = np.array([ [1,2, 3], [4, 5, 6], [7,8,9]])

        print("### A ###:")
        print(A)

        print("### B ###:")
        print(B)

        print("### C = A @ B ###")
        print(A@B)
        print("######")

        C = np.zeros((3, 3))
        N1 = 2
        N2 = 2
        N3 = 4
        it = tilings.Tiling(A.shape, B.shape, N1, N2, N3)
        for i, j, k in it:
            a = A[i:i+N1, k:k+N3]
            b = B[k:k+N3, j:j+N2]
            print( a @ b )
            print("----")
            C[i:i+N1, j:j+N2] += a@b
        
        print("#####")
        print(C)

        print( np.allclose(C, A@B) )

    def test_general_tiling(self):
        for _ in range(1000):
            N1 = random.randint(1, 100)
            N2 = random.randint(1, 100)
            N3 = random.randint(1, 100)

            n1 = random.randint(N1, 200)
            n2 = random.randint(N2, 200)
            n3 = random.randint(N3, 200)

            A = np.random.random((n1, n3))
            B = np.random.random((n3, n2))

            C = np.zeros((n1, n2))
            
            it = tilings.Tiling(A.shape, B.shape, N1, N2, N3)
            for i, j, k in it: 
                a = A[i:i+N1, k:k+N3]
                b = B[k:k+N3, j:j+N2]
                C[i:i+N1, j:j+N2] += a@b

            r = np.allclose(C, A@B, rtol=0.1)
            print( r )
            if not r:
                print(C)
                print(A@B)
            self.assertTrue(r)