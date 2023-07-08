import numpy as np
from .fault_models import *  # Contains the class Fault
import uuid


class TransformedFault:

    # TODO: implement this method (and the class Fault) such that f.t can be a string!
    def __init__(self, f: Fault, t: np.ndarray):
        s = np.array([f.x, f.y, f.t])
        t_inv = np.linalg.inv(t)
        nu = t_inv @ s
        self.location = nu


class SystolicArray:

    def __init__(self, n1: int, n2: int, n3: int, t: np.ndarray):
        """
        Initialize an object that performs the actual systolic multiplication C = A*B using the systolic equations

        Parameters
        ---
        n1 : rows of the first matrix A (corresponding to the rows of C)
        n2 : columns of the matrix B (corresponding to the columns of C)
        n3 : columns of the matrix A (corresponding to the rows of the matrix B)
        """
        self.N1 = n1
        self.N2 = n2
        self.N3 = n3
        self.T = t

        self.should_inject = False
        self.fault_list = {}

    def get_fault_list(self):
        return self.fault_list

    def add_fault(self, fault: Fault):
        self.should_inject = True
        f = TransformedFault(fault, self.T)
        index = int(uuid.uuid1())
        self.fault_list[index] = f
        return index

    def clear_all_faults(self):
        self.fault_list = {}
        self.should_inject = False

    def clear_single_fault(self, id):
        self.fault_list.pop(id)
        self.should_inject = self.fault_list.__len__() == 0

    def matmul(self, A, B, history: list = []):
        """
        Performs the matrix multiplication between A and B.
        A must be size (ar, ac), B must have size (br, bc), with ac = br.
        The output C has size (ar, bc).

        Parameters
        ---
        A : first matrix
        B : second matrix
        history : this parameter is used to report the iterations over i, j and k of the multiplication

        Returns
        ---
        o : multiplication a * b
        """

        # We only can do multiplication of 2D matrices!
        assert len(A.shape) == 2 and len(B.shape) == 2, "matmul only accepts 2D matrices!!!"

        ar, ac = A.shape
        br, bc = B.shape

        if ac != br:
            raise Exception("matrix not compatible!")

        N1 = ar + 1
        N2 = bc + 1
        N3 = br + 1  # same as ac + 1

        # TODO: Figure out a way to implement folding (or tiling)!
        assert self.N1 >= N1
        assert self.N2 >= N2
        assert self.N3 >= N3

        # TODO: replace this arrays with simpler structures that takes into account
        # only the actual data we are using, not the entire iteration vector space
        a = np.zeros((N1, N2, N3))
        b = np.zeros((N1, N2, N3))
        c = np.zeros((N1, N2, N3))

        # input operations
        j = 0
        for i in range(1, N1):
            for k in range(1, N3):
                a_i = 1 if i == 0 else i
                a_k = 1 if k == 0 else k
                a[i, j, k] = A[a_i - 1, a_k - 1]
        i = 0
        for j in range(1, N2):
            for k in range(1, N3):
                b_j = 1 if j == 0 else j
                b_k = 1 if k == 0 else k
                b[i, j, k] = B[b_k - 1, b_j - 1]

        # actual computations
        for i in range(1, N1):
            for j in range(1, N2):
                for k in range(1, N3):
                    # print(str(i) + " " + str(j) + " " + str(k))
                    a[i, j, k] = a[i, j - 1, k]
                    b[i, j, k] = b[i - 1, j, k]

                    # Actual injection
                    # TODO: Figure out some good way to characterize and implement the fault!
                    if self.should_inject:
                        for fault in self.fault_list.values():
                            if ([i, j, k] == fault.location).all():
                                print("Injecting a! Old value ", end=" ")
                                print(a[i, j, k], end=" ")
                                newValue = int(a[i, j, k]) ^ (1 << 8)
                                a[i, j, k] = newValue
                                print("newValue ", newValue)

                    c[i, j, k] = c[i, j, k - 1] + a[i, j - 1, k] * b[i - 1, j, k]

        history.extend([a, b, c])

        C = c[1:, 1:, N3 - 1]
        return C
