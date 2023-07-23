import numpy as np
from .fault_models import *  # Contains the class Fault
import uuid
import logging
from . import utils


class TransformedFault:

    # TODO: implement this method (and the class Fault) such that f.t can be a string!
    def __init__(self, f: Fault, t: np.ndarray):
        s = np.array([f.x, f.y, f.t])
        t_inv = np.linalg.inv(t)
        nu = t_inv @ s
        self.location = nu


class SystolicArray:

    def __init__(self, n1: int, n2: int, n3: int, T: np.ndarray):
        """
        Initialize an object that performs the actual systolic multiplication C = A*B using the systolic equations

        Parameters
        ---
        n1 : rows of the first matrix A (corresponding to the rows of C)
        n2 : columns of the matrix B (corresponding to the columns of C)
        n3 : columns of the matrix A (corresponding to the rows of the matrix B)
        """
        logging.info(f"[SystolicArray] instantiating a new systolic array...")
        self.N1 = n1
        self.N2 = n2
        self.N3 = n3
        self.T = T

        logging.info(f"[SystolicArray] N1: {n1}, N2: {n2}, N3: {n3}")

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

    def matmul(self, A, B):
        """
        Performs the matrix multiplication between A and B.
        A must be size (ar, ac), B must have size (br, bc), with ac = br.
        The output C has size (ar, bc).

        Parameters
        ---
        A : first matrix
        B : second matrix

        Returns
        ---
        o : multiplication a * b
        """
        logging.info(f"[SystolicArray] processing matrix multiplication...")

        logging.info(f"[SystolicArray] checking all the requirements")
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
        assert self.N1 >= N1, f"N1 ({self.N1}) is too small for this matrix multiplication ({N1})"
        assert self.N2 >= N2, f"N2 ({self.N2}) is too small for this matrix multiplication ({N2})"
        assert self.N3 >= N3, f"N3 ({self.N3}) is too small for this matrix multiplication ({N3})"

        logging.info(f"[SystolicArray] requirements OK!")

        logging.info(f"[SystolicArray] prepering data structures...")

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

        logging.info(f"[SystolicArray] data structures ready")

        logging.info(f"[SystolicArray] starting computation")

        # utils.print_matrix_in_index(a, 1)

        # Forward a and b values (the fast way)
        for i in range(1, N1):
            a[:, i, :] = a[:, i-1, :]
            b[i, :, :] = b[i-1, :, :]

        if self.should_inject:
            # TODO: perform the injection on a and b
            # The thing is, when injecting on a and b and the forward is done as above, one should
            # consider multiple injections in different times. So, it is crucial to consider the precedence of the
            # forwarding of the faults. Specifically: if two faults occur on a in two different times t1 and t2 (with
            # t1 < t2), the values after t1 will be all corrupted and propagated forward. On top of that, the fault
            # that occurs in t2 will corrupt the already corrupted values of a.
            pass

        # actual computations
        for i in range(1, N1):
            for j in range(1, N2):
                for k in range(1, N3):
                    # print(str(i) + " " + str(j) + " " + str(k))

                    # Originally the algorithm was performed like this, but it's quite inefficient...
                    # a[i, j, k] = a[i, j - 1, k]
                    # b[i, j, k] = b[i - 1, j, k]

                    # Actual injection
                    if self.should_inject:
                        # TODO: implement the fault on c
                        for fault in self.fault_list.values():
                            if ([i, j, k] == fault.location).all():
                                loggingString = "Injecting a! Old Value:  " + str(a[i, j, k])
                                newValue = int(a[i, j, k]) ^ (1 << 8)
                                a[i, j, k] = newValue
                                loggingString += " - New Value: " + str(a[i, j, k])
                                logging.info(loggingString)

                    c[i, j, k] = c[i, j, k - 1] + a[i, j - 1, k] * b[i - 1, j, k]

        """ for j in range(N1-1):
            r = a[:, j, :] == a[:, j-1, :]
            print(r.all())

        exit(0) """

        # c = c[:,:,0:N3] + a[i, j - 1, k] * b[i - 1, j, k]
        logging.info(f"[SystolicArray] computation done")

        # print(("[SystolicArray] collecting history"))
        # history.extend([a, b, c])

        C = c[1:, 1:, N3 - 1]

        del a, b, c

        return C
