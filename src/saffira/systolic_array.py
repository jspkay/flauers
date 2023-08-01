import numpy as np
from .fault_models import *  # Contains the class Fault
import uuid
import logging
from . import utils
from .utils import LineType
import bitarray

class InjectionError(Exception):

    def __init__(self, message):
        super().__init__(self, message)

class SystolicArray:

    def __init__(self, n1: int, n2: int, n3: int, T: np.ndarray, dtype: np.dtype = np.dtype(np.int8)):
        """
        Initialize an object that performs the actual systolic multiplication C = A*B using the systolic equations

        Parameters
        ---
        n1 : rows of the first matrix A (corresponding to the rows of C)
        n2 : columns of the matrix B (corresponding to the columns of C)
        n3 : columns of the matrix A (corresponding to the rows of the matrix B)
        T : projection matrix used for determining the type of systolic array
        dtype : represents the type of the data seen in the hardware. It only corresponds to the type of input and outputs
        """
        logging.info(f"[SystolicArray] instantiating a new systolic array...")
        self.N1 = n1
        self.N2 = n2
        self.N3 = n3
        self.T = T
        self.dtype = dtype

        # Explicit dtype for accumulation without overflow
        self.mac_dtype = np.dtype(dtype.kind + str(dtype.itemsize * 4))


        logging.info(f"[SystolicArray] N1: {n1}, N2: {n2}, N3: {n3}")

        self.should_inject = False
        self.fault_list = {}
        self._a_faults = []
        self._b_faults = []
        self._c_faults = []

    def get_fault_list(self):
        return self.fault_list

    def add_fault(self, fault: Fault):
        self.should_inject = True
        fault.transform(self.T)
        index = int(uuid.uuid1())
        self.fault_list[index] = fault

        # Populate the lists self._X_faults with X in [a, b, c]
        self._preperare_injection_parameters()

        return index

    def clear_all_faults(self):
        self.fault_list = {}
        self._a_faults = []
        self._b_faults = []
        self._c_faults = []
        self.should_inject = False

    def clear_single_fault(self, id):
        self.fault_list.pop(id)
        self.should_inject = self.fault_list.__len__() == 0
        self._preperare_injection_parameters()

    def _preperare_injection_parameters(self):
        for fault in self.fault_list.values():
            if fault.line == LineType.a:
                self._a_faults.append(fault)
            elif fault.line == LineType.b:
                self._b_faults.append(fault)
            elif fault.line == LineType.c:
                self._c_faults.append(fault)

        def fn(f):
            return f.t_start
        self._a_faults.sort(key=fn)
        self._b_faults.sort(key=fn)
        self._c_faults.sort(key=fn)

        logging.debug(f"[_prepare_injection_parameters] lists list")
        logging.debug(f"a_faults: {self._a_faults}")
        logging.debug(f"b_faults: {self._b_faults}")
        logging.debug(f"c_faults: {self._c_faults}")

    def _inject_value(self, old_value : np.ndarray,
                      srb: bool = None, # srb -> Should Reverse Bits (take a look at class Fault)
                      bit: int = None,
                      polarity: int = None) -> np.ndarray:
        dtype = old_value.dtype
        logging.debug(f"[SystolicArray._inject_value] injecting! old value is {old_value}")
        bit_value = bitarray.bitarray()
        bit_value.frombytes(old_value.tobytes())
        logging.debug(f"[SystolicArray._inject_value] bit_value {bit_value}")
        bit_value.bytereverse()
        logging.debug(f"[SystolicArray._inject_value] after reverse {bit_value}")
        if srb:
            bit_value.reverse()
        bit_value[bit] = polarity
        logging.debug(f"[SystolicArray._inject_value] after injection {bit_value}")
        if srb:
            bit_value.reverse()
        bit_value.bytereverse()
        logging.debug(f"[SystolicArray._inject_value] after reverse {bit_value}")
        new_value = np.frombuffer(bit_value.tobytes(), dtype=dtype)[0]
        logging.debug(f"[SystolicArray._inject_value] new value {new_value}")
        return new_value

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

        A = np.array(A, dtype=self.dtype)
        B = np.array(B, dtype=self.dtype)

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

        logging.info(f"[SystolicArray] preparing data structures...")

        # TODO: replace this arrays with simpler structures that takes into account
        # only the actual data we are using, not the entire iteration vector space
        a = np.zeros((N1, N2, N3), dtype=self.dtype)
        b = np.zeros((N1, N2, N3), dtype=self.dtype)
        c = np.zeros((N1, N2, N3), dtype=self.mac_dtype)

        # input operations
        logging.debug(f"[SystolicArray] performing input operations...")
        j = 0
        for i in range(0, N1):
            for k in range(0, N3):
                a_i = 1 if i == 0 else i
                a_k = 1 if k == 0 else k
                a[i, j, k] = A[a_i - 1, a_k - 1]
                # REMEMBER: although we are initializing the whole array, it doesn't make sense (for our mathematical
                # framework) to talk about a[i, j, k] when either i == 0 or k == 0

        if self.should_inject:
            logging.debug(f"[SystolicArray] injecting line a input values")
            for fault in self._a_faults:
                i, j, k = fault.iteration_start
                if i > 0 and j >= 0 and k > 0: # we'll take care of this fault later
                    logging.debug(f"[SystolicArray] skipping fault in {i, j, k}")
                    continue
                # at least one among i, j and k is 0. So we need to change the value
                logging.debug(f"[SystolicArray] injecting {i, j, k}")
                i = i if i > 0 else 1
                j = j if j >= 0 else 0 # j can actually be 0
                k = k if k > 0 else 1
                logging.debug(f"[SystolicArray] new iteration values are {i, j, k}")
                a[i, j, k] = self._inject_value(A[i-1, k-1], fault.should_reverse_bits, fault.bit, fault.polarity)

        i = 0
        for j in range(1, N2):
            for k in range(1, N3):
                b_j = 1 if j == 0 else j
                b_k = 1 if k == 0 else k
                b[i, j, k] = B[b_k - 1, b_j - 1]
                # REMEMBER: as previously stated for a, it doesn't make sense (for our mathematical
                # framework) to talk about b[i, j, k] when either j == 0 or k == 0

        # exit(0)

        logging.info(f"[SystolicArray] data structures ready")

        logging.info(f"[SystolicArray] forwarding a and b values... ")

        # Forward a and b values (the fast way)
        for i in range(1, N1):
            # Forward the values of a and b all at once (for performance)
            a[:, i, :] = a[:, i-1, :]
            b[i, :, :] = b[i-1, :, :]

        if self.should_inject:
            pass
            # TODO: perform proper the injection on a and b
            # The thing is, when injecting on a and b and the forward is done as above, one should
            # consider multiple injections in different times. So, it is crucial to consider the precedence of the
            # forwarding of the faults. Specifically: if two faults occur on a in two different times t1 and t2 (with
            # t1 < t2), the values after t1 will be all corrupted and propagated forward. On top of that, the fault
            # that occurs in t2 will corrupt the already corrupted values of a.

            # For now only stuck-at injection work properly.
            # On the actual computation cycles, we only iterate for i, j, k all greater strictly greater than 0.
            # It may happen that when injecting the values of i, j, and k (computed using the inverse of the projection
            # matrix) could smaller than or equal to 0. Thus, we need to inject the input values if that's the case
            for fault in self._a_faults:
                it_start = list(map(lambda x: int(x), fault.iteration_start))
                it_stop = list(map(lambda x: int(x), fault.iteration_stop))
                for i in range(it_start[0], min(it_stop[0]+1, N1)):
                    for j in range(it_start[1], min(it_stop[1]+1, N2)):
                        for k in range(it_start[2], min(it_stop[2]+1, N3)):
                            logging.debug(f"[SystolicArray] injecting line a on iteration {(i,j,k)}")
                            logging.debug(f"[SystolicArray] line a old value is {a[i, j, k-1]}")
                            a[i, j, k] = self._inject_value(a[i, j, k], srb=fault.should_reverse_bits, bit=fault.bit, polarity=fault.polarity)
                            logging.debug(f"[SystolicArray] line a new value is {a[i, j, k]}")
            for fault in self._b_faults:
                # TODO: implement as for a
                pass

        # actual computations
        logging.debug(f"[SystolicArray] starting the actual computations...")
        for i in range(1, N1):
            for j in range(1, N2):
                for k in range(1, N3):
                    # print(str(i) + " " + str(j) + " " + str(k))

                    # Originally the algorithm was performed like this, but it's quite inefficient...
                    # a[i, j, k] = a[i, j - 1, k]
                    # b[i, j, k] = b[i - 1, j, k]

                    # Actual injection
                    if self.should_inject:
                        # logging.debug(f"Checking for injection on c")
                        # TODO: implement the fault on c
                        for fault in self._c_faults:
                            if utils.is_comprised([i, j, k], fault.iteration_start, fault.iteration_stop):
                                old_value = c[i, j, k]
                                logging.info(f"[SystolicArray] injecting c in iteration {(i, j, k)}")
                                logging.debug(f"[SystolicArray] old value of c[{i, j, k}] = {old_value}")

                                c[i, j, k - 1] = self._inject_value(c[i, j, k], fault.should_reverse_bits, fault.bit, fault.polarity)
                                """
                                bit_value.frombytes( old_value.tobytes() )
                                bit_value.bytereverse()
                                if fault.should_reverse_bits:
                                    bit_value.reverse()
                                bit_value[fault.bit] = fault.polarity
                                if fault.should_reverse_bits:
                                    bit_value.reverse()
                                bit_value.bytereverse()

                                new_value = np.frombuffer(bit_value.tobytes(), dtype=self.dtype)[0]

                                c[i, j, k] = new_value
                                bit_value = bitarray.bitarray()
                                """

                                logging.debug(f"[SystolicArray] injected value c[{i, j, k}] = {c[i, j, k]}")

                    c[i, j, k] = ( c[i, j, k - 1] +
                                    np.array(a[i, j - 1, k], dtype=self.mac_dtype) *
                                    np.array(b[i - 1, j, k], dtype=self.mac_dtype)
                                    )
                    print(c[i, j, k])

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
