import numpy as np
import uuid
import logging
import bitarray
from collections import defaultdict

from .utils import LineType
from .fault_models import *  # Contains the class Fault
from .exceptions import *
from . import utils
from . import cpu

class SystolicArray:

    def __init__(self,
                 n1: int, n2: int, n3: int,
                 T: np.ndarray,
                 in_dtype: np.dtype = np.dtype(np.int8),
                 mac_dtype: np.dtype = None,
                 optimized = False,
                 use_legacy = True,
                 # TODO approximate_multiplier = None, approximate_adder = None
                 ):
        """
        Initialize an object that performs the actual systolic multiplication C = A*B using the systolic equations

        Parameters
        ---
        n1 : rows of the first matrix A (corresponding to the rows of C)
        n2 : columns of the matrix B (corresponding to the columns of C)
        n3 : columns of the matrix A (corresponding to the rows of the matrix B)
        T : projection matrix used for determining the type of systolic array
        in_dtype : represents the type of the input data seen in the hardware. It only corresponds to the type of input and outputs
        mac_dtype: (optional) corresponds to the hardware type used for the accumulation registers. If none, is computed automatically to be 4 times wider than in_dtype
        computation_method: (optional) corresponds to the computation method used for the multiplication computation, whether made on the GPU or using the traditional slow method of iterating on i, j and k
        custom_matmul: (optional) corresponds to the custom matrix multiplication operation
        """

        logging.info(f"[SystolicArray] instantiating a new systolic array...")

        # Physical parameters
        self.N1 = n1
        self.N2 = n2
        self.N3 = n3
        self.T = T # Transformation matrix
        logging.info(f"[SystolicArray] N1: {self.N1}, N2: {self.N2}, N3: {self.N3}")

        # dtypes
        self.in_dtype = np.dtype(in_dtype)
        # Explicit dtype for accumulation without overflow
        if mac_dtype is not None:  # Explicit mac_dtype given
            self.mac_dtype = np.dtype(mac_dtype)
        elif self.in_dtype.kind == "i":
            self.mac_dtype = np.dtype(in_dtype.kind + str(in_dtype.itemsize * 4))
        else:
            self.mac_dtype = np.dtype(in_dtype)

        # Injection parameters
        self.should_inject = False
        self.fault_list = {}
        self.physical_mapping = {}
        self.physical_space = []
        self.physical_PEs = []
        self.injected_points_list = {}

        nbits = self.in_dtype.itemsize*8
        self.injection_a = np.zeros((self.N1, self.N2, self.N3), dtype=f"int{nbits}")
        self.injection_b = np.zeros((self.N1, self.N2, self.N3), dtype=f"int{nbits}")
        self.injection_c = np.zeros((self.N1, self.N2, self.N3),
                    dtype=f"int{nbits}" if self.in_dtype.kind=="f" else "int{nbits*4}")

        # It is not possible to get all the iterative positions mathematically using P^-1, so we use this function
        # to map the iteartion space to the physical space and we will have a list of iterative points (i, j, k)
        # associated to the physical points (x, y).
        self._iter_to_phy()

        self.computation_time = 1 + (self.N1 + self.N2 + self.N3) - self.T[2,:].sum()
            # This is the basic formula that says the time for a single computation is (1 + t_max - t_min) where t_min
            # and t_max are the extremes of the set of time points computed as pi * nu (pi is the time-projection vector
            # and nu = (i, j, k) ).

        # TODO Approximated operators
        # self.multiplier = np.matmul if approximate_multiplier is None else approximate_multiplier
        # self.adder = np.add

        # Method choice (performance related)
        self.optimized = optimized
        self.use_legacy = use_legacy


######### CORE #######################################
    def matmul(self, A, B):
        """
        Performs the matrix multiplication between A and B.
        A must be size (ar, ac), B must have size (br, bc), with ac = br.
        The output C has size (ar, bc).
        This functions checks the correctness of the parameters just before calling the actual matmuls

        Parameters
        ---
        A : first matrix
        B : second matrix

        Returns
        ---
        o : multiplication a * b
        """
        logging.info(f"[SystolicArray] processing matrix multiplication...")

        # ##################### Safe Casting and data structures instantiations
        # Safe casting to avoid overflow
        if not isinstance(A, np.ndarray):
            logging.warning(f"[SystolicArray] Matrix A is not instance of numpy.ndarray. This may cause problems")
        if not isinstance(B, np.ndarray):
            logging.warning(f"[SystolicArray] Matrix B is not instance of numpy.ndarray. This may cause problems")

        # A = A.astype(self.dtype, casting="safe")
        # Would be nice to use the astype method,
        # but it doesn't check dynamically whether the values fit in the new type
        A_np = np.array(A, dtype=self.in_dtype)
        if not (A_np == A).all():
            raise CastingError(
                f"Couldn't convert A from {type(A)} to {self.in_dtype} because some values are greater than admissible.\n"
                f"The max value is: {np.max(A)}. Have you considered signed and unsigned types?\n"
                f"Matrix A was: \n{A}")
        # B = B.astype(self.dtype, casting="safe")
        B_np = np.array(B)
        if not (B_np == B).all():
            raise CastingError(
                f"Couldn't convert B from {type(B)} to {self.in_dtype} because some values are greater than admissible."
                f"Matrix B was: \n{B}")

        A = A_np
        B = B_np
        del A_np
        del B_np

        logging.info(f"[SystolicArray] checking all the requirements")
        # We only can do multiplication of 2D matrices!
        assert len(A.shape) == 2 and len(B.shape) == 2, "matmul only accepts 2D matrices!!!"

        ### Actual comptation 
        res = None
        if self.use_legacy and self.optimized:
            res = self._matmul_old_opt(A, B)
        elif self.use_legacy and not self.optimized:
            res = self._matmul_old(A, B)
        elif not self.use_legacy and self.optimized:
            raise NotImplementedError("This feature has not yet been implemented. You can't use use_legacy=False and optimized=True")
        else: #not self.use_legacy and not self.optimized
            res = self._matmul_new(A, B)
        return res

    def _matmul_old_opt(self, A, B):
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

        C = np.zeros((N1-1, N2-1), dtype=self.mac_dtype)
        if self.in_dtype.name == "float32":
            inj_a = np.zeros((N1, N2, N3), dtype=np.int32)
            inj_b = np.zeros((N1, N2, N3), dtype=np.int32)
            inj_c = np.zeros((N1, N2, N3), dtype=np.int32)
            #inj_c[1, 1, 1] = 0xFFFFFFFF
            #inj_c[1, 1, 2] = 0xFFFFFFFF
            # inj_c[1, 1, 3] = 0xFFFFFFFF
            cpu.injected_matmul_old_f32(A, B, C,
                inj_a, inj_b, inj_c, 0)
        elif self.in_dtype.name == "float64":
            inj_c = np.zeros((N1, N2, N3))
            cpu.injected_matmul_old_f64(A, B, C,
                np.zeros((N1, N2, N3)), np.zeros((N1, N2, N3)), inj_c, 0)
            raise Exception("ERROR GENERIC")
        elif self.in_dtype.kind == "i":
            cpu.injected_matmul_old_int(A, B, C,
                np.zeros((N1, N2, N3)), np.zeros((N1, N2, N3)), np.zeros((N1, N2, N3)), 0)
            raise Exception("ERROR GENERIC")
        else:
            raise Exception(f"Unsupported dtype: {self.in_dtype}")
        """ print(A.shape)
        A.tofile("A.h5")
        print(B.shape)
        B.tofile("B.h5")
        print(C.shape)
        C.tofile("C.h5") """

        # assert np.allclose(C, (A@B), atol=1e-6, rtol=1e-5)

        return C

    def _matmul_new(self, A: np.ndarray, B: np.ndarray):
        # C = A @ B  # This is the golden part
        C = self.multiplier(A, B)
        C_f = np.zeros_like(C)

        for element, accs in self.injected_points_list.items():
            i, j = element
            if i >= A.shape[0] or j >= B.shape[1]:
                continue  # We skip elements not used for this computation
            ks = accs.keys()
            fault = list(self.injected_points_list[(i,j)].values())[0]
            k_min = min(ks)
            k_max = max(ks)
            a_bar = A[i, k_min:k_max+1]  # Because of how the mapping on a[i, j, k] is done
            b_bar = B[k_min:k_max+1, j]
            c_g = self.multiplier(a_bar, b_bar)  # a_bar @ b_bar
            C_f[i, j] = c_g - self._fault_function(a_bar, b_bar, fault)
        return C + C_f

    def _matmul_old(self, A, B):

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

        logging.info(f"[SystolicArray] preparing data structures...")

        # TODO: replace this arrays with simpler structures that takes into account
        # only the actual data we are using, not the entire iteration vector space
        a = np.zeros((N1, N2, N3), dtype=self.in_dtype)
        b = np.zeros((N1, N2, N3), dtype=self.in_dtype)
        c = np.zeros((N1, N2, N3), dtype=self.mac_dtype)

        # ############################# Input operations and injections on a and b
        logging.debug(f"[SystolicArray] performing input operations...")
        j = 0
        for i in range(0, N1):
            for k in range(0, N3):
                a_i = 1 if i == 0 else i
                a_k = 1 if k == 0 else k
                a[i, j, k] = A[a_i - 1, a_k - 1]
                # REMEMBER: although we are initializing the whole array, it doesn't make sense (for our mathematical
                # framework) to talk about a[i, j, k] when either i == 0 or k == 0
        i = 0
        for j in range(1, N2):
            for k in range(1, N3):
                b_j = 1 if j == 0 else j
                b_k = 1 if k == 0 else k
                b[i, j, k] = B[b_k - 1, b_j - 1]
                # REMEMBER: as previously stated for a, it doesn't make sense (for our mathematical
                # framework) to talk about b[i, j, k] when either j == 0 or k == 0

        logging.debug(f"[SystolicArray] forwarding a and b values... ")

        # Forward a and b values (the fast way)
        for i in range(1, N1):
            # Forward the values of a and b all at once (for performance)
            a[:, i, :] = a[:, i - 1, :]
            b[i, :, :] = b[i - 1, :, :]

        # Injecting values on lines a and b
        if self.should_inject:
            # ### line a
            logging.debug(f"[SystolicArray] [a] injecting line a")
            for nu, fault_list in self._line_faults[LineType.a.value - 1].items():
                i, j, k = nu
                if i >= N1 or j >= N2 or k >= N3:
                    logging.info(f"[SystolicArray] [a] iteration {nu} is not used for this computation "
                                 f"with size {N1, N2, N3}, so it won't be injected")
                    continue
                for fault in fault_list:
                    logging.debug(f"[SystolicArray] [a] injecting iteration {i, j, k}")
                    a[i, j - 1, k] = self._inject_value(a[i, j - 1, k], fault.should_reverse_bits, fault.bit,
                                                        fault.polarity)
            # ### line b
            logging.debug(f"[SystolicArray] [b] injecting line b")
            for nu, fault_list in self._line_faults[LineType.b.value - 1].items():
                i, j, k = nu
                if i >= N1 or j >= N2 or k >= N3:
                    logging.info(f"[SystolicArray] [b] iteration {nu} is not used for this computation "
                                 f"with size {N1, N2, N3}, so it won't be injected")
                    continue
                for fault in fault_list:
                    logging.debug(f"[SystolicArray] [b] injecting iteration {i, j, k}")
                    b[i - 1, j, k] = self._inject_value(b[i - 1, j, k], fault.should_reverse_bits, fault.bit,
                                                        fault.polarity)

            logging.info(f"[SystolicArray] data structures ready")

        # ###################################### Actual computations and injections on c
        logging.debug(f"[SystolicArray] starting the actual computations...")
        for i in range(1, N1):
            for j in range(1, N2):
                for k in range(1, N3):

                    # Originally the algorithm was performed like this, but it's quite inefficient...
                    # a[i, j, k] = a[i, j - 1, k]
                    # b[i, j, k] = b[i - 1, j, k]

                    c[i, j, k] = (c[i, j, k - 1] +
                                  # here the explicit cast is required for avoiding overflow
                                  a[i, j - 1, k].astype(dtype=self.mac_dtype, casting="safe") *
                                  b[i - 1, j, k].astype(dtype=self.mac_dtype, casting="safe")
                                  )

                    # Actual injection
                    if self.should_inject:
                        fault_list = self._line_faults[LineType.c.value - 1].get((i, j, k))
                        if fault_list is not None:
                            for fault in fault_list:
                                c[i, j, k] = self._inject_value(
                                    c[i, j, k],
                                    fault.should_reverse_bits,
                                    fault.bit,
                                    fault.polarity,
                                )

        """ for j in range(N1-1):
            r = a[:, j, :] == a[:, j-1, :]
            print(r.all())

        exit(0) """

        # c = c[:,:,0:N3] + a[i, j - 1, k] * b[i - 1, j, k]
        logging.info(f"[SystolicArray] computation done")

        C = c[1:, 1:, N3 - 1]

        del a, b, c

        return C

############ Fault related functions ########################################################
    def get_fault_list(self):
        return self.fault_list

    def add_fault(self, fault: Fault, perform_injection_prep: bool = True):
        self.should_inject = True
        index = int(uuid.uuid1())
        self.fault_list[index] = fault

        if perform_injection_prep:
            # Populate the dictionaries self._line_faults
            self._preperare_injection_parameters()
        logging.debug(f"[SystolicArray] fault added, fault_parameters: {self._line_faults}")
        return index

    def clear_all_faults(self):
        self.fault_list = {}
        self._preperare_injection_parameters()
        self.should_inject = False

    def clear_single_fault(self, id):
        self.fault_list.pop(id)
        self.should_inject = self.fault_list.__len__() == 0
        self._preperare_injection_parameters()

    def _iter_to_phy(self):
        logging.debug(f"[SystolicArray] starting iter_to_phy")
        P = self.T[0:2, :]  # space projection matrix
        logging.debug(f"[SystolicArray] space projection matrix is P: \n{P}")
        for i in range(1, self.N1):
            for j in range(1, self.N2):
                for k in range(1, self.N3):
                    nu = np.array([i, j, k])
                    eps = tuple(
                        self.T @ nu )  # We can't use np.ndarray as a key for the dictionary, so we convert it into tuple
                    s = eps[0:2]
                    t = eps[2]
                    nu = tuple(nu)
                    point_list = self.physical_mapping.get(s)
                    if point_list is None:
                        self.physical_mapping.update({s: [(nu, t)]})
                    else:
                        point_list.append((nu, t))
                    self.physical_space.append( eps )
                    self.physical_PEs.append( s )
        logging.debug(f"[SystolicArray] iteration space to physical space done:\n{self.physical_mapping}")
        if logging.root.level <= logging.DEBUG:
            from matplotlib import pyplot as plt
            plt.set_loglevel("info")
            points = np.array(list(self.physical_mapping.keys()))
            plt.scatter(points[:, 0], points[:, 1])
            plt.gca().invert_yaxis()
            plt.show()
            del plt

    def _preperare_injection_parameters(self):
        # self._line_faults is an array of dictionaries. There are as many dictionaries are LineTypes.
        # Each dictionary uses the triplet (i, j, k) as the key and each value should be a list of faults affecting
        # that iteration.
        self._line_faults = [{} for i in LineType]

        # Computing the flow directions for each line
        P = self.T[0:2, :]  # space projection matrix
        logging.debug(f"P matrix is\n{P}")
        flow_dirs = [  # Computed from the Uniform Recurrent Equations system
            P @ np.array([1, 0, 0]),  # for b (same order of LineType: b first, a second, and c last)
            P @ np.array([0, 1, 0]),  # for a
            P @ np.array([0, 0, 1]),  # for c
        ]
        del P
        logging.debug(f"[SystolicArray] flow_dirs is\n{flow_dirs}")

        # Computing T_inv for space conversion
        T_inv = np.linalg.inv(self.T)

        # This is for self._matmul_old
        for fault in self.fault_list.values():  # we don't care about the keys, they are generated with uuid for the user
            # iteration variables
            s = np.array([fault.x, fault.y])
            t_start = fault.t_start
            t_stop = fault.t_stop
            # For each fault we want to inject all the PEs cascading from the first and forwarding the faulty value
            while True:  # injecting the PEs on the same flow_direction
                # logging.debug(f"[SystolicArray] preparing injection for element {s}")
                nu_list = self.physical_mapping.get(tuple(s))
                if nu_list is None:
                    # Given the first element (x, y) is part of the physical space, the other elements are always
                    # distant flow_dirs, so if we find a None it means we got outside :D
                    #logging.debug(f"[SystolicArray] element {s} is not part of the physical space. Breaking")
                    break

                for nu, t in nu_list:  # time loop
                    if t_stop >= t >= t_start:  # This means element nu will be injected
                        # logging.debug(f"iteration {nu} will be injected")
                        injected_list = self._line_faults[fault.line.value - 1].get(nu)
                        if injected_list is None:
                            self._line_faults[fault.line.value - 1].update({nu: [fault]})
                        else:
                            injected_list.append(fault)

                # we follow the next element to be injected given the flow direction of that line
                s = s + flow_dirs[fault.line.value - 1]
                if (flow_dirs[fault.line.value - 1] == np.zeros((3, 1))).all():
                    # logging.debug(f"[SystolicArray] element {s} constitutes a stationary variable. Breaking")
                    break
                # starting and stopping time are incremented by one for the next element
                t_start += 1
                t_stop += 1
        print("Injected list")
        from pprint import pprint
        # pprint(self._line_faults)
        

        # This is for _matmul_new
        injected_points_list = defaultdict(dict)
        T_det = round(np.linalg.det(self.T))
        # For each fault we want to inject all the PEs cascading from the first and forwarding the faulty value
        for fault in self.fault_list.values():  # we don't care about the keys, they are generated with uuid for the user
            s = np.array([fault.x, fault.y])
            t_start = fault.t_start if fault.t_start >= 3 else 3  # t_start and t_stop are used for
            t_stop = fault.t_stop
            while True:
                # iteration variables
                        ####  fault time-forwarding
                logging.debug(f"[SystolicArray] preparing injection for element {s}")
                x, y = s
                t = t_start
                while (x + y + t) % T_det != 0:  # We find the first t such that (i, j, k) is discrete
                    t += 1
                # TODO: Refactor injected_points_list to have min-max ranges associated to the faults!
                while t <= t_stop and t <= self.computation_time:  # Then we compute all the points jumping of T_det
                    i, j, k = (T_inv @ np.array([x, y, t])).astype(dtype=np.int32)
                    logging.debug(f"[SystolicArray] iteration {i, j, k} will be injected")
                    n = injected_points_list[(i, j)].get(k)
                    if n is None: n = fault
                    injected_points_list[(i, j)][k] = n # we keep track of how many faults we inject in element (i, j)
                                              # TODO: we may consider to change n and rather have a fault-list of sort
                    t += T_det  # we increment the time
                t_stop += 1  # we consider the fault affecting the element for the next CCs
                s = s + flow_dirs[fault.line.value - 1]  # moving in space
                t_start += 1
                t_stop += 1
                if tuple(s) not in self.physical_PEs:
                    # Given the first element (x, y) is part of the physical space, the other elements are always
                    # distant flow_dirs, so if we find a None it means we got outside :D
                    logging.debug(f"[SystolicArray] element {s} is not part of the physical space. Breaking")
                    break
                if (flow_dirs[fault.line.value - 1] == np.zeros((3, 1))).all():  # Unless we don't
                    logging.debug(f"[SystolicArray] element {s} constitutes a stationary variable. Breaking")
                    break

            nu_list = []
            for t in range(t_start, t_stop + 1):
                x = np.array([*s, t])
                nu = T_inv @ x
                # print(nu)
                # nu_list.append(nu)
            # nu_start

            # self._element_faults
        # pprint(injected_points_list)
        self.injected_points_list = injected_points_list

    def _fault_function(self, a: np.ndarray, b: np.ndarray, fault: Fault):
        if fault.line == LineType.a:
            for i in range(len(a)):
                a[i] = self._inject_value(a,
                                          fault.should_reverse_bits,
                                          fault.bit, fault.polarity)
        if fault.line == LineType.b:
            for i in range(len(b)):
                b[i] = self._inject_value(b,
                                          fault.should_reverse_bits,
                                          fault.bit, fault.polarity)
        c = a @ b
        if fault.line == LineType.c:
                c = self._inject_value(c,
                                      fault.should_reverse_bits,
                                      fault.bit, fault.polarity)
        return c


    def _inject_value(self, old_value: np.ndarray,
                      srb: bool = None,  # srb -> Should Reverse Bits (take a look at class Fault)
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
