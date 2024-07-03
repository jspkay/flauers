import sys
sys.path.append("..")
import flauers

import numpy as np
import unittest
import itertools
from numba import cuda

# TODO Parametrize the tests https://github.com/wolever/parameterized
#   You want to execute all the tests with the different strategies (with and without numba, on the gpu, 
#   with and without the legacy algorithm)

class TestSystolicArray(unittest.TestCase):
    
    def test_instantiation(self):
        iterator = itertools.product(
            range(12), range(12), range(12)
        )

        matrices = [
            flauers.projection_matrices.output_stationary,
            flauers.projection_matrices.row_stationary,
            flauers.projection_matrices.col_stationary,
            flauers.projection_matrices.no_local_reuse,
            flauers.projection_matrices.row_stationary_eq,
            flauers.projection_matrices.col_stationary_eq
        ]

        for _ in range(1000):
            for matrix in matrices:
                for N1, N2, N3 in iterator:
                    A = (np.random.random((N1, N3)) * 1e9).astype(np.int8)
                    B = (np.random.random((N3, N2)) * 1e9).astype(np.int8)
                    array = flauers.SystolicArray(
                        N1, N2, N3,
                        matrix,
                    )

                    C = array.matmul(A, B)
                    Cok = A.astype(np.int32) @ B.astype(np.int32)

                    self.assertTrue( np.allclose(C, Cok) )

    def test_instantiation_gpu(self):
        iterator = itertools.product(
            range(1, 12), range(1, 12), range(1, 12)
        )

        matrices = [
            flauers.projection_matrices.output_stationary,
            flauers.projection_matrices.row_stationary,
            flauers.projection_matrices.col_stationary,
            flauers.projection_matrices.no_local_reuse,
            flauers.projection_matrices.row_stationary_eq,
            flauers.projection_matrices.col_stationary_eq
        ]

        for _ in range(1):
            for matrix in matrices:
                for N1, N2, N3 in iterator:
                    A = (np.random.random((N1, N3)) * 1e9).astype(np.float32)
                    B = (np.random.random((N3, N2)) * 1e9).astype(np.float32)
                    array = flauers.SystolicArray(
                        N1, N2, N3,
                        matrix,
                        in_dtype = np.float32,
                        use_gpu=True,
                    )

                    Cok = A.astype(np.float32) @ B.astype(np.float32)

                    A = cuda.to_device(A)
                    B = cuda.to_device(B)
                    print(N1, N2, N3)
                    C = array.matmul(A, B, tiling=True)

                    self.assertTrue( np.allclose(C, Cok),
                    msg=f"projection matrix is\n{matrix}\n"
                        f"A is\n{A.copy_to_host()}\n"
                        f"B is\n{B.copy_to_host()}\n"
                        f"C is\n{C.copy_to_host()}\n"
                        f"Cok is\n{Cok}"
                        )

                    del A, B, C, Cok


class TestSystolicArrayOSInjections(unittest.TestCase):

    def setUp(self):
        self.N1 = 9
        self.N2 = 10
        self.N3 = 11

        self.hw = flauers.SystolicArray(
            self.N1, self.N2, self.N3,
            flauers.projection_matrices.output_stationary,
            in_dtype = np.int8
        )

        self.A = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ], dtype = np.int8)

        self.B = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]
        ], dtype = np.int8)

        A = (np.random.random((self.N1, self.N3)) * 1e9).astype(np.int8)
        B = (np.random.random((self.N3, self.N2)) * 1e9).astype(np.int8)

        self.C = self.A.astype(np.int32) @ self.B.astype(np.int32)

    def test_injection_c(self):
        couples = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        
        for dx, dy in couples:
            f = flauers.fault_models.StuckAt(
                "c",
                x = dx+1, y = dy+1,
                bit = 5, polarity = 1,
                msb = "last"
            )

            self.hw.add_fault(f)

            C = self.hw.matmul(self.A, self.B)

            """
            When we perform this injection, we have a different value in register c.
            This means that rather than doning [1 1 1] * [1 1 1]' = 3 we will have something
            very different!
            Specifically, the sequence of operations is the following:

            c = 0
            repeat 3-times:
                c += 1 * 1
                c |= 0x1 << 5  # this is because of the injection
            
            So the result will be 3 | (0x1 << 5) = 32 + 3 = 35
            """

            self.assertEqual(C[dx, dy], self.C[dx, dy] + 32)

    def test_injection_a(self):
        couples = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        
        for dx, dy in couples:
            f = flauers.fault_models.StuckAt(
                "a",
                x = dx+1, y = dy+1,
                bit = 5, polarity = 1,
                msb = "last"
            )

            self.hw.add_fault(f)

            C = self.hw.matmul(self.A, self.B)

            """
            When we perform this injection, we have a different value in register a (containing values from A).
            This means that rather than doning 1*1 + 1*1 + 1*1, 
            we will have (1 | (1<<5))*1 + (1 | (1<<5))*1 + (1 | (1<<5))*1 = 33 + 33 + 33 = 99 (in 0, 0)
            The same thing happens on the (0, 1) and (0, 2) with different values
            
            """

            for row in range(self.A.shape[0]):
                for col in range(self.B.shape[1]):
                    value = 0
                    for k in range(self.A.shape[1]):
                        if row == dx and col >= dy:    
                            value += ( self.A[row, k] | 32 ) * self.B[k, col]
                        else:
                            value += ( self.A[row, k] ) * self.B[k, col]
                    self.assertEqual(C[row, col], value, 
                        msg=f"assert failed for {row, col} when injecting element {(dx, dy)} (PE: {dx+1, dy+1})"
                    )

            self.hw.clear_all_faults()

    def test_injection_b(self):
        couples = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        
        for dx, dy in couples:
            f = flauers.fault_models.StuckAt(
                "b",
                x = dx+1, y = dy+1,
                bit = 5, polarity = 1,
                msb = "last"
            )

            self.hw.add_fault(f)

            C = self.hw.matmul(self.A, self.B)

            """
            When we perform this injection, we have a different value in register a (containing values from A).
            This means that rather than doning 1*1 + 1*1 + 1*1, 
            we will have (1 | (1<<5))*1 + (1 | (1<<5))*1 + (1 | (1<<5))*1 = 33 + 33 + 33 = 99 (in 0, 0)
            The same thing happens on the (0, 1) and (0, 2) with different values
            
            """

            for row in range(self.A.shape[0]):
                for col in range(self.B.shape[1]):
                    value = 0
                    for k in range(self.A.shape[1]):
                        if row >= dx and col == dy:    
                            value += self.A[row, k] * ( self.B[k, col] | 32 )
                        else:
                            value += self.A[row, k] * ( self.B[k, col] )
                    self.assertEqual(C[row, col], value, 
                        msg=f"assert failed for {row, col} when injecting element {(dx, dy)} (PE: {dx+1, dy+1})"
                    )

            self.hw.clear_all_faults()

class TestSystolicArrayOSInjectionsCuda(unittest.TestCase):

    def setUp(self):
        self.N1 = 9
        self.N2 = 10
        self.N3 = 11

        self.hw = flauers.SystolicArray(
            self.N1, self.N2, self.N3,
            flauers.projection_matrices.output_stationary,
            in_dtype = np.int8,
            use_gpu = True,
        )

        self.A = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ], dtype = np.int8)

        self.B = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]
        ], dtype = np.int8)

        A = (np.random.random((self.N1, self.N3)) * 1e9).astype(np.int8)
        B = (np.random.random((self.N3, self.N2)) * 1e9).astype(np.int8)

        self.C = self.A.astype(np.int32) @ self.B.astype(np.int32)

        self.A = cuda.to_device(self.A)
        self.B = cuda.to_device(self.B)

    def test_injection_c(self):
        couples = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        
        for dx, dy in couples:
            f = flauers.fault_models.StuckAt(
                "c",
                x = dx+1, y = dy+1,
                bit = 5, polarity = 1,
                msb = "last"
            )

            self.hw.add_fault(f)

            C = self.hw.matmul_cuda(self.A, self.B)

            """
            When we perform this injection, we have a different value in register c.
            This means that rather than doning [1 1 1] * [1 1 1]' = 3 we will have something
            very different!
            Specifically, the sequence of operations is the following:

            c = 0
            repeat 3-times:
                c += 1 * 1
                c |= 0x1 << 5  # this is because of the injection
            
            So the result will be 3 | (0x1 << 5) = 32 + 3 = 35
            """

            self.assertEqual(C[dx, dy], self.C[dx, dy] | 32)

    def test_injection_a(self):
        couples = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        
        for dx, dy in couples:
            f = flauers.fault_models.StuckAt(
                "a",
                x = dx+1, y = dy+1,
                bit = 5, polarity = 1,
                msb = "last"
            )

            self.hw.add_fault(f)

            C = self.hw.matmul(self.A, self.B)

            """
            When we perform this injection, we have a different value in register a (containing values from A).
            This means that rather than doning 1*1 + 1*1 + 1*1, 
            we will have (1 | (1<<5))*1 + (1 | (1<<5))*1 + (1 | (1<<5))*1 = 33 + 33 + 33 = 99 (in 0, 0)
            The same thing happens on the (0, 1) and (0, 2) with different values
            
            """

            for row in range(self.A.shape[0]):
                for col in range(self.B.shape[1]):
                    value = 0
                    for k in range(self.A.shape[1]):
                        if row == dx and col >= dy:    
                            value += ( self.A[row, k] | 32 ) * self.B[k, col]
                        else:
                            value += ( self.A[row, k] ) * self.B[k, col]
                    self.assertEqual(C[row, col], value, 
                        msg=f"assert failed for {row, col} when injecting element {(dx, dy)} (PE: {dx+1, dy+1})"
                    )

            self.hw.clear_all_faults()

    def test_injection_b(self):
        couples = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        
        for dx, dy in couples:
            f = flauers.fault_models.StuckAt(
                "b",
                x = dx+1, y = dy+1,
                bit = 5, polarity = 1,
                msb = "last"
            )

            self.hw.add_fault(f)

            C = self.hw.matmul(self.A, self.B)

            """
            When we perform this injection, we have a different value in register a (containing values from A).
            This means that rather than doning 1*1 + 1*1 + 1*1, 
            we will have (1 | (1<<5))*1 + (1 | (1<<5))*1 + (1 | (1<<5))*1 = 33 + 33 + 33 = 99 (in 0, 0)
            The same thing happens on the (0, 1) and (0, 2) with different values
            
            """

            for row in range(self.A.shape[0]):
                for col in range(self.B.shape[1]):
                    value = 0
                    for k in range(self.A.shape[1]):
                        if row >= dx and col == dy:    
                            value += self.A[row, k] * ( self.B[k, col] | 32 )
                        else:
                            value += self.A[row, k] * ( self.B[k, col] )
                    self.assertEqual(C[row, col], value, 
                        msg=f"assert failed for {row, col} when injecting element {(dx, dy)} (PE: {dx+1, dy+1})"
                    )

            self.hw.clear_all_faults()



class TestSystolicArrayRSInjections(unittest.TestCase):
    def setUp(self):
        self.N1 = 9
        self.N2 = 10
        self.N3 = 11

        self.hw = flauers.SystolicArray(
            self.N1, self.N2, self.N3,
            flauers.projection_matrices.row_stationary,
            in_dtype = np.int8
        )

        self.A = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ], dtype = np.int8)

        self.B = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]
        ], dtype = np.int8)

        # A = (np.random.random((self.N1, self.N3)) * 1e9).astype(np.int8)
        # B = (np.random.random((self.N3, self.N2)) * 1e9).astype(np.int8)

        self.C = self.A.astype(np.int32) @ self.B.astype(np.int32)

    def test_injection_c(self):
        couples = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        
        for dx, dy in couples:
            f = flauers.fault_models.StuckAt(
                "c",
                x = dx+1, y = -(dy+1),
                bit = 5, polarity = 1,
                msb = "last"
            )

            self.hw.add_fault(f)

            C = self.hw.matmul(self.A, self.B)

            """
            When we perform this injection, we have a different value in register c.
            This means that rather than doning [1 1 1] * [1 1 1]' = 3 we will have something
            very different!
            Specifically, the sequence of operations is the following:

            c = 0
            repeat 3-times:
                c += 1 * 1
                c |= 0x1 << 5  # this is because of the injection
            
            So the result will be 3 | (0x1 << 5) = 32 + 3 = 35
            """

            self.assertEqual(C[dx, dy], self.C[dx, dy] + 32)

    def test_injection_a(self):
        couples = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        
        for dx, dy in couples:
            f = flauers.fault_models.StuckAt(
                "a",
                x = dx+1, y = dy+1,
                bit = 5, polarity = 1,
                msb = "last"
            )

            self.hw.add_fault(f)

            C = self.hw.matmul(self.A, self.B)

            """
            When we perform this injection, we have a different value in register a (containing values from A).
            This means that rather than doning 1*1 + 1*1 + 1*1, 
            we will have (1 | (1<<5))*1 + (1 | (1<<5))*1 + (1 | (1<<5))*1 = 33 + 33 + 33 = 99 (in 0, 0)
            The same thing happens on the (0, 1) and (0, 2) with different values
            
            """

            for row in range(self.A.shape[0]):
                for col in range(self.B.shape[1]):
                    value = 0
                    for k in range(self.A.shape[1]):
                        if row == dx and col >= dy:    
                            value += ( self.A[row, k] | 32 ) * self.B[k, col]
                        else:
                            value += ( self.A[row, k] ) * self.B[k, col]
                    self.assertEqual(C[row, col], value, 
                        msg=f"assert failed for {row, col} when injecting element {(dx, dy)} (PE: {dx+1, dy+1})"
                    )

            self.hw.clear_all_faults()

    def test_injection_b(self):
        couples = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        
        for dx, dy in couples:
            f = flauers.fault_models.StuckAt(
                "b",
                x = dx+1, y = dy+1,
                bit = 5, polarity = 1,
                msb = "last"
            )

            self.hw.add_fault(f)

            C = self.hw.matmul(self.A, self.B)

            """
            When we perform this injection, we have a different value in register a (containing values from A).
            This means that rather than doning 1*1 + 1*1 + 1*1, 
            we will have (1 | (1<<5))*1 + (1 | (1<<5))*1 + (1 | (1<<5))*1 = 33 + 33 + 33 = 99 (in 0, 0)
            The same thing happens on the (0, 1) and (0, 2) with different values
            
            """

            for row in range(self.A.shape[0]):
                for col in range(self.B.shape[1]):
                    value = 0
                    for k in range(self.A.shape[1]):
                        if row >= dx and col == dy:    
                            value += self.A[row, k] * ( self.B[k, col] | 32 )
                        else:
                            value += self.A[row, k] * ( self.B[k, col] )
                    self.assertEqual(C[row, col], value, 
                        msg=f"assert failed for {row, col} when injecting element {(dx, dy)} (PE: {dx+1, dy+1})"
                    )

            self.hw.clear_all_faults()

class TestSystolicArrayTiling(unittest.TestCase):

    def setUp(self):
        self.N1 = 2
        self.N2 = 2
        self.N3 = 100
        self.hw = flauers.SystolicArray(
            self.N1, self.N2, self.N3,
            flauers.projection_matrices.output_stationary
        )

        self.A = np.ones( (10, 10), dtype=np.int8)
        self.B = np.ones( (10, 10), dtype=np.int8)
        self.C = self.A.astype(np.int32) @ self.B.astype(np.int32)

    def tearDown(self):
        self.hw.clear_all_faults()

    def test_no_injection_tiling(self):
        self.assertRaises(
            flauers.exceptions.DimensionError,
            self.hw.matmul, self.A, self.B,
        )

        C = self.hw.matmul(self.A, self.B, tiling=True)

        self.assertTrue(
            np.allclose( C, self.C)
        )
    
    def test_injecting_c_tiling(self):
        dx = 0; dy = 0
        
        f = flauers.fault_models.StuckAt(
            "c", 
            x = dx + 1, y = dy + 1,
            bit = 5, polarity = 1,
            msb = "last"
        )
        self.hw.add_fault(f)
    
        C = self.hw.matmul(self.A, self.B, tiling=True)

        for row in range(self.A.shape[0]):
                for col in range(self.B.shape[1]):
                    value = 0
                    for k in range(self.A.shape[1]):
                        if (row%2) == dx and (col%2) == dy:    
                            value += self.A[row, k] * self.B[k, col]
                            value |= 32
                        else:
                            value += self.A[row, k] * self.B[k, col]
                    self.assertEqual(C[row, col], value, 
                        msg=f"assert failed for {row, col} when injecting element {(dx, dy)} (PE: {dx+1, dy+1})"
                    )


