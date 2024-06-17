from . import injects
from . import mamtuls_int
from . import matmuls_float

import unittest
import numpy as np
import random

class TestBasicInjections(unittest.TestCase):
    
    def test_inject32(self):
        value = np.int32(0xFFFFFF)
        for i in range(32): # Stuck at 0
            bitstring = np.int32(0x1 << i)
            res = injects.inject_32(value, bitstring, 0)
            self.assertEqual(res, value & ~bitstring)

        value = np.int32(0)
        for i in range(32): # Stuck at 1
            bitstring = np.int32(0x1 << i)
            res = injects.inject_32(value, bitstring, 1)
            self.assertEqual(res, value | bitstring)
        
        value = np.int32(0xFFFFFF)
        for i in range(32): # bit flip
            bitstring = np.int32(0x1 << i)
            res = injects.inject_32(value, bitstring, 2)
            self.assertEqual(res, value ^ bitstring, 
                msg=f"Doesn't hold for value {value}, bitstring {bitstring}, bit {i}"
            )

        for i in range(100):
            bitstring = random.randint(-2**31, 2**31)
            res = injects.inject_32(value, bitstring, 3)
            self.assertEqual(res, bitstring)

    def test_inject64(self):
        value = np.int64(~0x0)
        for i in range(64): # Stuck at 0
            bitstring = np.int64(0x1) << i
            res = injects.inject_64(value, bitstring, 0)
            self.assertEqual(res, value & ~bitstring)

        value = np.int32(0)
        for i in range(64): # Stuck at 1
            bitstring = np.int64(0x1) << i
            res = injects.inject_64(value, bitstring, 1)
            self.assertEqual(res, value | bitstring)
        
        value = np.int32(~0x0)
        for i in range(64): # bit flip
            bitstring = np.int64(0x1) << i
            res = injects.inject_64(value, bitstring, 2)
            self.assertEqual(res, value ^ bitstring, 
                msg=f"Doesn't hold for value {value}, bitstring {bitstring}, bit {i}"
            )

        for i in range(100):
            bitstring = random.randint(-2**31, 2**31)
            res = injects.inject_64(value, bitstring, 3)
            self.assertEqual(res, bitstring)

    def test_inject_int(self):
        dtypes = [np.int8, np.int16, np.int32, np.int64]

        for dtype in dtypes:
            bits = np.dtype(dtype).itemsize * 8

            value = dtype(~0x0)
            for i in range(bits): # Stuck at 0
                bitstring = dtype(dtype(0x1) << i)
                res = injects.inject_int(value, bitstring, 0)
                self.assertEqual(res, value & ~bitstring)

            value = dtype(0)
            for i in range(bits): # Stuck at 1
                bitstring = dtype(dtype(0x1) << i)
                res = injects.inject_int(value, bitstring, 1)
                self.assertEqual(res, value | bitstring)
            
            value = dtype(~0x0)
            for i in range(bits): # bit flip
                bitstring = dtype(dtype(0x1) << i)
                res = injects.inject_int(value, bitstring, 2)
                self.assertEqual(res, value ^ bitstring, 
                    msg=f"Doesn't hold for value {value}, bitstring {bitstring}, bit {i}"
                )

            for i in range(100):
                bitstring = random.randint(-2**(bits-1), 2**(bits-1) )
                res = injects.inject_int(value, bitstring, 3)
                self.assertEqual(res, bitstring)

class TestMatmulsInt(unittest.TestCase):
    def test_noinjection(self):
        dtypes = {
            np.int8: np.int32,
            np.int16: np.int64
            }

        N1 = 10
        N2 = 10
        N3 = 10

        scale = 1e9
        
        for intype in [0, 1, 2]:
            for dtype, out_dtype in dtypes.items():
                ina = np.zeros((N1, N2, N3), dtype = dtype)
                inb = ina
                inc = ina

                A = np.array(
                        np.random.rand(N1,N3) * scale,
                        dtype=dtype
                    )
                B = np.array(
                        np.random.rand(N3,N2) * scale,
                        dtype=dtype
                    )

                Cok = A.astype(out_dtype)@B.astype(out_dtype)

                C = np.zeros((N1, N2), dtype=out_dtype)
                mamtuls_int.injected_matmul_old_int(
                    A, B, C,
                    ina, inb, inc,
                    intype
                )

                self.assertTrue( np.allclose(C, Cok) )

    def test_random_injection_c(self):
        dtypes = {
            np.int8: np.int32,
            np.int16: np.int64
            }

        N1 = 10
        N2 = 10
        N3 = 10

        scale = 1e9
        
        for intype in [0, 1, 2]:
            for dtype, out_dtype in dtypes.items():
                cbits = np.dtype(out_dtype).itemsize * 8
                
                A = np.array(
                        np.random.rand(N1,N3) * scale,
                        dtype=dtype
                    )
                B = np.array(
                        np.random.rand(N3,N2) * scale,
                        dtype=dtype
                    )

                ina = np.zeros((N1, N2, N3), dtype = dtype)
                inb = np.zeros((N1, N2, N3), dtype = dtype)
                inc = np.zeros((N1, N2, N3), dtype = dtype)

                x = random.randrange(0, N1)
                y = random.randrange(0, N2)
                inc[x, y, :] = out_dtype(0x1) << random.randrange(0, cbits)

                Cok = A.astype(out_dtype) @ B.astype(out_dtype)

                C = np.zeros((N1, N2), dtype=out_dtype)
                mamtuls_int.injected_matmul_old_int(
                    A, B, C,
                    ina, inb, inc,
                    intype
                )

                self.assertLessEqual(
                    np.invert((C == Cok)).sum(),
                    1
                )

    def test_sa0(self):
        dtypes = {
            np.int8: np.int32,
            np.int16: np.int64
            }

        N1 = 10
        N2 = 10
        N3 = 10

        for _ in range(1000):
            for dtype, out_dtype in dtypes.items():
                cbits = np.dtype(out_dtype).itemsize * 8

                A = np.array(
                        -1 * np.ones((N1,N3)),
                        dtype=dtype
                    )
                B = np.array(
                        -1 * np.ones((N3,N2)),
                        dtype=dtype
                    )

                ina = np.zeros((N1, N2, N3), dtype = dtype)
                inb = np.zeros((N1, N2, N3), dtype = dtype)
                inc = np.zeros((N1, N2, N3), dtype = out_dtype)

                x = random.randrange(0, N1)
                y = random.randrange(0, N2)
                shift = random.randrange(0, cbits)
                inc[x, y, -1] = out_dtype(0x1) << shift

                Cok = A.astype(out_dtype) @ B.astype(out_dtype)

                C = np.zeros((N1, N2), dtype=out_dtype)
                mamtuls_int.injected_matmul_old_int(
                    A, B, C,
                    ina, inb, inc,
                    0
                )

                self.assertEqual(
                    C[x, y] & (out_dtype(0x1) << shift),
                    0
                )

                self.assertLessEqual(
                    np.invert((C == Cok)).sum(),
                    1
                )

    def test_sa1(self):
        dtypes = {
            np.int8: np.int32,
            np.int16: np.int64
            }

        N1 = 10
        N2 = 10
        N3 = 10
        
        for _ in range(1000):
            for dtype, out_dtype in dtypes.items():
                cbits = np.dtype(out_dtype).itemsize * 8

                A = np.array(
                        -1 * np.ones((N1,N3)),
                        dtype=dtype
                    )
                B = np.array(
                        1 * np.ones((N3,N2)),
                        dtype=dtype
                    )

                ina = np.zeros((N1, N2, N3), dtype = dtype)
                inb = np.zeros((N1, N2, N3), dtype = dtype)
                inc = np.zeros((N1, N2, N3), dtype = out_dtype)

                x = random.randrange(0, N1)
                y = random.randrange(0, N2)
                shift = random.randrange(0, cbits)
                inc[x, y, -1] = out_dtype(0x1) << shift

                Cok = A.astype(out_dtype) @ B.astype(out_dtype)

                C = np.zeros((N1, N2), dtype=out_dtype)
                mamtuls_int.injected_matmul_old_int(
                    A, B, C,
                    ina, inb, inc,
                    1
                )

                self.assertNotEqual(
                    C[x, y] & (out_dtype(0x1) << shift),
                    0
                )

                self.assertLessEqual(
                    np.invert((C == Cok)).sum(),
                    1
                )

class TestMatmulsFloat32(unittest.TestCase):

    def test_float32_noinjection(self):
        N1 = 10
        N2 = 10
        N3 = 10

        scale = 100
        
        for intype in [0, 1, 2]:
            ina = np.zeros((N1, N2, N3), dtype = np.int32)
            inb = ina
            inc = ina

            A = np.array(
                    np.random.rand(N1,N3) * scale,
                    dtype=np.float32
                )
            B = np.array(
                    np.random.rand(N3,N2) * scale,
                    dtype=np.float32
                )

            Cok = A @ B

            C = np.zeros((N1, N2), dtype=np.float32)
            matmuls_float.injected_matmul_old_f32(
                A, B, C,
                ina, inb, inc,
                intype
            )

            self.assertTrue( np.allclose(C, Cok) )

    def test_float32_sa0(self):

        N1 = 10
        N2 = 10
        N3 = 10

        for _ in range(1000):
            cbits = 32

            A = np.array(
                    np.random.random((N1, N3)) * 100,
                    dtype=np.float32
                )
            B = np.array(
                    np.random.random((N3, N2)) * 100,
                    dtype=np.float32
                )

            ina = np.zeros((N1, N2, N3), dtype = np.int32)
            inb = np.zeros((N1, N2, N3), dtype = np.int32)
            inc = np.zeros((N1, N2, N3), dtype = np.int32)

            x = random.randrange(0, N1)
            y = random.randrange(0, N2)
            shift = random.randrange(0, cbits)
            inc[x, y, -1] = np.int32(0x1) << shift

            Cok = A@B

            C = np.zeros((N1, N2), dtype=np.float32)
            matmuls_float.injected_matmul_old_f32(
                A, B, C,
                ina, inb, inc,
                0
            )

            value = C.view(np.int32)[x, y]
            self.assertEqual(
                value & (np.int32(0x1) << shift),
                0
            )

            self.assertLessEqual(
                np.invert( np.isclose(C, Cok) ).sum(),
                1
            )

    def test_float32_sa1(self):
        N1 = 10
        N2 = 10
        N3 = 10
        
        for _ in range(1000):
            cbits = 32

            A = np.array(
                    np.random.random((N1, N3)) * 100,
                    dtype=np.float32
                )
            B = np.array(
                    np.random.random((N3, N2)) * 100,
                    dtype=np.float32
                )

            ina = np.zeros((N1, N2, N3), dtype = np.int32)
            inb = np.zeros((N1, N2, N3), dtype = np.int32)
            inc = np.zeros((N1, N2, N3), dtype = np.int32)

            x = random.randrange(0, N1)
            y = random.randrange(0, N2)
            shift = random.randrange(0, cbits)
            inc[x, y, -1] = np.int32(0x1) << shift

            Cok = A @ B

            C = np.zeros((N1, N2), dtype=np.float32)
            matmuls_float.injected_matmul_old_f32(
                A, B, C,
                ina, inb, inc,
                1
            )

            value = C.view(np.int32)[x, y]
            self.assertNotEqual(
                value & (np.int32(0x1) << shift),
                0
            )

            self.assertLessEqual(
                np.invert( np.isclose(C, Cok) ).sum(),
                1
            )

