from . import matmuls_int
from . import matmuls_float
from . import lowerings
from . import utils

from numba import cuda
import unittest
import numpy as np
import random
import torch

if cuda.is_available():

    class TestLowerings(unittest.TestCase):
        def test_easy(self):
            a = np.array([[1,2,3], [4,5,6], [7, 8, 9]])
            b = np.array([[1,2], [3,4]])

            al = cuda.device_array((2, 6), dtype=np.float32)
            lowerings.S_Im2Col_lower_activation[32,8](a, al, b.shape[0], False)
            print(al.copy_to_host())
            bl = cuda.device_array((6,2), dtype=np.float32)
            lowerings.S_Im2Col_lower_kernel[32,8](b, bl, b.shape[0], False)
            print(bl.copy_to_host())
            c = al.copy_to_host()@bl.copy_to_host()
            print(c)

            aT = torch.from_numpy(a).view(1, 1, 3, 3).type(torch.float32)
            bT = torch.from_numpy(b).view(1, 1, 2, 2).type(torch.float32)
            cT = torch.zeros((2, 2))

            cok = torch.nn.functional.conv2d(aT, bT)
            print(cok)
            self.assertTrue( np.allclose(c, cok) )

        def test_additive(self):
            result = cuda.device_array((2,2), dtype=np.float32)
            utils.zero_init_matrix[32, 8](result)
            for _ in range(3):
                a = np.array([[1,2,3], [4,5,6], [7, 8, 9]])
                b = np.array([[1,2], [3,4]])

                al = cuda.device_array((2, 6), dtype=np.float32)
                lowerings.S_Im2Col_lower_activation[32,8](al, a, b.shape[0], False)
                print(al.copy_to_host())
                bl = cuda.device_array((6,2), dtype=np.float32)
                lowerings.S_Im2Col_lower_kernel[32,8](bl, b, b.shape[0], False)
                print(bl.copy_to_host())
                c = al.copy_to_host()@bl.copy_to_host()
                c = cuda.to_device(c)
                lowerings.S_Im2Col_lift[32, 8](result, c, True)

            print("c is ", c.copy_to_host())
            aT = torch.from_numpy(a).view(1, 1, 3, 3).type(torch.float32)
            bT = torch.from_numpy(b).view(1, 1, 2, 2).type(torch.float32)
            cT = torch.zeros((2, 2))

            cok = torch.nn.functional.conv2d(aT, bT)
            print("cok is ", cok)
            self.assertTrue(np.allclose(result, 3*cok))
    


    class TestMatmulsInt(unittest.TestCase):
        def test_noinjection_i32(self):
            dtypes = {
                np.int8: np.int32,
                np.int16: np.int64
                }

            N1 = 10
            N2 = 10
            N3 = 10

            scale = 1e9

            dtype = np.int8
            out_dtype = np.int32
            
            for intype in [0, 1, 2]:
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

                # arrays to cuda
                A = cuda.to_device(A)
                B = cuda.to_device(B)
                C = cuda.device_array((N1, N2), dtype=out_dtype)
                matmuls_int.injected_matmul_old_int8[32,32](
                    A, B, C,
                    ina, inb, inc,
                    np.int8(intype)
                )

                C = C.copy_to_host()
                print(C)
                print(Cok)
                self.assertTrue( np.allclose(C, Cok), 
                    msg = f"Wrong! A:\n{A.copy_to_host()}\n"
                    f"B:\n{B.copy_to_host()}\n"
                    f"C:\n{C}\nCok:\n{Cok}"
                )

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
                    matmuls_int.injected_matmul_old_int8[128, 128](
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
                    matmuls_int.injected_matmul_old_int8[128, 128](
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
                # np.int16: np.int64
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
                    C = cuda.to_device(C)
                    matmuls_int.injected_matmul_old_int8[128, 128](
                        A, B, C,
                        ina, inb, inc,
                        1
                    )
                    C = C.copy_to_host()

                    self.assertNotEqual(
                        C[x, y] & (out_dtype(0x1) << shift),
                        0,
                        msg = f"C[x, y] is {C[x, y]} - faulty C[x,y] is supposed to be {C[x, y] & (out_dtype(0x1) << shift)}, Cok[x,y] is {Cok[x, y]}"
                    )

                    self.assertLessEqual(
                        np.invert((C == Cok)).sum(),
                        1,
                        msg = f"A:\n{A}\n"
                                f"B:\n{B}\n"
                                f"C:\n{C}\n"
                                f"Cok:\n{Cok}"
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
                matmuls_float.injected_matmul_old_float32[N1, N2](
                    A, B, C,
                    ina, inb, inc,
                    intype
                )

                print(C)
                print(Cok)
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
                matmuls_float.injected_matmul_old_float32[128, 100](
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
                matmuls_float.injected_matmul_old_float32[128, 100](
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
