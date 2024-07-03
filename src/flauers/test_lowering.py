import unittest
import numba.cuda
import numpy as np
import torch

from . import lowerings
from .cuda import utils

if numba.cuda.is_available():
    class TestLoweringsCuda(unittest.TestCase):
        def test_full_wrappers(self):
            a = np.array([[1,2,3], [4,5,6], [7, 8, 9]], dtype=np.float32)
            b = np.array([[1,2], [3,4]], dtype=np.float32)

            lolif = lowerings.S_Im2Col(a.shape, b.shape, use_gpu=True)
            a = numba.cuda.to_device(a)
            al = lolif.lower_activation(a)
            print(al.copy_to_host())
            b = numba.cuda.to_device(b)
            bl = lolif.lower_kernel(b)
            print(bl.copy_to_host())
            c = al.copy_to_host()@bl.copy_to_host()
            print(c)
            c = numba.cuda.to_device(c)
            c = lolif.lift(c)

            aT = torch.from_numpy(a.copy_to_host()).view(1, 1, 3, 3).type(torch.float32)
            bT = torch.from_numpy(b.copy_to_host()).view(1, 1, 2, 2).type(torch.float32)
            cT = torch.zeros((2, 2))

            cok = torch.nn.functional.conv2d(aT, bT)
            print(cok)
            self.assertTrue( np.allclose(c, cok) )

        def test_big_wrappers_additive(self):
            input_size = 28
            kernel_size = 5
            rounds = 100

            output_size = input_size - kernel_size + 1

            result = numba.cuda.device_array((output_size, output_size), dtype=np.float32)
            utils.zero_init_matrix[32, 8](result)

            a = np.random.random((input_size, input_size)).astype(np.float32)
            b = np.random.random((kernel_size, kernel_size)).astype(np.float32)

            print(a)

            a = numba.cuda.to_device(a)
            b = numba.cuda.to_device(b)
            
            lolif = lowerings.S_Im2Col(a.shape, b.shape, use_gpu=True)
            for _ in range(rounds):
                al = lolif.lower_activation(a)
                print(al.copy_to_host())
                bl = lolif.lower_kernel(b)
                print(bl.copy_to_host())
                c = al.copy_to_host()@bl.copy_to_host()
                c = numba.cuda.to_device(c)
                lolif.lift_cuda(result, c, True)

            print(f"c is ", c.copy_to_host())
            aT = torch.from_numpy(a.copy_to_host()).view(1, 1, *a.shape).type(torch.float32)
            bT = torch.from_numpy(b.copy_to_host()).view(1, 1, *b.shape).type(torch.float32)
            cT = torch.zeros(result.shape)

            cok = torch.nn.functional.conv2d(aT, bT)
            print(f"cok is ", cok)
            self.assertTrue(np.allclose(result, rounds*cok))

        def test_wrappers_additive(self):
            result = numba.cuda.device_array((2,2), dtype=np.float32)
            utils.zero_init_matrix[32, 8](result)

            a = np.array([[1,2,3], [4,5,6], [7, 8, 9]])
            b = np.array([[1,2], [3,4]])

            a = numba.cuda.to_device(a)
            b = numba.cuda.to_device(b)
            
            lolif = lowerings.S_Im2Col(a.shape, b.shape, use_gpu=True)
            for _ in range(3):
                al = lolif.lower_activation(a)
                print(al.copy_to_host())
                bl = lolif.lower_kernel(b)
                print(bl.copy_to_host())
                c = al.copy_to_host()@bl.copy_to_host()
                c = numba.cuda.to_device(c)
                lolif.lift_cuda(result, c, True)

            print("c is ", c.copy_to_host())
            aT = torch.from_numpy(a.copy_to_host()).view(1, 1, 3, 3).type(torch.float32)
            bT = torch.from_numpy(b.copy_to_host()).view(1, 1, 2, 2).type(torch.float32)
            cT = torch.zeros((2, 2))

            cok = torch.nn.functional.conv2d(aT, bT)
            print("cok is ", cok)
            self.assertTrue(np.allclose(result, 3*cok))