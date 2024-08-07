import numpy as np

import flauers as si
from flauers import projection_matrices as pm
import torch
import unittest
import logging

# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARNING)  
    
class Tests(unittest.TestCase):

    def test_matmul(self):
        print("Running test_matmul")
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])

        c_sa = si.matmul(a, b)
        c_np = np.matmul(a, b)

        print(c_sa)
        print(c_np)

        result = c_sa == c_np
        self.assertTrue(result.all())

    def test_convolve(self):
        print("Running test_convolve")
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # a = np.ones((3,3))
        b = np.array([[1, 2], [3, 4]])

        c_sa = si.convolve(a, b)
        print(c_sa)

        aT = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
        bT = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))

        c_torch = torch.nn.functional.conv2d(aT, bT)

        print(c_torch)

        result = c_sa == np.array(c_torch)
        self.assertTrue(result.all())

    def test_convolve_with_explicit_instantiation(self):
        print("Running test_convolve_with_explicit_instantiation")
        # input data
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # a = np.ones((3,3))
        b = np.array([[1, 2], [3, 4]])

        # Add a dimension for the channels
        a = np.expand_dims(a, 0)
        b = np.expand_dims(b, 0)

        # Instantiate the systolic array with physical parameters N1, N2, N3
        hw = si.SystolicArray(28, 28, 28, pm.output_stationary)

        print("######## Slim Kernel!")
        transformation = si.lowerings.C_SlimKernel(a.shape, b.shape)  # Define the lowering

        print("A is")
        print(transformation.lower_activation(a))
        print("B is")
        print(transformation.lower_kernel(b))

        print("######## Im2Col!")
        a_im2col = a.squeeze(0)
        b_im2col = b.squeeze(0)
        transformation = si.lowerings.S_Im2Col(a_im2col.shape, b_im2col.shape)  # Define the lowering

        print("A is")
        print(transformation.lower_activation(a_im2col))
        print("B is")
        print(transformation.lower_kernel(b_im2col))

        # Compute the matrix multiplication between the lowered inputs
        c_sa = hw.matmul(transformation.lower_activation(a_im2col), transformation.lower_kernel(b_im2col))
        c_sa = transformation.lift(c_sa)  # lift the result
        print("Expected: ")
        print(c_sa)

        # Perform the convolution with pytorch
        aT = torch.from_numpy(a).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
        bT = torch.from_numpy(b).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
        c_torch = torch.nn.functional.conv2d(aT, bT)
        print("Ground Truth:")
        print(c_torch)

        result = c_sa == np.array(c_torch)
        self.assertTrue(result.all())

    def test_convolution_with_im2col(self):
        print("Running test_convolution_with_im2col")
        # input data
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = np.array([[1, 2], [3, 4]])

        # Instantiate the systolic array with physical parameters N1, N2, N3
        hw = si.SystolicArray(28, 28, 28, pm.output_stationary)

        transformation = si.lowerings.S_Im2Col(a.shape, b.shape)  # Define the lowering

        print("A is")
        print(transformation.lower_activation(a))
        print("B is")
        print(transformation.lower_kernel(b))

        # Perform the actual multiplication
        c_sa = hw.matmul(transformation.lower_activation(a), transformation.lower_kernel(b))
        c_sa = transformation.lift(c_sa)  # lift the result
        print("Expected: ")
        print(c_sa)

        # Perform the convolution with pytorch
        aT = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
        bT = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
        c_torch = torch.nn.functional.conv2d(aT, bT)
        print("Ground Truth:")
        print(c_torch)
        result = c_sa == np.array(c_torch)
        self.assertTrue(result.all())

    def test_convolve_with_array(self):
        print("Running test_convolve_with_array")
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # a = np.ones((3,3))
        b = np.array([[1, 2], [3, 4]])

        array = si.SystolicArray(10, 10, 10, si.projection_matrices.output_stationary)
        c_sa = si.convolve_with_array(a, b, lowering=si.lowerings.S_Im2Col, array=array)
        print(c_sa)

        aT = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
        bT = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))

        c_torch = torch.nn.functional.conv2d(aT, bT)

        print(c_torch)

        result = c_sa == np.array(c_torch)
        self.assertTrue(result.all())

    def test_weird_test(self):
        def test_weird_test():
            a = np.array([[0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0., 0., 0., 0.],
                    [0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,     0., 0., 0., 0.],
                    [0., 27113., 27404.,     0.,     0.,     0.,     0.,     0.,     0., 0., 0., 0.],
                    [0., 35910., 29883.,     0.,     0.,     0.,     0.,     0.,     0., 0., 0., 0.],
                    [0.,  9691.,  5909.,   279.,  5322.,     0.,     0., 24430., 17666., 0., 0., 0.],
                    [0.,     0.,     0.,    81.,   411.,     0., 18986., 38559., 13138., 0., 0., 0.],
                    [0.,     0.,     0.,     0.,     0.,     0., 39196., 40244.,     0., 0., 0., 0.],
                    [0.,     0.,     0.,     0.,     0., 16462., 36795.,  8109.,     0., 0., 0., 0.],
                    [0.,     0.,     0.,     0., 10893., 38902., 31807.,     0.,     0., 0., 0., 0.],
                    [0.,     0.,     0.,   783., 33197., 37076.,     0.,     0.,     0., 0., 0., 0.],
                    [0.,     0.,     0., 19453., 39297., 16767.,     0.,     0.,     0., 0., 0., 0.],
                    [0.,     0.,     0., 46700., 51064.,     0.,     0.,     0.,     0., 0., 0., 0.]])
            b = np.array([[-102.,  -36.,  17.,  17., -21.],
                        [   3., -127., -25., -42.,  -7.],
                        [  64.,  -30., -48., -73., -70.],
                        [  63.,   21.,  87.,  43.,  -6.],
                        [   5.,  -62., -28., -38.,   1.]])

            array = si.SystolicArray(100, 100, 150, si.projection_matrices.output_stationary, in_dtype=np.dtype(np.int8) )
            c_sa = si.convolve_with_array(a, b, lowering=si.lowerings.S_Im2Col, array=array)
            # print(c_sa)
            # print(c_sa.dtype)

            aT = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.int8))
            bT = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.int8))

            c_torch = torch.nn.functional.conv2d(aT, bT)

            result = c_sa == np.array(c_torch)
        self.assertRaises(si.exceptions.CastingError, test_weird_test)

    def test_injection_simple(self):
        print("Running injection_simple")
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # b = np.ones((2,2))
        b = np.array([[10, 11], [12, 13]])

        array = si.SystolicArray(10, 10, 10,
                                si.projection_matrices.col_stationary,
                                in_dtype=np.dtype(np.int8))
        f = si.fault_models.StuckAt("c", x=1, y=1, bit=0, polarity=1, msb="first")
        array.add_fault(f)
        c_sa = si.convolve_with_array(a, b, lowering=si.lowerings.S_Im2Col, array=array)

        aT = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
        bT = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))

        c_torch = torch.nn.functional.conv2d(aT, bT)

        print(c_sa)
        print(c_torch)
        print(c_sa == np.array(c_torch) )
        self.assertTrue(True)

    def test_weight_stationary(self):
        a = np.ones( (28, 28) )
        b = np.ones((5, 5))

        hw = si.SystolicArray(
            n1 = 25, n2 = 25, n3 = 141,
            T = si.projection_matrices.output_stationary
        )
        c_sa = si.convolve_with_array(a, b, array= hw,lowering=si.lowerings.S_Im2Col)
        print(c_sa)
        self.assertTrue(True)

    def test_old_new_method(self):
        print("Running test_old_new_method")
        # a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # b = np.ones((2,2))
        # b = np.array([[10, 11], [12, 13]])

        a = np.ones((2, 5))
        b = np.ones((5, 3))

        array = si.SystolicArray(10, 10, 10,
                                si.projection_matrices.output_stationary,
                                in_dtype=np.dtype(np.int8), use_legacy=True )
        f = si.fault_models.StuckAt("c", x=1, y=1, bit=0, polarity=1, msb="first")
        array.add_fault(f)
        c_old = array.matmul(a, b)

        array = si.SystolicArray(10, 10, 10,
                                si.projection_matrices.output_stationary,
                                in_dtype=np.dtype(np.int8), use_legacy=False)
        f = si.fault_models.StuckAt("c", x=1, y=1, bit=0, polarity=1, msb="first")
        array.add_fault(f)

        self.assertRaises(NotImplementedError, array.matmul, a, b,)
        return

        print("old_method:\n", c_old)
        print("new_method:\n", c_new)
        a, b = c_old, c_new
        self.assertTrue( (a == b).all() )
