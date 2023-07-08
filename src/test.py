import numpy as np

import systolic_injector as si
from systolic_injector import systolic_array as sa
from systolic_injector import projection_matrices as pm
import torch
import unittest


def test_matmul():
    print("Running test_matmul")
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    c_sa = si.matmul(a, b)
    c_np = np.matmul(a, b)

    result = c_sa == c_np
    return result


def test_convolve():
    print("Running test_convolve")
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # a = np.ones((3,3))
    b = np.array([[1, 1], [1, 1]])

    a = np.expand_dims(a, 0)
    b = np.expand_dims(b, 0)
    c_sa = si.convolve(a, b)

    aT = torch.from_numpy(a).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    bT = torch.from_numpy(b).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))

    c_torch = torch.nn.functional.conv2d(aT, bT)

    return c_sa == np.array(c_torch)


def test_convolve_with_explicit_instantiation():
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
    h = []  # Prepare history

    print("######## Slim Kernel!")
    transformation = si.lowerings.SlimKernel(a.shape, b.shape)  # Define the lowering

    print("A is")
    print(transformation.lower_activation(a))
    print("B is")
    print(transformation.lower_kernel(b))

    print("######## Im2Col!")
    transformation = si.lowerings.Im2Col(a.shape, b.shape)  # Define the lowering

    print("A is")
    print(transformation.lower_activation(a))
    print("B is")
    print(transformation.lower_kernel(b))

    # Compute the matrix multiplication between the lowered inputs
    c_sa = hw.matmul(transformation.lower_activation(a), transformation.lower_kernel(b), h)
    c_sa = transformation.lift(c_sa)  # lift the result
    print("Expected: ")
    print(c_sa)

    # Perform the convolution with pytorch
    aT = torch.from_numpy(a).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    bT = torch.from_numpy(b).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    c_torch = torch.nn.functional.conv2d(aT, bT)
    print("Ground Truth:")
    print(c_torch)

    return c_sa == np.array(c_torch)

def test_convolution_with_im2col():
    print("Running test_convolution_with_im2col")
    # input data
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[1, 2], [3, 4]])

    # Add a dimension for the channels
    a = np.expand_dims(a, 0)
    b = np.expand_dims(b, 0)

    # Instantiate the systolic array with physical parameters N1, N2, N3
    hw = si.SystolicArray(28, 28, 28, pm.output_stationary)
    h = []  # Prepare history

    transformation = si.lowerings.Im2Col(a.shape, b.shape)  # Define the lowering

    print("A is")
    print(transformation.lower_activation(a))
    print("B is")
    print(transformation.lower_kernel(b))

    # Perform the actual multiplication
    c_sa = hw.matmul(transformation.lower_activation(a), transformation.lower_kernel(b), h)
    c_sa = transformation.lift(c_sa)  # lift the result
    print("Expected: ")
    print(c_sa)

    # Perform the convolution with pytorch
    aT = torch.from_numpy(a).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    bT = torch.from_numpy(b).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    c_torch = torch.nn.functional.conv2d(aT, bT)
    print("Ground Truth:")
    print(c_torch)

    return c_sa == np.array(c_torch)


class Tests(unittest.TestCase):

    def test_matmul(self):
        result = test_matmul()
        self.assertTrue(result.all())

    def test_convolve(self):
        result = test_convolve()
        self.assertTrue(result.all())

    def test_convolve_with_explicit_instantiation(self):
        result = test_convolve_with_explicit_instantiation()
        self.assertTrue(result.all())

    def test_convolution_with_im2col(self):
        result = test_convolution_with_im2col()
        self.assertTrue(result.all())


if __name__ == "__main__":
    result = test_matmul()
    print(result)

    result = test_convolve()
    print(result)

    result = test_convolve_with_explicit_instantiation()
    print(result)
