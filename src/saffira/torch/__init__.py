import numpy as np

import saffira as si
import saffira.torch; si.torch = saffira.torch # consistent naming
from saffira import projection_matrices as pm
import torch
import torchvision
import unittest
import logging

si.USE_CORE = True

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.WARNING)


def test_matmul() -> np.ndarray:
    print("Running test_matmul")
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    c_sa = si.matmul(a, b)
    c_np = np.matmul(a, b)

    print(c_sa)
    print(c_np)

    result = c_sa == c_np
    return result


def test_convolve() -> np.ndarray:
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

    return c_sa == np.array(c_torch)


def test_convolve_with_explicit_instantiation() -> np.ndarray:
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

    # Perform the convolution with torch
    aT = torch.from_numpy(a).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    bT = torch.from_numpy(b).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    c_torch = torch.nn.functional.conv2d(aT, bT)
    print("Ground Truth:")
    print(c_torch)

    return c_sa == np.array(c_torch)


def test_convolution_with_im2col() -> np.ndarray:
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

    # Perform the convolution with torch
    aT = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    bT = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    c_torch = torch.nn.functional.conv2d(aT, bT)
    print("Ground Truth:")
    print(c_torch)

    return c_sa == np.array(c_torch)


def test_convolve_with_array() -> np.ndarray:
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

    return c_sa == np.array(c_torch)


def test_weird_test() -> np.ndarray:
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

    array = si.SystolicArray(100, 100, 150, si.projection_matrices.output_stationary)
    c_sa = si.convolve_with_array(a, b, lowering=si.lowerings.S_Im2Col, array=array)
    # print(c_sa)
    # print(c_sa.dtype)

    aT = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    bT = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))

    c_torch = torch.nn.functional.conv2d(aT, bT)

    # print(c_torch)

    return c_sa == np.array(c_torch)


def test_injection_simple() -> np.ndarray:
    print("Running injection_simple")
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # b = np.ones((2,2))
    b = np.array([[10, 11], [12, 13]])

    array = si.SystolicArray(10, 10, 10, si.projection_matrices.output_stationary, in_dtype=np.dtype(np.int8))
    f = si.fault_models.StuckAt("c", x=1, y=1, bit=0, polarity=1, msb="first")
    array.add_fault(f)
    c_sa = si.convolve_with_array(a, b, lowering=si.lowerings.S_Im2Col, array=array)

    aT = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    bT = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))

    c_torch = torch.nn.functional.conv2d(aT, bT)

    print(c_sa)
    print(c_torch)
    print(c_sa == np.array(c_torch) )

def test_lenet():

    nn = torch.nn

    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.cnv1 = nn.Conv2d(1, 6, 5, stride=1)
            self.relu1 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2, 2)
            self.cnv3 = nn.Conv2d(6, 16, 5, stride=1)
            self.relu3 = nn.ReLU()
            self.pool4 = nn.MaxPool2d(2, 2)
            self.fc5 = nn.Linear(256, 120)
            self.relu5 = nn.ReLU()
            self.fc6 = nn.Linear(120, 84)
            self.relu6 = nn.ReLU()
            self.fc7 = nn.Linear(84, 10)
            self.softmax = nn.Softmax()

        def forward(self, image):
            self.out1 = self.relu1(self.cnv1(image))
            self.out2 = self.pool2(self.out1)
            self.out3 = self.relu3(self.cnv3(self.out2))
            self.out4 = self.pool4(self.out3)
            self.out4_flt = torch.flatten(self.out4.permute(0, 2, 3, 1), 1)
            self.out5 = self.relu5(self.fc5(self.out4_flt))
            self.out6 = self.relu6(self.fc6(self.out5))
            self.out7 = self.fc7(self.out6)

            return self.out7, self.out6

        def forward_rescaling(self, image, bit):
            self.out1 = self.rescaling(self.relu1(self.cnv1(image)), bit)
            self.out2 = self.pool2(self.out1)
            self.out3 = self.rescaling(self.relu3(self.cnv3(self.out2)), bit)
            self.out4 = self.pool4(self.out3)
            self.out4_flt = torch.flatten(self.out4.permute(0, 2, 3, 1), 1)
            self.out5 = self.rescaling(self.relu5(self.fc5(self.out4_flt)), bit)
            self.out6 = self.rescaling(self.relu6(self.fc6(self.out5)), bit)
            self.out7 = self.rescaling(self.fc7(self.out6), bit)

            return self.out7, self.out6

        def load_params(self, addr, device):
            b0 = np.float32(np.load(addr + "conv2d_1_weights.npy"))
            b0 = np.transpose(b0, (3, 2, 0, 1))
            b1 = np.float32(np.load(addr + "conv2d_1_bias.npy")).reshape(6)

            b2 = np.float32(np.load(addr + "conv2d_2_weights.npy"))
            b2 = np.transpose(b2, (3, 2, 0, 1))
            b3 = np.float32(np.load(addr + "conv2d_2_bias.npy")).reshape(16)

            b4 = np.float32(np.load(addr + "dense1_weights.npy")).T
            b5 = np.float32(np.load(addr + "dense1_bias.npy")).reshape(120)

            b6 = np.float32(np.load(addr + "dense2_weights.npy")).T
            b7 = np.float32(np.load(addr + "dense2_bias.npy")).reshape(84)

            b8 = np.float32(np.load(addr + "dense3_weights.npy")).T
            b9 = np.float32(np.load(addr + "dense3_bias.npy")).reshape(10)

            self.cnv1.weight = nn.Parameter(torch.from_numpy(b0).to(device), requires_grad=False)  # .type(torch.int8)
            self.cnv1.bias = nn.Parameter(torch.from_numpy(b1).to(device), requires_grad=False)  # .type(torch.int8)

            self.cnv3.weight = nn.Parameter(torch.from_numpy(b2).to(device), requires_grad=False)  # .type(torch.int8)
            self.cnv3.bias = nn.Parameter(torch.from_numpy(b3).to(device), requires_grad=False)  # .type(torch.int8)

            self.fc5.weight = nn.Parameter(torch.from_numpy(b4).to(device), requires_grad=False)  # .type(torch.int8)
            self.fc5.bias = nn.Parameter(torch.from_numpy(b5).to(device), requires_grad=False)  # .type(torch.int8)

            self.fc6.weight = nn.Parameter(torch.from_numpy(b6).to(device), requires_grad=False)  # .type(torch.int8)
            self.fc6.bias = nn.Parameter(torch.from_numpy(b7).to(device), requires_grad=False)  # .type(torch.int8)

            self.fc7.weight = nn.Parameter(torch.from_numpy(b8).to(device), requires_grad=False)  # .type(torch.int8)
            self.fc7.bias = nn.Parameter(torch.from_numpy(b9).to(device), requires_grad=False)  # .type(torch.int8)

        def rescaling(self, in_activation, bit):
            size = in_activation.size()
            batch = size[0]
            if in_activation.dim() == 4:
                features = size[1] * size[2] * size[3]
            else:
                features = size[1]

            in_act_cp = torch.reshape(in_activation, (batch, features))

            min_act, _ = torch.min(in_act_cp, 1)
            max_act, _ = torch.max(in_act_cp, 1)

            scaling_factor = (max_act - min_act) / ((2 ** bit - 1) / 4)
            scaling_factor = torch.unsqueeze(scaling_factor, 1)

            out_activation = torch.floor(torch.round(in_act_cp / scaling_factor))
            out_activation = torch.nan_to_num(out_activation)
            out_activation = torch.reshape(out_activation, in_activation.size())

            return out_activation

    my_device = torch.device("cpu")
    hw = si.SystolicArray(28, 28, 150,
                           si.projection_matrices.output_stationary,
                           in_dtype=np.dtype(np.int8)   )
    systolic_net = LeNet().to( my_device )


    # IMPORTANT: do the parameter loading before the replacement of the layer!!!
    addr = r'./params/'
    systolic_net.load_params(addr, my_device)

    indeces = si.torch.get_conv2d_layer_indices(systolic_net)
    for index in indeces:
        si.torch.replace_conv2d_layer(systolic_net, index, hw)

    print("The neural network is")
    print(systolic_net)

    batch_size = 500

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    correct_systolic = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data[0].to(my_device), data[1].to(my_device)
            images = torch.floor(images * 127)

            total += labels.size(0)

            outputs, _ = systolic_net.forward_rescaling(images, 8)

            _, predicted = torch.max(outputs.data, 1)
            correct_systolic += (predicted == labels).sum().item()
            systolic_accuracy = correct_systolic / total
            print(f"[Systolic] correct one batch: {100 * systolic_accuracy}%")
            break

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

    def test_convolve_with_array(self):
        result = test_convolve_with_array()
        self.assertTrue(result.all())

    def test_weird_test(self):
        self.assertRaises(si.exceptions.CastingError, test_weird_test)

    def test_injection_simple(self):
        test_injection_simple()
        self.assertTrue(True)


if __name__ == "__main__":
    # while True:
    #test_lenet()
    test_convolve()
    exit(0)
    try:
        test_weird_test()
    except Exception as e:
        print("So, we failed!")
        print("This is the exception!")
        print(e)
        print(type(e))
        print("DONE")
