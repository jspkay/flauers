from systolic_injector import *
from systolic_injector import systolic_array as sa
from systolic_injector import projection_matrices as pm
import torch

if __name__ == "__main__":
    a = np.array([[1, 2], [3,4]])
    b = np.array([[5,6], [7,8]])
  
    c = matmul(a,b)
    print("expected output")
    print(c)

    C = np.matmul(a,b)
    print("ground truth")
    print(C)

    a = np.array([[1,2,3], [4,5,6], [7,8,9]])
    #a = np.ones((3,3))
    b = np.array([[1,1],[1,1]])

    print("Computing convolution between A:")
    print(a)
    print("and B:")
    print(b)

    a = np.expand_dims(a, 0)
    b = np.expand_dims(b, 0)
    c = convolve(a, b)
    print("expected")
    print(c)

    aT = torch.from_numpy(a).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    bT = torch.from_numpy(b).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    print("ground truth")
    C = torch.nn.functional.conv2d(aT, bT)
    print(C)

    hw = SystolicArray(28, 28, 28, pm.output_stationary)
    h = []
    transformed = lowerings.ExpensiveLowering(a.shape, b.shape)
    result = hw.matmul(transformed.lower_activation(a), transformed.lower_kernel(b), h)
    result = transformed.lift(result)
    print(result)
    print(type(h[0]))
