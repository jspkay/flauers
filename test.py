from systolic_injector import *
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
    
    c = convolution(a,b)
    print("expected")
    print(c)

    aT = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    bT = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    print("ground truth")
    C = torch.nn.functional.conv2d(aT, bT)
    print(C)

