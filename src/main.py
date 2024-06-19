import flauers as si
import flauers.torch
si.torch = flauers.torch
from flauers import projection_matrices as pm

import torch
import torch.nn as nn

import torch.nn.functional as F
from torchvision.transforms import v2

import logging
import numpy as np
from tqdm.auto import tqdm
from timeit import timeit

# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARNING)

def lenet_test():
    a = LeNet()
    model = torch.load("best_model", map_location="cpu")
    model.eval()
    pil_to_tensor = tv.transforms.Compose([
        # stransforms.RandomResizedCrop(size
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    mnist_test = tv.datasets.MNIST(".", download=True, train=False, transform=pil_to_tensor)
    validation_loader = torch.utils.data.DataLoader(mnist_test, batch_size = 128, shuffle=False, num_workers=4, pin_memory=True)

    tot = 0
    correct = 0
    i=0
    with torch.no_grad():
        for img in validation_loader:
            inputs, labels = img
            outputs = model(inputs.to("cpu"))
            tot += len(labels)
            correct += (outputs.argmax(1) == labels.to("cpu")).float().sum()
            if i==2:
                break
            i+=1
        acc = correct / tot
    print("model accuracy is ", acc.item()*100)

    hw = si.SystolicArray(30, 30, 300, si.projection_matrices.output_stationary, in_dtype=np.float32, mac_dtype=np.float32)
    compatible = si.torch.compatible_layers(model)
    print(compatible)
    si.torch.replace_layers(model, compatible, hardware = hw, tiling = True)
    model.eval()

    tot = 0
    correct = 0
    i = 0
    with torch.no_grad():
        for img in tqdm(validation_loader):
            inputs, labels = img
            outputs = model(inputs.to("cpu"))
            tot += len(labels)
            correct += (outputs.argmax(1) == labels.to("cpu")).float().sum()
            if i==2:
                break
            i+=1
        acc = correct / tot
    print("model accuracy is ", acc.item()*100)

def timing():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    b = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]], dtype=np.float32)

    hw_opt = si.SystolicArray(10, 10, 10, pm.no_local_reuse, in_dtype=np.int8, optimized=True)
    hw = si.SystolicArray(10, 10, 10, pm.no_local_reuse, in_dtype=np.int8, optimized=False)
    f = si.fault_models.StuckAt(line="c", x=0, y=0, bit=3, polarity=1, msb="last")

    c0 = timeit(lambda: a @ b, number=1_000_000)
    print(f"Standard matmul requires {c0:.4f}s")

    c = timeit(lambda: hw_opt.matmul(a, b), number=1_000_000)
    print(f"Injected matmul with CPU optimization requires {c:.4f}s")

    c1 = timeit(lambda: hw.matmul(a, b), number=1_000_000)
    print(f"Old matmul with \"naive\" implementation requires {c1:.4f}s")

def general_matmul(): 
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    b = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]], dtype=np.float32)

    hw_opt = si.SystolicArray(10, 10, 10, pm.no_local_reuse, in_dtype=np.int8, optimized=True)
    hw = si.SystolicArray(10, 10, 10, pm.no_local_reuse, in_dtype=np.int8, optimized=False)
    f = si.fault_models.StuckAt(line="c", x=0, y=0, bit=3, polarity=1, msb="last")
    hw.add_fault(f)
    hw_opt.add_fault(f)

    c = hw.matmul(a, b)
    print("old:\n", c)
    c = hw_opt.matmul(a, b)
    print("optimized:\n", c)
    print("expected:\n", a@b)

    
def prova():
    A = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]], dtype=np.int8)
    B = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]], dtype=np.int8)

    hw = si.SystolicArray(10, 10, 10, pm.output_stationary)
    C = hw.matmul(A, B)
    print(C)

def basic_tiling():
    A = np.array([ [1,2, 3], [4, 5, 6], [7,8,9]])
    B = np.array([ [1,2, 3], [4, 5, 6], [7,8,9]])

    print("### A ###:")
    print(A)

    print("### B ###:")
    print(B)

    print("### C = A @ B ###")
    print(A@B)
    print("######")

    C = np.zeros((3, 3))
    N1 = 2
    N2 = 2
    N3 = 4
    it = flauers.tilings.Tiling(A, B, N1, N2, N3)
    for a, b, i, j in it:
        print(a)
        print(b)
        print(a @ b)
        print("----")    
        C[i:i+N1, j:j+N2] += a@b
    
    print("#####")
    print(C)

    print( np.allclose(C, A@B) )

def tiling():
    np.random.seed(0)
    A = (np.random.random((6, 6))* 10).astype(np.int8)
    B = (np.random.random((6, 3))* 10).astype(np.int8)

    print(A)
    print(B)

    N1, N2, N3 = 2, 2, 20

    hw = flauers.SystolicArray(N1, N2, N3, 
            flauers.projection_matrices.output_stationary,
            in_dtype=np.float32, optimized = True)
    f = flauers.fault_models.StuckAt("a", x=0, y=0, bit=7, polarity=1)
    hw.add_fault(f)

    Cok = A.astype(np.int32)@B.astype(np.int32)
    C = hw.matmul(A, B, tiling=True)
    print("cok")
    print(Cok)
    print("c")
    print(C)
    print( np.allclose(C, Cok) )

def test_injection_simple():
    print("Running injection_simple")
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # b = np.ones((2,2))
    b = np.array([[10, 11], [12, 13]])

    array = si.SystolicArray(10, 10, 10,
                            si.projection_matrices.output_stationary,
                            in_dtype=np.dtype(np.int8), optimized=True)
    x, y = array.physical_PEs
    f = si.fault_models.StuckAt("c", x=0, y=0, bit=0, polarity=1, msb="first")
    array.add_fault(f)
    c_sa = si.convolve_with_array(a, b, lowering=si.lowerings.S_Im2Col, array=array)

    aT = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    bT = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))

    c_torch = torch.nn.functional.conv2d(aT, bT)

    print(c_sa)
    print(c_torch)
    print(c_sa == np.array(c_torch) )

def physical_space():
    from matplotlib import pyplot as plt 
    hw = si.SystolicArray(10, 10, 10, 
    si.projection_matrices.output_stationary)
    pes = np.array(hw.space_projection())
    print(pes.shape)
    print(pes)
    plt.scatter(pes[:, 0], pes[:, 1])
    plt.show()

if __name__ == "__main__":
    lenet_test()
