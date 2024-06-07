import numpy as np

import flauers as si
from flauers import projection_matrices as pm
import torch
import torch.nn as nn
import torchvision as tv
from torchvision.transforms import v2
import torch.nn.functional as F
import unittest
import logging

from timeit import timeit

# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARNING)

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding="valid")
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding="valid")
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x.to("cpu")
        x = self.conv1(x).to("cpu")
        
        x = self.pool(F.relu(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_L1(self, x):
        return self.conv1(x)

    def forward_rest(self, x):
        x = self.pool(F.relu(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
    validation_loader = torch.utils.data.DataLoader(mnist_test, batch_size = 2000, shuffle=False, num_workers=4, pin_memory=True)

    with torch.no_grad():
        for img in validation_loader:
            inputs, labels = img
            outputs = model(inputs.to("cpu"))
            tot += len(labels)
            correct += (outputs.argmax(1) == labels.to("cpu")).float().sum()

        acc = correct / tot
    print("model accuracy is ", acc.item()*100)

    hw = sa.SystolicArray(30, 30, 300, sa.projection_matrices.output_stationary, in_dtype=np.float32, mac_dtype=np.float32)
    sa.torch.replace_conv2d_layer(model, 1, hw)
    model.eval()

    with torch.no_grad():
        for img in validation_loader:
            inputs, labels = img
            outputs = model(inputs.to("cpu"))

            tot += len(labels)
            correct += (outputs.argmax(1) == labels.to(torch.device("cpu")).float().sum())

        acc = correct / tot

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

    


if __name__ == "__main__":
    timing()


