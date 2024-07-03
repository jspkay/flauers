from . import lenet
import flauers
import flauers.torch

import torch
from torchvision.transforms import v2
import unittest
import torchvision as tv
import numpy as np
from tqdm import tqdm

import sys
setattr(sys.modules["__main__"], "LeNet", lenet.LeNet)

class TestTorch(unittest.TestCase):

    def test_conv_simple_layer(self):
        torch_correct =             torch.nn.Conv2d(2, 2, 3, stride=1, padding="valid", dilation=1, groups=1, bias=True, padding_mode="zeros", device="cuda", dtype=torch.float32 )
        flauer_layer = flauers.torch.SystolicConv2d(2, 2, 3, stride=1, padding="valid", dilation=1, groups=1, bias=True, padding_mode="zeros", device="cuda", dtype=torch.float32 )
        
        torch_correct.weight = torch.nn.Parameter(
                                  torch.ones((2, 2, 3, 3), device="cuda"))
        flauer_layer.load_weights(torch.ones((2, 2, 3, 3), device="cuda"))
        torch_correct.bias = torch.nn.Parameter(torch.ones((2), device="cuda"))
        flauer_layer.bias  = torch.nn.Parameter(torch.ones((2), device="cuda"))

        A = torch.ones((1, 2, 4, 4), device="cuda")
        with torch.no_grad():
            B = flauer_layer(A)
            Bok = torch_correct(A)
        print(B)
        print(Bok)
        self.assertTrue(np.allclose(B.cpu(), Bok.cpu()))

    def test_linear_simple_layer(self):
        torch_correct = torch.nn.Linear(32, 10 , bias=True, device="cuda")
        flauer_layer = flauers.torch.SystolicLinear(32, 10, tiling=False, device="cuda")

        torch_correct.weight = torch.nn.Parameter(
                                   torch.ones((32, 10), device="cuda"))
        flauer_layer.load_weights( torch.ones((32,10), device="cuda") )
        torch_correct.bias = torch.nn.Parameter(torch.ones(10, device="cuda"))
        flauer_layer.bias = torch.nn.Parameter(torch.ones(10, device="cuda"))

        A = torch.ones((32), device="cuda")
        with torch.no_grad():
            B = flauer_layer(A)
            Bok = torch_correct(A)
        print(B)
        print(Bok)
        self.assertTrue(np.allclose(B.cpu(), Bok.cpu()))


    def test_conv_and_linear_cpu_noinj(self):
        model = torch.load("best_model", map_location="cpu",)
        model.eval()
        
        pil_to_tensor = tv.transforms.Compose([ # SOOO, WHAT ABOUT v2?
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
                if i==1:
                    break
                i+=1
            golden_acc = correct / tot
        print("model accuracy is ", golden_acc.item()*100)

    def test_conv_and_linear_cpu(self):
        model = torch.load("best_model", map_location="cpu",)
        model.eval()
        
        pil_to_tensor = tv.transforms.Compose([ # SOOO, WHAT ABOUT v2?
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
                if i==1:
                    break
                i+=1
            golden_acc = correct / tot
        print("model accuracy is ", golden_acc.item()*100)

        hw = flauers.SystolicArray(
            30, 30, 300, 
            flauers.projection_matrices.output_stationary,
            in_dtype=np.float32,
            mac_dtype=np.float32
            )
        compatible = flauers.torch.compatible_layers(model)
        print(compatible)
        flauers.torch.replace_layers(model, compatible, hardware = hw, tiling = True)
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
                if i==1:
                    break
                i+=1
            systolic_acc = correct / tot
        print("systolic model accuracy is ", systolic_acc.item()*100)

        self.assertEqual(golden_acc, systolic_acc)

    def test_conv_and_linear_cuda_noinj(self):
        model = torch.load("best_model", map_location="cuda")
        model.eval()
        model.to("cuda")
        
        pil_to_tensor = tv.transforms.Compose([ # SOOO, WHAT ABOUT v2?
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]) 

        mnist_test = tv.datasets.MNIST(".", download=True, train=False, transform=pil_to_tensor)
        validation_loader = torch.utils.data.DataLoader(mnist_test, batch_size = 128, shuffle=False, num_workers=4)

        tot = 0
        correct = 0
        i=0
        with torch.no_grad():
            for img in validation_loader:
                inputs, labels = img
                outputs = model(inputs.to("cuda"))
                tot += len(labels)
                correct += (outputs.argmax(1) == labels.to("cuda")).float().sum()
                if i==1:
                    break
                i+=1
            golden_acc = correct / tot
        print("model accuracy is ", golden_acc.item()*100)
    
    def test_conv_and_linear_cuda(self):
        model = torch.load("best_model", map_location="cuda",)
        model.eval()
        
        pil_to_tensor = tv.transforms.Compose([ # SOOO, WHAT ABOUT v2?
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
                outputs = model(inputs.to("cuda"))
                tot += len(labels)
                correct += (outputs.argmax(1) == labels.to("cuda")).float().sum()
                if i==1:
                    break
                i+=1
            golden_acc = correct / tot
        print("model accuracy is ", golden_acc.item()*100)

        hw = flauers.SystolicArray(
            30, 30, 300, 
            flauers.projection_matrices.output_stationary,
            in_dtype=np.float32,
            mac_dtype=np.float32,
            use_gpu = True
            )
        compatible = flauers.torch.compatible_layers(model)
        print(compatible)
        flauers.torch.replace_layers(model, compatible, device="cuda", hardware = hw, tiling = True)
        model.eval()

        tot = 0
        correct = 0
        i = 0
        with torch.no_grad():
            for img in tqdm(validation_loader):
                inputs, labels = img
                outputs = model(inputs.to("cuda"))
                tot += len(labels)
                correct += (outputs.argmax(1) == labels.to("cuda")).float().sum()
                if i==1:
                    break
                i+=1
            systolic_acc = correct / tot
        print("systolic model accuracy is ", systolic_acc.item()*100)

        self.assertEqual(golden_acc, systolic_acc)
