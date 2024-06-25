from .lenet import *
import flauers
import flauers.torch


import unittest
# import torchvision as tv

class TestTorch(unittest.TestCase):

    def test_conv_and_linear_cpu_noinj(self):
        model = torch.load("best_model", map_location="cpu",)
        model.eval()
        """
        pil_to_tensor = tv.transforms.Compose([ # SOOO, WHAT ABOUT v2?
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]) """

        mnist_test = tv.datasets.MNIST(".", download=True, train=False) #, transform=pil_to_tensor)
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
            golden_acc = correct / tot
        print("model accuracy is ", acc.item()*100)

    def test_conv_and_linear_cpu(self):
        model = torch.load("best_model", map_location="cpu",)
        model.eval()
        """
        pil_to_tensor = tv.transforms.Compose([ # SOOO, WHAT ABOUT v2?
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]) """

        mnist_test = tv.datasets.MNIST(".", download=True, train=False) #, transform=pil_to_tensor)
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
            golden_acc = correct / tot
        print("model accuracy is ", acc.item()*100)

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
                if i==2:
                    break
                i+=1
            systolic_acc = correct / tot
        print("systolic model accuracy is ", acc.item()*100)

        self.assertEqual(golden_acc, systolic_acc)

    def test_conv_and_linear_cuda_noinj(self):
        model = torch.load("best_model", map_location="cuda",)
        model.eval()
        """
        pil_to_tensor = tv.transforms.Compose([ # SOOO, WHAT ABOUT v2?
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]) """

        mnist_test = tv.datasets.MNIST(".", download=True, train=False) #, transform=pil_to_tensor)
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
            golden_acc = correct / tot
        print("model accuracy is ", acc.item()*100)