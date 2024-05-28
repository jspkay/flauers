# Saffira relative imports
from .. import SystolicArray
from .. import projection_matrices
from .. import fault_models
from .. import lowerings

from ..__init__ import *

# others
import logging
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import concurrent.futures as futures

# Configuration variables
MULTIPROCESSING = True

# ********************************************************************************************
#                 Here we have to define our Conv2D layer description
# ********************************************************************************************

class SystolicConvolution(nn.Conv2d):

    def __init__(self, *args, hardware: SystolicArray = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization if needed

        self.device = torch.device("cpu")

        assert self.groups <= 1, "Convolutions with more than 1 groups are not possible for now!"

        self._name = 'SystolicConvolution'  # Custom name attribute
        if hardware is not None:  # if hardware is explicit, then use that!
            self.hw = hardware
        else:  # otherwise, automatically instantiate a new object with good dimensions
            self.hw = SystolicArray(
                100, 100, 150,
                projection_matrices.output_stationary,
                in_dtype=np.dtype(np.int16)
            )
            # fault = si.fault_models.StuckAt("a", x=1, y=1, bit=1, polarity=1, msb="last")
        # Set the padding attribute based on the input padding argument
        #self.padding = padding
        self.weights = None
        self.injecting = 0

        # Each element of the list should be a couple with the number of the channel and the fault:
        #   e.g. (-1, f) -> means that fault f will affect every channel
        #        (1, f) -> means that fault f will affect only channel 1
        self.channel_fault_list = []

    def add_fault(self, fault: fault_models.Fault, channel=-1):
        self.injecting += 1
        self.channel_fault_list.append((channel, fault))
        id = self.hw.add_fault(fault)
        return id

    def clear_faults(self):
        self.channel_fault_list = []
        self.injecting = 0
        self.hw.clear_all_faults()

    #def remove_fault(self, id):
    #    self.hw.clear_single_fault(id)
    #    self.injecting -= 1

    def load_weights(self, weights):
        # We assume that groups is always 1!
        # For more info visit
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

        self.weights = weights

    def _get_out_shape(self, H_in, W_in):
        output_h = np.floor(
            (H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) /
            self.stride[0] + 1)
        output_w = np.floor(
            (W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) /
            self.stride[1] + 1)
        return int(output_h), int(output_w)

    def forward(self, input):
        # TODO: Consider the out_channels and the in_channels
        logging.info("Starting forward!")

        '''
        # Calculate padding based on kernel size
        kernel_size_height, kernel_size_width = self.kernel_size
        padding_height = (kernel_size_height - 1) // 2
        padding_width = (kernel_size_width - 1) // 2
        padding = (int(padding_height), int(padding_width))
        '''
        if len(input.shape) == 4:  # We have batched inputs!
            batch_size = input.shape[0]
            out_shape = self._get_out_shape(input.shape[2], input.shape[3])

            print(f"out shape is {out_shape}")
            input = torch.nn.functional.pad(input, [self.padding[0], self.padding[0], self.padding[1], self.padding[1]])

            result = torch.zeros((batch_size, self.out_channels, *out_shape))

            print(f"[SystolicConvolution] starting batch-processing{'with injection!' if self.injecting >= 1 else ''}")
            bar = tqdm(range(batch_size), position=0, leave=True)
            it = iter(range(batch_size))

            if MULTIPROCESSING:
                # Parallelization
                with futures.ProcessPoolExecutor() as executor:
                    future_objects = {
                        executor.submit(
                            self._1grouping_conv,  # function
                            input[batch_index], out_shape  # arguments
                        ): batch_index for batch_index in it
                    }
                    for future in futures.as_completed(future_objects):
                        bar.update(1)
                        index = future_objects[future]
                        r = future.result()
                        result[index, :, :, :] = r
            else:
                for batch_index in it:
                    bar.update(1)
                    result[batch_index, :, :, :] = self._1grouping_conv(input[batch_index], out_shape)

            del out_shape
            del batch_size

        else:  # one image at a time
            out_shape = self._get_out_shape(input.shape[1], input.shape[2])
            result = self._1grouping_conv(input, out_shape)

            del out_shape

        del input

        return result

    def _1grouping_conv(self, input, out_shape):

        assert len(self.channel_fault_list) <= 1, "Only one fault admissible at a time!"

        result = torch.zeros((self.out_channels, *out_shape))
        if self.bias is not None:
            newBias = np.expand_dims(self.bias, (1,2) )
            result += newBias

        for c_out in range(self.out_channels):
            for c_in in range(self.in_channels):
                a = input[c_in, :, :]
                def zero_pads(X, pad):
                    """
                    X has shape (m, n_W, n_H, n_C)
                    """
                    X_pad = np.pad(X, ((pad, pad), (pad, pad)), 'constant', constant_values=0)
                    return X_pad

                b = self.weights[c_out, c_in, :, :]

                a = np.array(a)
                b = np.array(b)

                if self.channel_fault_list == [] or (
                    self.channel_fault_list[0][0] != c_out and
                    self.channel_fault_list[0][0] != -1
                ):
                    lolif = lowerings.S_Im2Col(a.shape, b.shape)
                    low_a = lolif.lower_activation(a)
                    low_b = lolif.lower_kernel(b)
                    x = np.matmul(low_a, low_b)
                    convolution = lolif.lift(x)
                    # convolution = convolve2d(a, b, mode="valid")

                else: # if self.channel_fault_list[0][0] == -1 or self.channel_fault_list[0][0] == c_out:
                    convolution = convolve_with_array(
                        a, b,
                        lowering=lowerings.S_Im2Col,
                        array=self.hw,
                    )

                result[c_out] += convolution

                """
                # ############# START DEBUGGING STUFF
                aT = a.expand(1, 1, -1, -1)
                bT = b.expand(1, 1, -1, -1)
                gt = np.array(torch.nn.functional.conv2d(aT, bT, stride=1)).squeeze(0).squeeze(0)
                r = np.abs(gt - convolution)
                c_no_zeros = convolution
                c_no_zeros[c_no_zeros==0] = 1
                r = r / c_no_zeros
                norm = np.linalg.norm(r, np.inf) # infinity norm
                if norm > 0.1: # error is greater than 1%
                    print(f"############# FOUND DIFFERENCE with norm {norm}")
                    print(a)
                    print(b)
                    print( np.array(gt) == convolution )
                    print(convolution)
                    print(gt)
                    exit(0)
                # ############## END DEBUGGING STUFF """

        return result

def compatible_layers(model: torch.nn.Module):
    res = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            res.append(name)
    return res

def replace_layer(model: torch.nn.Module, name: str, hardware: SystolicArray):
    layer = model.get_submodule(name)
    layer_name = name.split(".")[-1]
    parent_name = '.'.join(name.split(".")[:-1])
    parent = model.get_submodule(parent_name)

    if not isinstance(layer, nn.Conv2d):
        raise Exception("Not a conv2d Layer!")
        # Replace the convolution layer with the custom MyConv2D class

    conv_layer = layer
    new_conv_layer = SystolicConvolution(conv_layer.in_channels, conv_layer.out_channels,
                                         conv_layer.kernel_size, conv_layer.stride,
                                         conv_layer.padding, conv_layer.dilation,
                                         conv_layer.groups, conv_layer.bias is not None,
                                         hardware=hardware)

    # Copy the weights and biases from the original layer to the new layer
    new_conv_layer.weight.data = conv_layer.weight.data.clone()
    new_conv_layer.load_weights(conv_layer.weight.data.clone())
    if conv_layer.bias is not None:
        new_conv_layer.bias.data = conv_layer.bias.data.clone()

    # change the layer
    setattr(parent, layer_name, new_conv_layer)

    # Update the layer name in the model
    new_conv_layer._get_name = new_conv_layer._get_name


# ********************************************************************************************
#            This function extracts the Conv2D layers of the network architecture
# ********************************************************************************************
def get_conv2d_layer_indices(model):
    conv2d_indices = []
    for i, module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            conv2d_indices.append(i)
    print("In this network, there are", len(conv2d_indices), "Conv2D layers")
    print("Conv2D Layer Indices:", conv2d_indices)
    return conv2d_indices

# ********************************************************************************************
#      This function replace the Pytorch Conv2D layer with our customized Conv2D layer
# ********************************************************************************************
def replace_conv2d_layer(model, layer_num, hardware: SystolicArray = None):
    # Get the list of modules in the model
    modules = list(model.modules())
    # print(modules)

    if layer_num < 0 or layer_num >= len(modules):
        raise ValueError('Invalid layer number')

    # Get the selected convolution layer
    conv_layer = modules[layer_num]

    if isinstance(conv_layer, nn.Conv2d):
        # Replace the convolution layer with the custom MyConv2D class
        new_conv_layer = SystolicConvolution(conv_layer.in_channels, conv_layer.out_channels,
                                             conv_layer.kernel_size, conv_layer.stride,
                                             conv_layer.padding, conv_layer.dilation,
                                             conv_layer.groups, conv_layer.bias is not None,
                                             hardware=hardware)

        # Copy the weights and biases from the original layer to the new layer
        new_conv_layer.weight.data = conv_layer.weight.data.clone()
        new_conv_layer.load_weights(conv_layer.weight.data.clone())
        if conv_layer.bias is not None:
            new_conv_layer.bias.data = conv_layer.bias.data.clone()

        # Replace the selected layer in the model with the new convolution layer
        # TODO Generalize this piece of code such that it is possible to substitute the convolution on any model
        parent_module = modules[0]
        """print("PARENT: ", end=" ")
        print(parent_module)"""
        for name, module in parent_module.named_children():
            # print(name, " -> ", module)
            if module is conv_layer:
                # print("YEP! THAT'S IT!")
                setattr(parent_module, name, new_conv_layer)
                break
        # Update the layer name in the model
        # new_conv_layer._get_name = lambda: new_conv_layer._name # OLD VERSION
        new_conv_layer._get_name = new_conv_layer._get_name
    else:
        raise ValueError('The selected layer is not a Conv2D layer')
