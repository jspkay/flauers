# Saffira relative imports
from .. import SystolicArray
from .. import projection_matrices
from .. import fault_models

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
    # TODO Copy from the other project!
    pass


# ********************************************************************************************
#      This function replace the Pytorch Conv2D layer with our customized Conv2D layer
# ********************************************************************************************
def replace_conv2d_layer(model, layer_num):
    # Get the list of modules in the model
    modules = list(model.modules())
    print(modules)

    if layer_num < 0 or layer_num >= len(modules):
        raise ValueError('Invalid layer number')

    # Get the selected convolution layer
    conv_layer = modules[layer_num]

    if isinstance(conv_layer, nn.Conv2d):
        # Replace the convolution layer with the custom MyConv2D class
        new_conv_layer = SystolicConvolution(conv_layer.in_channels, conv_layer.out_channels,
                                             conv_layer.kernel_size, conv_layer.stride,
                                             conv_layer.padding, conv_layer.dilation,
                                             conv_layer.groups, conv_layer.bias is not None)

        # Copy the weights and biases from the original layer to the new layer
        new_conv_layer.weight.data = conv_layer.weight.data.clone()
        new_conv_layer.load_weights(conv_layer.weight.data.clone())
        if conv_layer.bias is not None:
            new_conv_layer.bias.data = conv_layer.bias.data.clone()

        # Replace the selected layer in the model with the new convolution layer
        # TODO Generalize this piece of code such that it is possible to substitute the convolution on any model
        parent_module = modules[0]
        print("PARENT: ", end=" ")
        print(parent_module)
        for name, module in parent_module.named_children():
            print(name, " -> ", module)
            if module is conv_layer:
                print("YEP! THAT'S IT!")
                setattr(parent_module, name, new_conv_layer)
                break
        # Update the layer name in the model
        # new_conv_layer._get_name = lambda: new_conv_layer._name # OLD VERSION
        new_conv_layer._get_name = new_conv_layer._get_name
    else:
        raise ValueError('The selected layer is not a Conv2D layer')