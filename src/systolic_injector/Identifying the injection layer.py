import torch
import torch.nn as nn

#********************************************************************************************
##                Here we have to define our Conv2D layer description
#********************************************************************************************

class MyConv2D(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MyConv2D, self).__init__(*args, **kwargs)
        # Additional initialization if needed
        self._name = 'MyConv2D'  # Custom name attribute

    def forward(self, input):
        # Custom forward pass logic
        # Modify the behavior of the convolution layer here
        # Set all the outputs to zero
        output1 = super(MyConv2D, self).forward(input)
        output = torch.zeros_like(output1)
        return output


#********************************************************************************************
##           This function extracts the Conv2D layers of the network architecture
#********************************************************************************************
def get_conv2d_layer_indices(model):
    conv2d_indices = []
    for i, module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            conv2d_indices.append(i)
    print("In this network, there are", len(conv2d_indices), "Conv2D layers")
    print("Conv2D Layer Indices:", conv2d_indices)
    return conv2d_indices



#********************************************************************************************
##     This function replace the Pytorch Conv2D layer with our customized Conv2D layer
#********************************************************************************************
def replace_conv2d_layer(model, layer_num):
    # Get the list of modules in the model
    modules = list(model.modules())

    if layer_num < 0 or layer_num >= len(modules):
        raise ValueError('Invalid layer number')

    # Get the selected convolution layer
    conv_layer = modules[layer_num]

    if isinstance(conv_layer, nn.Conv2d):
        # Replace the convolution layer with the custom MyConv2D class
        new_conv_layer = MyConv2D(conv_layer.in_channels, conv_layer.out_channels,
                                  conv_layer.kernel_size, conv_layer.stride,
                                  conv_layer.padding, conv_layer.dilation,
                                  conv_layer.groups, conv_layer.bias is not None)

        # Copy the weights and biases from the original layer to the new layer
        new_conv_layer.weight.data = conv_layer.weight.data.clone()
        if conv_layer.bias is not None:
            new_conv_layer.bias.data = conv_layer.bias.data.clone()

        # Replace the selected layer in the model with the new convolution layer
        parent_module = modules[layer_num - 1]
        for name, module in parent_module.named_children():
            if module is conv_layer:
                setattr(parent_module, name, new_conv_layer)
                break
        # Update the layer name in the model
        new_conv_layer._get_name = lambda: new_conv_layer._name
    else:
        raise ValueError('The selected layer is not a Conv2D layer')

#********************************************************************************************
##     Example to run: AlexNet
#********************************************************************************************
# Example usage
from torchvision.models import AlexNet

# Create an instance of the AlexNet model
model = AlexNet()
# Define the shape of the input tensor
batch_size = 1
num_channels = 1
height = 224
width = 224


# Example usage
alexnet = AlexNet()
conv2d_indices = get_conv2d_layer_indices(alexnet)

#********************************************************************************************
##     Ask the user for the layer number to replace for the faul injection
#********************************************************************************************

layer_num = int(input("From the Conv2D layers number (0 to {}), which one you wanna perform the Fault Injection: ".format(len(conv2d_indices)-1)))

#********************************************************************************************
##     call the function to replace the selected convolutional layer
#********************************************************************************************

replace_conv2d_layer(alexnet, conv2d_indices[layer_num])

#********************************************************************************************
##                              Generate a random input image
#********************************************************************************************


input_tensor = torch.randn(1, 3, 224, 224)

# Forward pass through the model
output = alexnet(input_tensor)

print('Output shape:', output.shape)
# Print the network structure with layer names
print('Network Structure:')
print(model)