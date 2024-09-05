# Saffira relative imports
from .. import SystolicArray
from .. import projection_matrices
from .. import fault_models
from .. import lowerings
from ..lowerings import LoLifType

from ..__init__ import *

# others
import logging
import torch
import torch.nn as nn
import numpy as np
from tqdm.autonotebook import tqdm, trange
import concurrent.futures as futures
import numba
import numba.cuda as cuda
# from numba_progress import ProgressBar

############### Helper functions ####################

def compatible_layers(model: torch.nn.Module):
    res = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or(
            isinstance(module, torch.nn.Linear)
        ):
            res.append(name)
    return res

def replace_layers(model: torch.nn.Module, 
                    names, # : str|list[bool], 
                    hardware: SystolicArray,
                    device: None|str|torch.device = None, 
                    tiling = False, # : bool|list[bool] = False,
                    deeper_faults = False,
                    gpu_griddim = 64,
                    gpu_blockdim = 128,
                    ):

    if isinstance(names, str):
        names = [names]
    if isinstance(tiling, bool):
        tiling = [tiling] * len(names)
    if device is None:
        device = "cuda" if hardware.use_gpu else "cpu"

    assert len(tiling) == len(names), "Please, give a list in Tiling with the same size of names!"

    idx = 0  # Needed for tiling
    for name in names:
        logging.debug(f"[torch] Replacing layer {name}")
        layer = model.get_submodule(name)
        layer_name = name.split(".")[-1]
        parent_name = '.'.join(name.split(".")[:-1])
        parent = model.get_submodule(parent_name)
            
        if isinstance(layer, nn.Conv2d):
            # Replace the convolution layer with the custom MyConv2D class
            conv_layer = layer
            new_layer = SystolicConv2d( in_channels = conv_layer.in_channels,
                                                out_channels = conv_layer.out_channels,
                                                kernel_size = conv_layer.kernel_size, 
                                                stride = conv_layer.stride,
                                                padding = conv_layer.padding,
                                                dilation = conv_layer.dilation,
                                                groups = conv_layer.groups,
                                                bias = conv_layer.bias is not None,
                                                padding_mode = conv_layer.padding_mode,

                                                device = device,
                                                gpu_griddim = gpu_griddim,
                                                gpu_blockdim = gpu_blockdim,

                                                hardware = hardware,
                                                tiling = tiling[idx],
                                                deeper_faults = deeper_faults,
                                                )
        elif isinstance(layer, nn.Linear):
            new_layer = SystolicLinear( in_features = layer.in_features,
                                        out_features = layer.out_features,
                                        bias = layer.bias is not None,

                                        device = device,
                                        gpu_griddim = gpu_griddim,
                                        gpu_blockdim = gpu_blockdim,

                                        hardware=hardware,
                                        tiling = tiling[idx],
                                        )
        else:
            raise Exception(f"The requested layer {name} is neither a Conv2d nor a Linear.")


        # Copy the weights and biases from the original layer to the new layer
        new_layer.load_weights(layer.weight.data.clone())
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data.clone()

        # change the layer
        setattr(parent, layer_name, new_layer)

        # Update the layer name in the model
        new_layer._get_name = new_layer._get_name

        idx += 1  # Needed for tiling

    return model

####################### Layer definitions ################

class SystolicLinear(nn.Linear):
    def __init__(self, *args,
                hardware: SystolicArray = None,
                tiling = False,
                gpu_griddim = 64,
                gpu_blockdim = 128,
                **kwargs):
        super().__init__(*args, **kwargs)

        self.weight = None
        self.injecting = 0

        self.tiling = tiling

        device = kwargs.get("device")
        self.use_gpu = device == "cuda" or device == torch.device("cuda")

        self._name = 'SystolicLinear'
        if hardware is not None:
            self.hw = hardware
        else:
            self.hw = SystolicArray(
                100, 100, 150,
                projection_matrices.output_stationary,
                in_dtype=np.dtype(np.float32),
                use_gpu=self.use_gpu,
                gpu_blockdim = gpu_blockdim,
                gpu_griddim = gpu_griddim
            )

    def load_weights(self, weight):
        self.weight = torch.nn.Parameter(weight.transpose(0, 1))

    def forward(self, fmap):
        batch_size = fmap.shape[0]

        if self.use_gpu:
            result = torch.zeros((batch_size, self.out_features), device="cuda")
            result = cuda.as_cuda_array(result)
            if self.tiling:
                part = self.hw.matmul_legacy_cuda_tiled(
                            result,
                            cuda.as_cuda_array(fmap.detach()),
                            cuda.as_cuda_array(self.weight.detach()),
                        )
            else:
                part = self.hw.matmul_legacy_cuda(
                            result,
                            cuda.as_cuda_array(fmap.detach()),
                            cuda.as_cuda_array(self.weight.detach()),
                        )
            result = torch.as_tensor(result, device="cuda") + self.bias
        else:
            result = torch.zeros((batch_size, self.out_features), device = "cpu")
            if self.tiling:
                part = self.hw.matmul_legacy_cpu_tiled(
                            fmap.numpy(),
                            self.weight.numpy(),
                        )
            else:
                part = self.hw.matmul_legacy_cpu(
                    fmap.numpy(),
                    self.weight.numpy(),
                    )
            result[:, :] = torch.as_tensor(part, device="cpu") + self.bias
            
        return result


class SystolicConv2d(nn.Conv2d):

    def __init__(self, *args,
                 hardware: SystolicArray = None,
                 multiprocessing = False,
                 tiling = False,
                 deeper_faults = False,
                 lowering: lowerings.LoLif = lowerings.S_Im2Col,
                 gpu_blockdim = 64,
                 gpu_griddim = 128,
                 **kwargs
                 ):
        # Additional initialization if needed
        super().__init__(*args, **kwargs)
        
        ###### Torch related
        self._name = 'SystolicConv2d'  # Custom name attribute

        # ### padding argument
        if (len(args) >=5 and args[4] == "valid") or (
                kwargs.get("padding") == "valid" ):
            self.padding = (0, 0)

        assert self.groups <= 1, "Convolutions with more than 1 groups are not possible for now!"

        self.weights = None

        # ### Simulation optimization
        device = kwargs.get("device")
        self.use_gpu = device == "cuda" or device == torch.device("cuda")
        self.gpu_blockdim = gpu_blockdim
        self.gpu_griddim = gpu_griddim

        ########## Hardware simulation related
        if hardware is not None:  # if hardware is explicit, then use that!
            self.hw = hardware
        else:  # otherwise, automatically instantiate a new object with good dimensions
            self.hw = SystolicArray(
                100, 100, 150,
                projection_matrices.output_stationary,
                in_dtype=np.dtype(np.float32),
                use_gpu = self.use_gpu,
                gpu_blockdim = gpu_blockdim,
                gpu_griddim = gpu_griddim
            )

        self.tiling = tiling

        self.lowering = lowering
        self.input_shape = None

        # ### faults
        self.injecting = 0
        self.deeper_faults = deeper_faults
            # Each element of the list should be a couple with the number of the channel and the fault:
            #   e.g. (-1, f) -> means that fault f will affect every channel
            #        (1, f) -> means that fault f will affect only channel 1
        self.channel_fault_list = []

        ### Deprecated
        self.MULTIPROCESSING = False

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
        self.weight = torch.nn.Parameter(weights)

    def _get_out_shape(self, H_in, W_in):
        output_h = np.floor(
            (H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) /
            self.stride[0] + 1
        )
        output_w = np.floor(
            (W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) /
            self.stride[1] + 1
        )
        return int(output_h), int(output_w)

    def forward(self, batch):
        # TODO: Consider the out_channels and the in_channels
        logging.info("Starting forward!")

        if self.input_shape != batch.shape:
            self._update_lolif(batch)

        if self.use_gpu:
            res = self._forward_cuda(batch)
            torch.cuda.empty_cache()
        else:
            res = self._forward_with_cpu(batch)
        
        return res

    def _update_lolif(self, batch):
        # Prepare lowering object
        self.input_shape = batch.shape
        if self.lowering.type == LoLifType.SINGLE:
            self.lolif = self.lowering(
                    (
                        batch.shape[2] + 2*self.padding[0], 
                        batch.shape[3] + 2*self.padding[1]
                    ),
                    self.weight.shape[2:4],
                    use_gpu = self.use_gpu,
                    gpu_blockdim = self.gpu_blockdim,
                    gpu_griddim = self.gpu_griddim
            )
        else: #if lowering.type == LoLifType.CHANNELED:
            assert False, "Channeled are not supported yet."
            self.lolif = self.lowering(
                    batch.shape[1:4], 
                    self.weight.shape, # TODO generalize the type!
            )

        # Instantiate the lowered weights object
        if self.use_gpu:
            self.weights = cuda.device_array((
                self.out_channels, 
                self.in_channels, 
                *self.lolif.lowered_kernel_shape
            ), dtype = np.float32)
        else:
            self.weights = torch.zeros((
                self.out_channels, 
                self.in_channels,
                *self.lolif.lowered_kernel_shape
            ))
    
        for c_out in range(self.out_channels):
            for c_in in range(self.in_channels):
                if self.use_gpu:
                    self.lolif.lower_kernel_cuda(
                        self.weights[c_out, c_in],
                        numba.cuda.as_cuda_array(
                            self.weight[c_out, c_in].detach()
                        )
                    )
                else:
                    self.weights[c_out, c_in] = self.lolif.lower_kernel_cpu(
                        self.weight[c_out, c_in]
                    )

    def _forward_cuda(self, batch):
        out_shape = self._get_out_shape(batch.shape[2], batch.shape[3])

        # insert padding
        batch = torch.nn.functional.pad(
                batch, [
                    self.padding[0], self.padding[0],
                    self.padding[1], self.padding[1],
                ]
            )

        # correct_conv = torch.nn.functional.conv2d(batch, self.weight, self.bias)

        # Cuda conversion CAI
        batch = cuda.as_cuda_array(batch.detach())
        assert len(batch.shape) == 4, "Cannot process unbatched inputs!"

        batch_size = batch.shape[0]
        result = torch.zeros( 
                    (batch_size, self.out_channels, *out_shape),  
                    dtype=torch.float32, device="cuda"
                ) 
        if self.bias is not None:
            result += self.bias.view((1, self.out_channels, 1, 1)) #resulting tensor
        result = cuda.as_cuda_array(result)

        lowered_result = cuda.device_array((self.lolif.lowered_activation_shape[0], self.lolif.lowered_kernel_shape[1]), dtype=result.dtype) 
        for idx in trange(0, batch_size, leave=False, dynamic_ncols=True,
                         desc=f"[SystolicConv2d] batched {'injected' if self.injecting >= 1 else ''}"
                         ):
            for c_out in range(self.out_channels):
                for c_in in range(self.in_channels):
                    lowered_act = self.lolif.lower_activation(batch[idx, c_in])
                    self.hw.matmul_legacy_cuda_tiled(lowered_result, lowered_act, self.weights[c_out, c_in])
                    self.lolif.lift_cuda(result[idx, c_out], lowered_result,  additive=True)

        result = torch.as_tensor(result, device="cuda")

        # if not torch.allclose(correct_conv, result, rtol=0.01):
        #     torch.save(torch.tensor(batch), "batch.pt")
        #     torch.save(self.weight, "wegith.pt")
        #     torch.save(self.bias, "bias.pt")
        #     torch.save(correct_conv, "correct_conv.pt")
        #     torch.save(result, "computed_conv.pt")
        #     assert False, f"Uh la la! Seems like there is something shady going on with SystolicConv2d!" # \ncorrect:\n{correct_conv}\ncomputed:\n{result}"

        return result

    def _forward_with_cpu(self, input):
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

            # print(f"out shape is {out_shape}")
            input = torch.nn.functional.pad(input, [self.padding[0], self.padding[0], self.padding[1], self.padding[1]])

            result = torch.zeros((batch_size, self.out_channels, *out_shape))

            # print(f"[SystolicConvolution] starting batch-processing{'with injection!' if self.injecting >= 1 else ''}")
            bar = trange(0, batch_size, leave=False, dynamic_ncols=True,
                         desc=f"[SystolicConv2d] batched {'injected' if self.injecting >= 1 else ''}", 
            )
            it = iter(range(batch_size))

            if self.MULTIPROCESSING:
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
                    result[batch_index, :, :, :] = self._1grouping_conv(input[batch_index], out_shape)
                    bar.update(1)

            bar.close()
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

                a = a.numpy()
                b = b.numpy()

                if not self.deeper_faults and (
                    self.channel_fault_list == [] or (
                        self.channel_fault_list[0][0] != c_out and
                        self.channel_fault_list[0][0] != -1
                    )
                ):
                    lolif = lowerings.S_Im2Col(a.shape, b.shape)
                    low_a = lolif.lower_activation(a)
                    low_b = lolif.lower_kernel(b)
                    x = np.matmul(low_a, low_b)
                    convolution = lolif.lift(x)
                    # convolution = convolve2d(a, b, mode="valid")

                else: # if self.channel_fault_list[0][0] == -1 or self.channel_fault_list[0][0] == c_out:
                    lolif = lowerings.S_Im2Col(a.shape, b.shape)
                    low_a = lolif.lower_activation(a)
                    low_b = lolif.lower_kernel(b)
                    x = self.hw.matmul_cpu(low_a, low_b)
                    convolution = lolif.lift(x)

                    """ OLD
                    convolution = convolve_with_array(
                        a, b,
                        lowering=lowerings.S_Im2Col,
                        array=self.hw,
                        tiling=self.tiling,
                    ) """

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


######### Soon to be depracated #############################################################

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
        new_conv_layer = SystolicConv2d(in_channels=conv_layer.in_channels,
                                             out_channels=conv_layer.out_channels,
                                             kernel_size=conv_layer.kernel_size, 
                                             stride=conv_layer.stride,
                                             padding=conv_layer.padding,
                                             dilation=conv_layer.dilation,
                                             groups=conv_layer.groups, 
                                             bias=conv_layer.bias is not None,
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
