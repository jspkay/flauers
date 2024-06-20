import numpy as np
from . import utils
import logging
from enum import Enum

# Indexes for the output of the lowerings
lowered_indices = {
    "HEIGHT": 0,
    "WIDTH": 1,
}


# Enums for the different names of LowLif objects (see LowLif for more infos)
class LoLifType(Enum):
    SINGLE = 0
    CHANNELED = 1
    PARALLEL = 2


class ShapeError(Exception):

    def __init__(self, message):
        super().__init__(self, message)


def _check_shape(a: tuple, lolif_type: LoLifType) -> None:
    """
    This function checks that the shape we give in input have the correct structure.
    Specifically:
        - we want a different number of dimensions depending on the type,
        - we only accept square matrices (i.e. width == height)
    """

    indexes = None

    lll = len(a)
    if lolif_type == LoLifType.SINGLE and lll != 2:
        raise ShapeError(f"The tensor has shape {a} and it is not bi-dimensional but a LowLif of NORMAL type!")

    if lolif_type == LoLifType.CHANNELED and lll != 3:
        raise ShapeError(f"The tensor has shape {a} and it is not tri-dimensional but a LowLif of CHANNELED type!")

    if lolif_type == LoLifType.PARALLEL and lll != 4:
        raise ShapeError(f"The tensor has shape {a} and it is not four-dimensional but a LowLif of BATCHED type!")

    # We assume the first is the batch size
    h_index = utils.input_indexes_mapping[lolif_type.value]["HEIGHT"]
    w_index = utils.input_indexes_mapping[lolif_type.value]["WIDTH"]
    if a[h_index] != a[w_index]:  # only square matrices allowed!
        raise ShapeError("The width and the height of the tensor must be the same! Only square matrices are allowed")


def _check_tensor_against_shape(tensor, shape):
    t_shape = tensor.shape
    if len(t_shape) != len(shape):
        raise ShapeError(f"The tensor you want to lower has a wrong shape! It is supposed to be {shape} but {t_shape} found")


# TODO: Add the stripe parameter for every class!!!


class LoLif:  # Lowering-Lifting class
    """
    This class is the one that performs the lowering (and lifting) of the tensors to (from) 2D multiplicable matrices.
    LowLif is actually supposed to be an abstract class which implements the basic common methods for every type of
    LowLif implemented.
    Each LowLif object is associated with a specific shape of the inputs (activations and kernels). The shapes are
    defined in the constructor and can change using the methods set_activation_shape and set_kernel_shape.
    Every time a new shape is assigned (or during construction) the method should compute the shapes of the lowered
    outputs.

    This object performs a transformation of a tensor (potentially) that takes into account batch-size, channels,
    heights and widths and spits out 2D matrices that can be MULTIPLIED together for performing the convolution.
    So, we only take into account matrix multiplication of 2D matrices. Classes that have LowLif as superclass, should
    specify the type of lowering they do directly in their name using a prefix:

    - N_ stands for classes that take into account only a single convolution
    - C_ is for convolutions with channels, so that activations and kernels have the same number of channels
    - B_ is for those convolutions that take into account also the batch size for channeled convolutions

    """

    activation_shape = None  # These two variables are used to check
    kernel_shape = None  # that the lowering we want to do is legal

    lowered_activation_shape = None  # These are used by implementing classes to infer the lowered shapes
    lowered_kernel_shape = None
    lifted_output_shape = None

    activation_size = 0  # size of input matrix (width and height per channel)
    kernel_size = 0  # size of kernel matrix (width and height per channel)
    output_size = 0  # size of the output (width and height of the input per channel)
    number_of_channels = 0  # number of channels
    number_conv_per_batch = 0  # number of convolutions per each batch

    HEIGHT = None
    WIDTH = None
    CHANNEL = None
    BATCH = None

    def __init__(self,
                 activation_shape: tuple,
                 kernel_shape: tuple,
                 stride_x: int = 1,
                 stride_y: int = 1,
                 lolif_type: LoLifType = LoLifType.SINGLE
                 ):
        logging.info("[LowLif super-class] initialization...")

        # Check the size of the things
        # TODO: Generalize the algorithm for any shape of matrices
        _check_shape(activation_shape, lolif_type)
        _check_shape(kernel_shape, lolif_type)

        logging.debug(f"[LoLif super-class] setting type to {lolif_type.name}")
        self._set_indices(lolif_type)

        if activation_shape[self.HEIGHT] <= kernel_shape[self.HEIGHT]:
            # We need to check that the kernel is smaller than the activation,
            # so we can actually perform the computation
            raise ShapeError("The kernel is bigger than the activation! Try inverting the operators.")

        # These values are explained above the constructor
        self.activation_shape = activation_shape
        self.kernel_shape = kernel_shape
        self.number_of_batches = activation_shape[self.BATCH] if self.BATCH is not None else None
        self.number_of_channels = activation_shape[self.CHANNEL] if self.CHANNEL is not None else None
        self.activation_size = activation_shape[self.HEIGHT] # If the activation is 5x5, then this variable will be 5
        self.kernel_size = kernel_shape[self.HEIGHT] # If the activation is 3x3, then this variable will be 3
        self.output_size = self.activation_size - self.kernel_size + 1

        logging.debug(f"[LowLif super-class] shapes: "
                     f"\n\tactivation_shape {self.activation_shape}"
                     f"\n\tkernel_shape {self.kernel_shape}")
        logging.debug(f"[LowLif super-class] number of batches: {self.number_of_batches}, "
                     f"number of channels: {self.number_of_channels}")
        logging.debug(f"[LowLif super-class] activation_size: {self.activation_size}, kernel_size: {self.kernel_size}")

        self.stride_x = stride_x
        self.stride_y = stride_y
        logging.debug(f"[LowLif super-class] stride was set to {self.stride_y, self.stride_x}")

        # The output shape contains the processing for every batch
        self.lifted_output_shape = (self.output_size, self.output_size)
        logging.debug(f"[LowLif super-class] output shape for a single output is {self.lifted_output_shape}.")

        logging.info(f"[LoLif super-class] initialization done")

    """
    Maybe we can have these methods... I dunno
    
    def set_activation_shape(self, new_shape):
        _check_shape(new_shape)

        if self.kernel_shape[CHANNELS] != new_shape[CHANNELS]:
            raise ShapeError("The number of channels of the kernel must be the same as the activation.")

        # if the shape is valid, we update it
        self.activation_shape = new_shape
        # self.d = new_shape[CHANNELS] No need to update!
        self.matrix_side_length = new_shape[HEIGHT]
        self.output_side_length = self.matrix_side_length - self.kernel_side_length + 1

    def set_kernel_shape(self, new_shape):
        _check_shape(new_shape)

        if self.activation_shape[CHANNELS] != new_shape[CHANNELS]:
            raise ShapeError("The number of channels of the kernel must be the same as the activation.")

        # if the shape is valid, we update it
        self.kernel_shape = new_shape
        # self.d = new_shape[CHANNELS] No need to update!W
        self.kernel_side_length = new_shape[HEIGHT]
        self.output_side_length = self.matrix_side_length - self.kernel_side_length + 1
    """

    def _set_indices(self, lolif_type: LoLifType):
        indices = utils.input_indexes_mapping[lolif_type.value]
        self.HEIGHT = indices.get("HEIGHT")
        self.WIDTH = indices.get("WIDTH")
        self.CHANNEL = indices.get("CHANNEL")
        self.BATCH = indices.get("BATCH")

    def lower_activation(self, activation):
        raise NotImplementedError()

    def lower_kernel(self, kernel):
        raise NotImplementedError()

    def lift(self, o: np.ndarray):
        raise NotImplementedError()


class S_Im2Col(LoLif):

    def __init__(self, activation_shape, kernel_shape):
        super().__init__(activation_shape, kernel_shape, lolif_type=LoLifType.SINGLE)

        m = self.kernel_size * self.activation_size
        self.lowered_activation_shape = (self.output_size, m)
        self.lowered_kernel_shape = (m, self.output_size)
        self.lifted_output_shape = (self.output_size, self.output_size)

        logging.debug(f"[S_Im2Col] shapes report\n\tlowered_activation: {self.lowered_activation_shape}"
                     f"\n\tlowered_kernel_shape: {self.lowered_kernel_shape}"
                     f"\n\tlifted_output_shape: {self.lifted_output_shape}")

        del m

    def lower_activation(self, activation):
        logging.info(f"[S_Im2Col] starting lowering_activation")
        _check_tensor_against_shape(activation, self.activation_shape)

        if not isinstance(activation, np.ndarray):
            logging.warning(f"[S_Im2Col] the argument object activation is not instance of numpy.ndarray. This can "
                            f"cause problems with the computation.")
            activation = np.array(activation)  # Convert to numpy array

        result = np.zeros(self.lowered_activation_shape, dtype=activation.dtype)  # Prepare output

        for r in range(self.output_size):
            result[r, :] = activation[r:r+self.kernel_size, :].flatten("F")

        logging.debug(f"lowered_activation: \n{result}")

        logging.info(f"[S_Im2Col] lowering_activation done")

        del activation, r

        return result

    def lower_kernel(self, kernel):
        # Stack all unfolded kernels row by row

        if not isinstance(kernel, np.ndarray):
            logging.warning(f"[S_Im2Col] the argument object kernel is not instance of numpy.ndarray. This can "
                            f"cause problems with the computation.")
            kernel = np.array(kernel)

        logging.info(f"[S_Im2Col] starting lowering_kernel")
        output = np.zeros(self.lowered_kernel_shape, dtype=kernel.dtype)
        col = kernel.flatten("F")
        l = self.kernel_size
        m = len(col)
        for c in range(self.lowered_kernel_shape[1]):
            s = c*l
            output[s:s+m, c] = col

        del col, l, m, kernel

        logging.debug(f"lowered_kernel: \n{output}")

        logging.info(f"[S_Im2Col] lowering_kernel done")
        return output

    def lift(self, o: np.ndarray):
        return o.reshape((self.output_size, self.output_size), order="C")


class C_SlimKernel(LoLif):
    """
    This class implements an expensive lowering whose kernel becomes _slim_. This is because the kernels
    are flattened and then concatenated together, so the lowered kernel will only have 1 column
    """

    def __init__(self, activation_shape, kernel_shape):
        super().__init__(activation_shape, kernel_shape, lolif_type=LoLifType.CHANNELED)
        # New shapes will be the following
        self.lowered_activation_shape = (
            self.output_size ** 2,
            self.kernel_size ** 2 * self.number_of_channels
        )
        self.lowered_kernel_shape = (self.kernel_size**2 * self.number_of_channels, 1)

    def lower_activation(self, activation):
        _check_tensor_against_shape(activation, self.activation_shape)

        result = np.zeros(self.lowered_activation_shape)

        # input transformation
        for r in range(self.output_size):
            for c in range(self.output_size):
                result[c * self.output_size + r, :] = activation[:, r:r + self.kernel_size,
                                                             c:c + self.kernel_size].flatten()

        return result

    def lower_kernel(self, kernel):
        _check_tensor_against_shape(kernel, self.kernel_shape)

        # kernel transformation
        result = np.zeros(self.lowered_kernel_shape)
        result[:, 0] = kernel.flatten()
        return result

    def lift(self, o: np.ndarray):
        return o.reshape((self.output_size, self.output_size), order="F")

'''
These two are gonna be implementation of method 1 of the paper https://arxiv.org/abs/1504.04343

class S_ChannelStackingIm2Col(LoLif):
    """
    This class implements an Im2Col in such a way that each channel is processed separately by stacking
    it on the height.
    """

    def _im2col(self, img):
        result = np.zeros((self.output_side_length ** 2, self.kernel_side_length ** 2))
        for r in range(self.output_side_length):
            for c in range(self.output_side_length):
                result[c * self.output_side_length + r, :] = img[r:r + self.kernel_side_length,
                                                             c:c + self.kernel_side_length].flatten()

        return result

    def __init__(self, activation_shape, kernel_shape):
        logging.info(f"[S_ChannelStackingIm2Col] initizialization...")
        super().__init__(activation_shape, kernel_shape)

        self.lowered_activation_shape = (
            self.number_of_channels * self.output_side_length ** 2, 
            self.kernel_side_length ** 2
        )
        self.lowered_kernel_shape = (self.kernel_side_length ** 2, self.number_of_channels)
        self._set_indices(LoLifType.SINGLE)

        logging.info(f"[N_ChannelStackingIm2Col] lowered_activation_shape is {self.lowered_activation_shape}.")
        logging.info(f"[N_ChannelStackingIm2Col] lowered_kernel_shape is {self.lowered_kernel_shape}.")

    def lower_activation(self, activation):
        logging.info(f"[ChannelStackingIm2Col] lowering activation...")
        _check_tensor_against_shape(activation, self.activation_shape)
        result = np.zeros(self.lowered_activation_shape)

        m_square = self.output_side_length ** 2
        logging.info(f"[ChannelStackingIm2Col] processing im2col for each channel")
        for i in range(self.number_of_channels):
            result[i * m_square: (i + 1) * m_square, :] = self._im2col(activation[i])

        logging.info(f"[ChannelStackingIm2Col] activation lowered successfully")
        return result

    def lower_kernel(self, kernel):
        # kernel transformation
        logging.info(f"[ChannelStackingIm2Col] lowering kernel...")
        _check_tensor_against_shape(kernel, self.kernel_shape)

        result = np.zeros(self.lowered_kernel_shape)
        logging.info(f"[ChannelStackingIm2Col] lowering the kernel channel-wise")
        for i in range(self.number_of_channels):
            result[:, i] = kernel[i, :, :].flatten()

        logging.info(f"[ChannelStackingIm2Col] kernel lowered successfully")
        return result

    def lift(self, o: np.ndarray):
        logging.info(f"[ChannelStackingIm2Col] lifting the result...")
        result = np.zeros(self.lifted_output_shape)
        m_square = self.output_side_length ** 2
        for i in range(self.number_of_channels):
            result[i, :, :] = o[i * m_square:(i + 1) * m_square, :].reshape(
                (self.output_side_length, self.output_side_length), order="F")

        logging.info(f"[ChannelStackingIm2Col] result lifted successfully!")
        return result

'''