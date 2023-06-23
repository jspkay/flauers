import numpy as np


class ShapeError(Exception):

    def __init__(self, message):
        super().__init__(self, message)


# indexes for the shapes
CHANNELS = 0
HEIGHT = 1
WIDTH = 2


def _check_shape(a):
    """
    This function checks that the shape we give in input have the correct structure.
    Specifically:
        - we want exactly three dimensions (width, height and channel)
        - we only accept square matrices (i.e. width == height)

    This will be expanded in the future. In the end, this function will also check for the batch size
    """
    if len(a) != 3:  # We need three dimensions (height, width and channel)
        raise ShapeError("The tensor has to be 3-dimensional: the first for the channel, second and third are height "
                         "and width")

    # We assume the first is the batch size
    if a[HEIGHT] != a[WIDTH]:  # only square matrices allowed!
        raise ShapeError("The width and the height of the tensor must be the same! Only square matrices are allowed")


def _check_tensor_against_shape(tensor, shape):
    t_shape = tensor.shape
    if len(t_shape) != len(shape):
        raise ShapeError("The tensor you want to lower has a wrong shape!")


class LowLif:  # Lowering-Lifting class
    activation_shape = None  # These two variables are used to check
    kernel_shape = None  # that the lowering we want to do is legal

    low_activation_shape = None  # These are used by implementing classes to infer the lowered shapes
    low_kernel_shape = None
    lift_output_shape = None

    n = 0  # size of input matrix (width and height per channel)
    k = 0  # size of kernel matrix (width and height per channel)
    m = 0  # size of the output (width and height of the input per channel), specifically it is n - m + 1
    d = 0  # number of channels

    def __init__(self, activation_shape: tuple, kernel_shape: tuple):
        # Check the size of the things
        # TODO: Generalize the algorithm for any shape of matrices
        _check_shape(activation_shape)
        _check_shape(kernel_shape)

        if activation_shape[HEIGHT] >= kernel_shape[
            HEIGHT]:  # We need to check that the kernel is smaller than the activation so
            # we can actually perform the computation
            raise ShapeError("The kernel is bigger than the ")

        # TODO: Implement batch size (i.e. a fourth dimension, since the first for now is the channel)
        self.activation_shape = activation_shape
        self.kernel_shape = kernel_shape
        if activation_shape[CHANNELS] != kernel_shape[CHANNELS]:  # We need a kernel for each channel of the activation
            raise ShapeError("The number of channels of the kernel must be the same as the activation.")

        # These values are explained above the constructor
        self.d = activation_shape[CHANNELS]
        self.n = activation_shape[HEIGHT]
        self.k = kernel_shape[HEIGHT]
        self.m = self.n - self.k + 1

    def set_activation_shape(self, new_shape):
        _check_shape(new_shape)

        if self.kernel_shape[CHANNELS] != new_shape[CHANNELS]:
            raise ShapeError("The number of channels of the kernel must be the same as the activation.")

        # if the shape is valid, we update it
        self.activation_shape = new_shape
        # self.d = new_shape[CHANNELS] No need to update!
        self.n = new_shape[HEIGHT]
        self.m = self.n - self.k + 1

    def set_kernel_shape(self, new_shape):
        _check_shape(new_shape)

        if self.activation_shape[CHANNELS] != new_shape[CHANNELS]:
            raise ShapeError("The number of channels of the kernel must be the same as the activation.")

        # if the shape is valid, we update it
        self.kernel_shape = new_shape
        # self.d = new_shape[CHANNELS] No need to update!W
        self.k = new_shape[HEIGHT]
        self.m = self.n - self.k + 1

    def lower_activation(self, activation):
        raise NotImplementedError()

    def lower_kernel(self, kernel):
        raise NotImplementedError()

    def lift(self, o: np.ndarray):
        raise NotImplementedError()


class SlimKernel(LowLif):
    """
    This class implements an expensive lowering whose kernel becomes _slim_. This is because the kernels
    are flattened and then concatenated together, so the lowered kernel will only have 1 column
    """

    def __init__(self, activation_shape, kernel_shape):
        super().__init__(activation_shape, kernel_shape)
        # New shapes will be the following
        self.low_activation_shape = (self.m ** 2, self.k ** 2 * self.d)
        self.low_kernel_shape = (self.k ** 2 * self.d, 1)

    def lower_activation(self, activation):
        _check_tensor_against_shape(activation, self.activation_shape)

        result = np.zeros(self.low_activation_shape)

        # input transformation
        for r in range(self.m):
            for c in range(self.m):
                result[c * self.m + r, :] = activation[:, r:r + self.k, c:c + self.k].flatten()

        return result

    def lower_kernel(self, kernel):
        _check_tensor_against_shape(kernel, self.low_activation_shape)

        # kernel transformation
        result = np.zeros(self.low_kernel_shape)
        result[:, 0] = kernel.flatten()
        return result

    def lift(self, o: np.ndarray):
        return o.reshape((self.m, self.m), order="F")


class ExpensiveLowering(LowLif):
    """
    This class implements an expensive lowering as well as in the SlimKernel class, but in this case
    the lowered kernel is _fatter_ since each channel is flattened but has its own column
    """

    def __init__(self, activation_shape, kernel_shape):
        super().__init__(activation_shape, kernel_shape)
        self.low_kernel_shape = (self.m ** 2, self.d)
        self.low_activation_shape = (self.m ** 2, self.k ** 2 * self.d)

    def lower_activation(self, activation):
        _check_tensor_against_shape(activation, self.activation_shape)

        result = np.zeros(self.low_activation_shape)

        # input transformation
        for r in range(self.m):
            for c in range(self.m):
                result[c * self.m + r, :] = activation[:, r:r + self.k, c:c + self.k].flatten()

        return result

    def lower_kernel(self, kernel):
        # kernel transformation
        shape = (self.m ** 2, self.d)
        result = np.zeros(shape)

        for i in range(self.d):
            result[:, i] = kernel[i, :, :].flatten()

        return result

    def lift(self, o: np.ndarray):
        return o.reshape((self.m, self.m), order="F")
