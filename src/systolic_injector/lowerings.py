import numpy as np


class ShapeError(Exception):

    def __init__(self):
        pass


def _check_shape(t):
    a = t.shape
    if len(a) != 3:
        raise ShapeError

    # We assume the first is the batch size
    if a[1] != a[2]:
        raise ShapeError


class LoweringLifting:
    n = 0  # size of input matrix
    k = 0  # size of kernel matrix
    m = 0  # size of the output
    d = 0  # number of channels

    def __init__(self, activation, kernel):
        # Check the size of the things
        _check_shape(activation)
        _check_shape(kernel)

        # The number of convolution has to be the same (for now...)
        # TODO: Generalize the approach to perform the convolution between every combination
        # TODO: Implement batch size
        if activation.shape[0] != kernel.shape[0]:
            raise ShapeError
        self.d = activation.shape[0]

        self.activation = activation
        self.kernel = kernel

        self.n = self.activation.shape[1]
        self.k = self.kernel.shape[1]
        self.m = self.n - self.k + 1

        self.activation_shape = None
        self.kernel_shape = None

    def get_activation(self):
        raise NotImplementedError()

    def get_kernel(self):
        raise NotImplementedError()

    def lift(self, o: np.ndarray):
        raise NotImplementedError()


class SlimKernel(LoweringLifting):

    def __init__(self, activation, kernel):
        super().__init__(activation, kernel)
        self.activation_shape = (self.m ** 2, self.k ** 2 * self.d)
        self.kernel_shape = (self.k ** 2 * self.d, 1)

    def get_activation(self):
        result = np.zeros(self.activation_shape)
        # input transformation

        for r in range(self.m):
            for c in range(self.m):
                result[c * self.m + r, :] = self.activation[:, r:r + self.k, c:c + self.k].flatten()

        return result

    def get_kernel(self):
        # kernel transformation
        result = np.zeros(self.kernel_shape)
        result[:,0] = self.kernel.flatten()
        return result

    def lift(self, o: np.ndarray):
        return o.reshape((self.m, self.m), order="F")


class ExpensiveLowering(LoweringLifting):
    def __init__(self, activation, kernel):
        super().__init__(activation, kernel)
        self.kernel_shape = ( self.m**2, self.d )
        self.activation_shape = (self.m ** 2, self.k ** 2 * self.d)

    def get_activation(self):
        shape = (self.m ** 2, self.k ** 2 * self.d)
        result = np.zeros(shape)
        # input transformation

        for r in range(self.m):
            for c in range(self.m):
                result[c * self.m + r, :] = self.activation[:, r:r + self.k, c:c + self.k].flatten()

        return result

    def get_kernel(self):
        # kernel transformation
        shape = ( self.m**2, self.d )
        result = np.zeros(shape)

        for i in range(self.d):
            result[:,i] = self.kernel[i, :, :].flatten()

        return result

    def lift(self, o: np.ndarray):
        return o.reshape((self.m, self.m), order="F")