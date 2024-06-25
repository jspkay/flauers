import numpy as np
import logging
import numba.cuda as cuda

from . import utils

"""
This file contains tiling strategies to support matrices that are 
bigger than the array itself.

Take a look at this
https://www.mathsisfun.com/geometry/tessellation.html
"""

# TODO: Insert different types of tiling. It is possible to schedule also depending on the faults

class Tiling:
    """
    TODO: description
    """

    def __init__(self, A_shape, B_shape, N1, N2, N3):
        """
        It is an iterator simulating the tiling of the inputs. The input matrices are A and B 
        such that it is not possible to perform A@B directly because of the constraints N1, N2, N3.

        If A.shape[0] > N1 then we need tiling. Otherwise the thing it's processed at once.
        If B.shape[1] > N2 then we tile.
        If A.shape[1] > N3 (or equivalently B.shape[0] > N3) then we tile


        Parameters
        ---
        A_shape: matrix input A shape
        B_shape: matrix input B shape
        N1: Tile height
        N2: Tile width
        N3: Tile depth
        """
        
        # TODO: Add the checks for len(A.shape) == 2 and such

        self.A_shape = A_shape
        self.B_shape = B_shape

        self.N1 = N1
        self.N2 = N2
        self.N3 = N3

        self.k = -1
        self.i = 0
        self.j = 0
    
    def __iter__(self):
        ar, ac = self.A_shape
        br, bc = self.B_shape

        # How many columns ? (column limit)
        self.ilim = np.ceil( ar / self.N1 )
        
        # How many rows ? (row limit)
        self.jlim = np.ceil( bc / self.N2 )

        # how many iterations  per value ? (iteration limit)
        self.klim = np.ceil( ac / self.N3 )

        return self

    def __next__(self):
        # Let's only do two dimensions for now

        self.k += 1
        if self.k >= self.klim:
            self.i += 1
            self.k = 0
            if self.i >= self.ilim:
                self.i = 0
                self.j += 1

        if self.j >= self.jlim:
            raise StopIteration

        
        istart = self.i * self.N1
        jstart = self.j * self.N2
        kstart = self.k * self.N3

        istop = (self.i + 1) * self.N1
        jstop = (self.j+1) * self.N2
        kstop = (self.k+1) * self.N3


        return istart, jstart, kstart