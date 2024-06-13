import numpy as np
import logging

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

    def __init__(self, A, B, N1, N2, N3):
        """
        It is an iterator simulating the tiling of the inputs. The input matrices are A and B 
        such that it is not possible to perform A@B directly because of the constraints N1, N2, N3.

        If A.shape[0] > N1 then we need tiling. Otherwise the thing it's processed at once.
        If B.shape[1] > N2 then we tile.
        If A.shape[1] > N3 (or equivalently B.shape[0] > N3) then we tile


        Parameters
        ---
        A: matrix input A 
        B: matrix input B 
        N1: Max width - A.shape[0] >= N1 -> 
        """
        
        self.A = A
        self.B = B
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
    
    def __iter__(self):
        return self

    def __next__(self):
