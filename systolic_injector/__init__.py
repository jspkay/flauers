import numpy as np
from .utils import *

output_stationary = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1] ])
row_stationary = np.array([[0, -1, 1], [-1, 1, 0], [1, 1, 1] ])

def printMatrixInIndex(A, index):
    assert len(A.shape) == 3

    a, b, c = A.shape

    if index == 0:
        for i in range(a):
            print(A[i,:,:])
    elif index == 1:
        for i in range(b):
            print(A[:,i,:])
    elif index == 2:
        for i in range(c):
            print(A[:,:,i])
    else:
        assert False, "index cannot have value " + str(index) 

def spaceTimeEquation(nu : np.ndarray, T : np.ndarray):
    """
        Args:
            nu -> 3D iteration vector 
            T -> space-time projection matrix
        Returns:
            eps -> a vector composed by [x, y, t] describing the position in space (x,y) and in time (t)
            of the operation corresponding to the given iteration vector
    """

    return T @ nu

def inverseSpaceTimeEquation(eps : np.ndarray, T : np.ndarray):
    """
        Args:
            eps -> space-time vector containing [x,y,t]
            T -> space-time projection matrix
        Returns:
            nu -> 3D iteration vector composed by [i, j, k]
    """

    Tinv = np.linalg.inv(T)
    nu = Tinv @ eps
    return nu

def matmul(A, B, injection=None, projectionMatrix=None):
    ''' 
    Performs the matrix multiplication between A and B.
    A must have size (ar, ac), B must have size (br, bc), with ac = br.
    The output C has size (ar, bc).
    '''

    ar, ac = A.shape
    br, bc = B.shape

    if ac != br:
        raise Exception("matrix not compatible!")

    N1 = ar+1
    N2 = bc+1
    N3 = br+1 # same as ac

    # Injection characterization
    shouldInject = injection != None
    
    if projectionMatrix is None and shouldInject:
        assert False, "Injection should be performed, but no space-time projection matrix has been given"

    if shouldInject:
        # T_inv = np.linalg.inv(projectionMatrix)

        s = np.array([ injection["x"], injection["y"], injection["t"] ])
        nu = inverseSpaceTimeEquation(s, projectionMatrix)
        #nu = T_inv @ s
        print("nu is ")
        print(nu)


    # TODO: replace this arrays with simpler structures that takes into account
    # only the actual data we are using, not the entire iteration vector space
    a = np.zeros((N1, N2, N3))
    b = np.zeros((N1, N2, N3))
    c = np.zeros((N1, N2, N3))

    # input operations
    j = 0
    for i in range(1, N1):
        for k in range(1, N3):
            ''' it's wrong here! You have to shift the matrix by one on the right,
            not the left! '''
            a_i = 1 if i == 0 else i
            a_k = 1 if k == 0 else k
            a[i,j,k] = A[a_i-1, a_k-1]
    i = 0
    for j in range(1, N2):
        for k in range(1, N3):
            b_j = 1 if j == 0 else j
            b_k = 1 if k == 0 else k
            b[i, j, k] = B[b_k-1,b_j-1]

    # actual computations
    for i in range(1, N1):
        for j in range(1, N2):
            for k in range(1, N3):
                # print(str(i) + " " + str(j) + " " + str(k))
                a[i,j,k] = a[i, j-1, k]
                b[i,j,k] = b[i-1, j, k]

                # Actual injection
                if shouldInject and i == nu[0] and j == nu[1] and k >= nu[2]:
                    print("Injecting a! Old value ", end=" ")
                    print(a[i,j,k], end=" ")
                    newValue = int(a[i,j,k]) ^ (1<<8)
                    a[i,j,k] = newValue
                    print("newValue ", newValue)

                c[i,j,k] = c[i, j, k-1] + a[i,j-1,k] * b[i-1,j,k]

    C = c[1:,1:,N3-1]

    return C



def convolution(A, B, N=-1):
    n, n1 = A.shape
    assert(n == n1)

    m, m1 = B.shape
    assert(m == m1)

    assert(m < n)

    if N == -1:
        N = n-m+1

    L = m*m+m
    print("L is ", L)

    A = transformation(A, N, m)
    print("A' is ", A)
    B = flattenKernel(B, N)
    print("B' is ", B)

    pm = np.array([[1, 0, 0],[0, 1, 0], [1,1,1]])
    inj = {"x": 1, "y": 1, "t": 0}

    return matmul(B,A, projectionMatrix=pm, injection=inj)

