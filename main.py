import numpy as np
import scipy.signal as sgn
import torch
from utils import *

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
        T_inv = np.linalg.inv(projectionMatrix)

        s = np.array([ injection["x"], injection["y"], injection["t"] ])
        nu = T_inv @ s
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

if __name__ == "__main__":
    a = np.array([[1, 2], [3,4]])
    b = np.array([[5,6], [7,8]])
  
    c = matmul(a,b)
    print("expected output")
    print(c)

    C = np.matmul(a,b)
    print("ground truth")
    print(C)

    a = np.array([[1,2,3], [4,5,6], [7,8,9]])
    #a = np.ones((3,3))
    b = np.array([[1,1],[1,1]])

    print("Computing convolution between A:")
    print(a)
    print("and B:")
    print(b)
    
    c = convolution(a,b)
    print("expected")
    print(c)

    aT = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    bT = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).type_as(torch.ones(1, dtype=torch.double))
    print("ground truth")
    C = torch.nn.functional.conv2d(aT, bT)
    print(C)

