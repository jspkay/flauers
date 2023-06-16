import numpy as np

def transformation(A, N, m, t=0):
    ''' t gives wether the transofrmation has to be vertial or horizontal'''

    n1 = N
    n2 = m*m+m

    n, n1 = A.shape
    assert n==n1, "A has to be square!"

    M = min(N, n-m+1)

    if t == 0:
        a = np.zeros((n2, n1))
        for i in range(M):
            a[:,i] = A[:,i:i+2].flatten("C")
    if t == 1:
        a = np.zeros((n1,n2))
        for i in range(M):
            a[i,:] = A[i:i+2, :].flatten("F")
    return a

def flattenKernel(B, N, t=0):
    m, m1 = B.shape
    assert m == m1, "B has to be a square matrix"
    
    L = m*m+m
    
    if t == 0:
        b = np.zeros([N,L])
        bflat = B.flatten("C")
        for i in range(N):
            pre = np.zeros([m*i])
            post = np.zeros([m * (N-i-1)  ])
            b[i,:] = np.concatenate([pre, bflat, post])

    if t == 1:
        b = np.zeros([L, N])
        bflat = B.flatten("F").reshape([1, m*m])
        for i in range(N):
            pre = np.zeros([1, m*i])
            post = np.zeros([1, m*(N-i-1)])
            b[:,i] = np.concatenate([pre, bflat, post], axis=1)

    return b
