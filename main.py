import numpy as np

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
        print("index cannot have value " + str(index))
        assert(False)

def matmul(A, B):
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
    print("N1 is " + str(N1))
    print("N2 is " + str(N2))
    print("N3 is " + str(N3))

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
            # b[i, j, k] = B[b_j-1,b_k-1]

    # actual computations
    for i in range(1, N1):
        for j in range(1, N2):
            for k in range(1, N3):
                # print(str(i) + " " + str(j) + " " + str(k))
                a[i,j,k] = a[i, j-1, k]
                b[i,j,k] = b[i-1, j, k]
                c[i,j,k] = c[i, j, k-1] + a[i,j-1,k] * b[i-1,j,k]
    print("a is")
    printMatrixInIndex(a, 1)
    print("b is")
    printMatrixInIndex(b, 0)
    print("c is")
    printMatrixInIndex(c,2)

    # output
    C = c[1:,1:,N3-1]
    #print(c)

    return C

if __name__ == "__main__":
    a = np.array([[1, 2], [3,4]])
    b = np.array([[5,6], [7,8]])

    print(a)
    print(b)

    c = matmul(a,b)
    print("expected output")
    print(c)

    C = np.matmul(a,b)
    print("ground truth")
    print(C)
