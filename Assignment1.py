import csv
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time



def read_matrix(file_loc):
    matrix = []
    with open (file_loc, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            matrix.append(row)
    size = len(matrix)
    return matrix, size

def generate_a(size):
    a = -1*np.ones(size-1)
    b = 2*np.ones(size)
    c = -1*np.ones(size-1)
    A = sp.diags([a, b, c], offsets=[-1, 0, 1], format = 'csr', dtype=np.float64)
    return A

def generate_b(size):
    b = np.ones(size)
    return b


def main():
    # Define different parameters for the solver
    # method = 'spsolve' # Options are the methods listed for scipy.sparse.linalg. For now, only spsolve
    load_matrixA = None # Options: None, file location of .csv file with matrix A
    load_matrixB = None # Options: None, file location of .csv file with matrix B
    n = int(2e6) # Size of the matrix
    A = []
    b = []

    if load_matrixA:
        A, n = read_matrix(load_matrixA)
    else:
        A = generate_a(n)

    if load_matrixB:
        b, b_n = read_matrix(load_matrixB)
        if b_n != n:
            print ('Matrices A and B have a different number of rows. Using a default B matrix')
            b = []
    if b == []:
        b = generate_b(n)

    start = time.time()


    x = spla.spsolve(A, b)
    res = np.linalg.norm(A.dot(x) - b)
    end = time.time()

    print ('x: ', x)
    print ('Residuals: ', res)
    print ('Time: ', end-start)


'''
    x = spla.bicg(A, b, rtol=0.05)
    res = np.linalg.norm(A.dot(x) - b)
'''



if __name__ == "__main__":
    main()