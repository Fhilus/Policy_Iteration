import petsc4py
import sys
import numpy as np
import time
import copy
from random import seed
from random import randint
from numpy import linalg as LA
petsc4py.init(sys.argv)
from petsc4py import PETSc


def seidel(A, x, b, n, tol, max_iter=100000):
    x_last = copy.deepcopy(x)
    L = np.tril(A)
    L_inv = np.linalg.inv(L)
    U = np.triu(A, +1)
    for k in range(max_iter):
        if (k < 1) or (LA.norm(x - x_last) > tol):
            x_last = copy.deepcopy(x)
            #calculate new x_i
            x = L_inv.dot(b - U.dot(x))
        else:
            i = max_iter
    print("solving with Gauss Seidel method")
    return x

def jacobi(x, A, b, tol, max_iter=100000): #solves Ax = b using Jacobi method
    # Create a vector of the diagonal elements of A
    # and subtract them from A
    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)
    R = A - D
    x_last = copy.deepcopy(x)

    # Iterate for N times
    for i in range(max_iter):
        if LA.norm(x_last - x) > tol or i < 1:
            x_last = copy.deepcopy(x)
            x = D_inv.dot((b - R.dot(x)))
        else:
            i = max_iter

    print("solving with Jacobi method")
    return x

def gmres1(n, init, A, b):
    r_0 = b - A.dot(init)
    beta = LA.norm(r_0)
    H = np.zeros((n, n))
    V = np.zeros((n, n))
    V[:, 0] = np.transpose(r_0 / beta)
    for j in range(n):
        w = A.dot(V[:,j])
        for i in range(j+1):
            H[i][j] = w.dot(V[:, i])
            w = w - H[i][j]*(V[:, i])
            if j+1 < n:
                H[j+1][j] = LA.norm(w)
                if H[j+1][j] == 0:
                    j = n
                else:
                    V[:, j+1] = w/H[j+1][j]
    y = np.linalg.lstsq(H, beta*(np.identity(n)[:, 0]), rcond=None)[0]
    x = init + V.dot(y)
    result = np.empty([n,1])
    for i in range(n):
        result[i] = x[i][i]

    print("solving with own gmres")
    return result

def petsc_gmres(n, A, b):
    A_mat = PETSc.Mat().createDense(A.shape, array=A) #initialise PETSc Matrix from numpy array
    b_vec = PETSc.Vec().createSeq(n) #iniitalise PETSc vector from numpy array
    b_vec.setValues(range(n), b)
    x = PETSc.Vec().createSeq(n) #PETSc stores solution here

    ksp = PETSc.KSP().create() #create environment for solving method
    ksp.setType('gmres') #choose method to be used

    pc = ksp.getPC()
    pc.setType('none') #no preconditioning

    ksp.setOperators(A_mat)
    ksp.setFromOptions()

    chosen_solver = ksp.getType()
    print(f"Solving with {chosen_solver:}")

    ksp.solve(b_vec, x)
    sol = x.getArray()

    return sol.reshape(n,1)

def petsc_minres(n, A, b):
    A_sym = np.block([[A, np.zeros((n,n))],
                      [np.zeros((n,n)), A.transpose()]])
    b_big = np.block([[b],[b]])
    A_mat = PETSc.Mat().createDense(A_sym.shape, array=A_sym) #initialise PETSc Matrix from numpy array
    b_vec = PETSc.Vec().createSeq(2*n) #iniitalise PATSc vector from numpy array
    b_vec.setValues(range(2*n), b_big)
    x = PETSc.Vec().createSeq(2*n) #PETSc stores solution here

    ksp = PETSc.KSP().create() #create environment for solving method
    ksp.setType('minres') #choose method to be used
    ksp.setTolerances(1e-05, 1e-50, 10000.0, 100000) #(atol, rtol, dtol, max iter)

    pc = ksp.getPC()
    pc.setType('none') #no preconditioning

    ksp.setOperators(A_mat)
    ksp.setFromOptions()

    chosen_solver = ksp.getType()
    print(f"Solving with {chosen_solver:}")

    ksp.solve(b_vec, x)
    #print(ksp.getTolerances())
    sol = x.getArray()[:n]

    return sol.reshape(n,1)
