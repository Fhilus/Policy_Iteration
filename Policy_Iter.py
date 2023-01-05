import petsc4py
import sys
import numpy as np
import time
import copy
import Policy_Iter_Variables
from random import seed
from random import randint
from numpy import linalg as LA
petsc4py.init(sys.argv)
from petsc4py import PETSc

def get_P(muh, n):
    P = np.zeros((n, n))
    for i in range(n):
        P[i] = Policy_Iter_Variables.Probabilities[muh[i], i, :]

    return P

def get_g(muh, n):
    g = np.zeros((n,1))
    for i in range(n):
        g[i] = Policy_Iter_Variables.cost[i][muh[i]]
    return g

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
            print(x)
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
    A_mat = PETSc.Mat().createDense(A.shape, array=A) #initialise PATSc Matrix from numpy array
    b_vec = PETSc.Vec().createSeq(n) #iniitalise PATSc vector from numpy array
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
    A_mat = PETSc.Mat().createDense(A_sym.shape, array=A_sym) #initialise PATSc Matrix from numpy array
    b_vec = PETSc.Vec().createSeq(2*n) #iniitalise PATSc vector from numpy array
    b_vec.setValues(range(2*n), b_big)
    x = PETSc.Vec().createSeq(2*n) #PETSc stores solution here

    ksp = PETSc.KSP().create() #create environment for solving method
    ksp.setType('minres') #choose method to be used

    pc = ksp.getPC()
    pc.setType('none') #no preconditioning

    ksp.setOperators(A_mat)
    ksp.setFromOptions()

    chosen_solver = ksp.getType()
    print(f"Solving with {chosen_solver:}")

    ksp.solve(b_vec, x)

    sol = x.getArray()[:n]

    return sol.reshape(n,1)

def PolicyEvaluation(n, alpha, g, P, method):
    A = np.identity(n) - alpha*P
    J = np.zeros((n,1))
    if method == 1:
        J = gmres1(n, J, A, g)
    if method == 2:
        print("solving with np.linalg.solve()")
        J = np.linalg.solve(A, g)
    if method == 3:
        J = petsc_gmres(n, A, g)
    if method == 4:
        J = petsc_minres(n, A, g)
    if method == 5:
        J = jacobi(J, A, g, 1e-06)
    if method == 6:
        J = seidel(A, J, g, n, 1e-06)
    print("J: ", J)
    return J

def PolicyImprovement(n, muh, alpha, J):
    m = Policy_Iter_Variables.Probabilities.shape[0]
    TJ = np.zeros((n,1)) #TJ are the expected cost using the cost function J
    for k in range(m):
        for i in range(n):
            TJ_calc = Policy_Iter_Variables.cost[i][k]
            for j in range(n):
                TJ_calc += alpha*Policy_Iter_Variables.Probabilities[k][i][j]*J[j][0]
            if k == 0:
                TJ[i][0] = TJ_calc
                muh[i][0] = k
            if TJ[i][0] >= TJ_calc:
                TJ[i][0]= TJ_calc
                muh[i][0] = k
    return TJ, muh

def PolicyIteration(n, method):
    muh = np.zeros((n, 1), int) #initial stationary policy
    g = get_g(muh, n) #cost vector depending on stationary policy
    P = get_P(muh, n) #Probablity matrixs depending on stationary policy
    alpha = 0.8 #discount factor
    J_old = np.zeros((n, 1))
    J = PolicyEvaluation(n, alpha, g, P, method)
    TJ, muh = PolicyImprovement(n, muh, alpha, J)
    iter = 0
    norm = LA.norm(TJ-J)
    while iter < 100 and norm > 1e-06 and (J != J_old).any():
        J_old = copy.deepcopy(J)
        g = get_g(muh, n)
        P = get_P(muh, n)
        J = PolicyEvaluation(n, alpha, g, P, method)
        TJ, muh = PolicyImprovement(n, muh, alpha, J)
        iter += 1
        norm = LA.norm(TJ - J)
        print(iter, ": ||TJ - J|| = ", norm)

    print("muh: ", muh)
    return muh



if __name__ == "__main__":
    n = Policy_Iter_Variables.Probabilities.shape[1]
    #method = 2 #chooses a method for solving the linear system
    muh = np.zeros((n,1))
    for i in range(1,7):
        muh_old = copy.deepcopy(muh)
        start_time = time.time()
        muh = PolicyIteration(n, i)
        print("--- %s seconds ---" % (time.time() - start_time))
        print((muh == muh_old).all())
    #for testing the solver methods use the following
    # A = np.array([[7., 5., 1.],
    #               [3., 8., 4.],
    #               [1., 6., 13.]])
    # b = np.array([[3.],[4.],[5.]])
    # x = np.zeros((3,1))
    # J1 = gmres1(3, x, A, b)
    # J2 = np.linalg.solve(A, b)
    # J3 = petsc_gmres(3, A, b)
    # J4 = petsc_minres(3, A, b)
    # J5 = jacobi(x, A, b, 1e-06)
    # J6 = seidel(A, x, b, 3, 1e-06)
    # print(J1)
    # print(J2)
    # print(J3)
    # print(J4)
    # print(J5)
    # print(J6)
