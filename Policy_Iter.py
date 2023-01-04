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



def PolicyEvaluation(n, alpha, g, P, method):
    A = np.identity(n) - alpha*P
    J = np.zeros((n,1))
    if method == 1:
        J = gmres1(n, J, A, g)
    if method == 2:
        J = np.linalg.solve(A, g)
    if method == 3:
        J = petsc_gmres(n, A, g)
    print(J)
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



if __name__ == "__main__":
    n = Policy_Iter_Variables.Probabilities.shape[1]
    method = 2 #chooses a method for solving the linear system
    start_time = time.time()
    PolicyIteration(n, 1)
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    PolicyIteration(n, 3)
    print("--- %s seconds ---" % (time.time() - start_time))
