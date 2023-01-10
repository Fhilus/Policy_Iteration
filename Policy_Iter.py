import petsc4py
import sys
import numpy as np
import time
import copy
import Policy_Iter_Variables
import Policy_Iter_Solvers
from random import seed
from random import randint
from numpy import linalg as LA
petsc4py.init(sys.argv)
from petsc4py import PETSc

def get_P(muh, n):
    P = np.zeros((n, n))
    for i in range(n):
        P[i] = Probabilities[muh[i], i, :]

    return P

def get_g(muh, n):
    g = np.zeros((n,1))
    for i in range(n):
        g[i] = cost[i][muh[i]]
    return g

def PolicyEvaluation(n, alpha, g, P, method):
    A = np.identity(n) - alpha*P
    J = np.zeros((n,1))
    if method == 1:
        J = Policy_Iter_Solvers.gmres1(n, J, A, g)
    if method == 2:
        print("solving with np.linalg.solve()")
        J = np.linalg.solve(A, g)
    if method == 3:
        J = Policy_Iter_Solvers.petsc_gmres(n, A, g)
    if method == 4:
        J = Policy_Iter_Solvers.seidel(A, J, g, n, 1e-06)
    if method == 5:
        J = Policy_Iter_Solvers.jacobi(J, A, g, 1e-06)
    if method == 6:
        J = Policy_Iter_Solvers.petsc_minres(n, A, g)
    print("J: ", J)
    return J

def PolicyImprovement(n, muh, alpha, J):
    m = Probabilities.shape[0]
    TJ = np.zeros((n,1)) #TJ are the expected cost using the cost function J
    for k in range(m):
        for i in range(n):
            TJ_calc = cost[i][k]
            for j in range(n):
                TJ_calc += alpha*Probabilities[k][i][j]*J[j][0]
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
    Probabilities, cost = Policy_Iter_Variables.setup(1) #choose model from Policy_Iter_Variables.py
    n = Probabilities.shape[1]
    muh = np.zeros((n,1))
    for i in range(1,6): #minres not included because it isnt fixed yet
        muh_old = copy.deepcopy(muh)
        start_time = time.time()
        muh = PolicyIteration(n, i) #i is the method to be used
        print("--- %s seconds ---" % (time.time() - start_time))
        print((muh == muh_old).all()) #test to see if PolicyIteration come to the same result depending on their solver
    #for testing the solver methods use the following
    #A = np.array([[7., 5., 1.],
    #              [3., 8., 4.],
    #              [1., 6., 13.]])
    #b = np.array([[3.],[4.],[5.]])
    #x = np.zeros((3,1))
    #J1 = gmres1(3, x, A, b)
    #J2 = np.linalg.solve(A, b)
    #J3 = petsc_gmres(3, A, b)
    #J4 = petsc_minres(3, A, b)
    #J5 = jacobi(x, A, b, 1e-06)
    #J6 = seidel(A, x, b, 3, 1e-06)
    #print(J1)
    #print(J2)
    #print(J3)
    #print(J4)
    #print(J5)
    #print(J6)
