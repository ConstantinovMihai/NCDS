""" Assignment 1 of NDCS course"""

import numpy as np
import control
import scipy.linalg as la
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# parameters
a = 5
b = 0
c = 5

# question 1.a, getting the gains
A = np.array([[5, -4.5], [0, 1]])
B = np.array([[0], [1]])

p1 = [-1+2j, -1-2j]
p2 = [-1, -2]
p3 = [1, 2]

k1 = control.place(A,B,p1)
k2 = control.place(A,B,p2)
k3 = control.place(A,B,p3)

print(f"The gains are {k1}, {k2}, {k3}.")

# question 1.b, stability ranges for the controllers
def getAtilda(A, B):
    # exponential of A_tilda
    A_tilda = np.hstack((A,B))
    zero_pad= np.zeros((1,3))
    A_tilda = np.vstack((A_tilda, zero_pad))
    return A_tilda


def getSystemMatrices(A, h):
    """ Implements the fancy trick and extracts F(h) and G(h) matrices
    """
    expAtilda = la.expm(A*h)
    F = expAtilda[:2,:2]
    G = expAtilda[:2,2]
    return F, G

# the next 3 functions are validated by reproducing the figure in the slides
def maxEigenvalue(F,G,K):
    """ Computes the max eigenvalue of F-G*K matrix
    """
   
    eigs, _ = np.linalg.eig(F - np.matmul(G.reshape(-1,1),K))
    eigs = np.sort(np.abs(eigs))

    return np.abs(eigs[1])

def getRhos(A, B, K, hmax=1, step=0.005):
    """ generates the array of maximum eigenvalues for a system
        with varying sampling time
    """
    hs = np.arange(0,hmax+step, step)
    rhos = []
    for h in hs:
        At  = getAtilda(A, B)
        F,G = getSystemMatrices(At, h)
        rho = maxEigenvalue(F, G, K)
        rhos.append(rho)
    return hs, np.array(rhos)


def plotRhos(hs, rhos, stab_limit=None, labelx = "h"):
    """ generates the plot of maximum eigenvalues sampling time h
    """
    plt.plot(hs, rhos)
    plt.title("Norm of the maximum eigenvalue")
    plt.grid()
    plt.xlim(min(hs), max(hs))
    plt.xlabel(labelx)
    plt.ylabel(r"$\rho$($F_{cl}(h)$)")
    plt.show()

    if stab_limit:
        return rhos[np.where(rhos==1)]

def constructF(a, b, c, h, tau):
    # according to the analytical solutions
    f00 = np.exp(a*h)
    f01 = (b+0.5) / (a+c) * (np.exp(a*h) - np.exp(-c*h))
    f02 = (b+0.5) / c * (np.exp(a*h)/a - np.exp(a*(h-tau))/a + 1/(a+c)*(-np.exp(a*h) + np.exp(a*(h-tau)) - np.exp(-c*(h-tau)) + np.exp(-c*h)))
    
    f10 = 0
    f11 = np.exp(-c*h)
    f12 = (1/c) * (np.exp(-c*(h-tau)) - np.exp(-c*h))

    f20 = 0
    f21 = 0
    f22 = 0

    F = np.array([[f00, f01, f02], [f10, f11, f12], [f20, f21, f22]])

    return F


def constructG(a, b, c, h, tau):
    g0 = (b + 0.5) / c * ((np.exp(a*(h-tau)) - 1) / a - 1/(a+c)*(np.exp(a*(h-tau)) - np.exp(-c*(h-tau))))
    g1 = 1/c * (1-np.exp(-c*(h-tau)))
    g2 = 1

    G = np.array([[g0], [g1], [g2]])

    return G

def constructF2(a, b, c, h, tau):
    # according to the analytical solutions
    f00 = np.exp(a*h)
    f01 = (b+0.5) / (a + c) * (np.exp(a*h) - np.exp(-c*h))
    f02 = (b + 0.5) / c * ((np.exp(a*(2*h-tau)) - 1) / a - 1/(a+c)*(np.exp(a*(2*h-tau)) - np.exp(-c*(2*h-tau))))
    f03 = (b + 0.5) / c * (np.exp(a*h)/a - np.exp(a*(2*h-tau))/a + 1/(a+c)*(-np.exp(a*h) + np.exp(a*(2*h-tau)) - np.exp(-c*(2*h-tau)) + np.exp(-c*h)))
    
    f10 = 0
    f11 = np.exp(-c*h)
    f12 = 1/c * (1-np.exp(-c*(2*h-tau)))
    f13 = (1/c) * (np.exp(-c*(2*h-tau)) - np.exp(-c*h))

    f20 = 0
    f21 = 0
    f22 = 0
    f23 = 0

    f30 = 0
    f31 = 0
    f32 = 1
    f33 = 0

    F = np.array([[f00, f01, f02, f03], [f10, f11, f12, f13], [f20, f21, f22, f23], [f30, f31, f32, f33]])

    return F


def constructG2(a, b, c, h, tau):
    g0 = 0
    g1 = 0
    g2 = 1
    g3 = 0

    G = np.array([[g0], [g1], [g2], [g3]])

    return G

# answer to Question 1
def question1():
    hs, rhos = getRhos(A, B, hmax=0.4, K=k3)
    plotRhos(hs, rhos)


# question 2
def question2(k = np.array([[-8.888888, 8, 0]])):
    hs   = np.arange(0, 0.4, 0.01)
    taus = np.arange(0, 0.6, 0.01)
    sol  = np.zeros((len(hs), len(taus)))

    for idxh, h in enumerate(hs):
        for idxtau, tau in enumerate(taus):
            if (tau < h):
                F = constructF(a, b, c, h, tau) 
                G = constructG(a, b, c, h, tau)
                eig_max = maxEigenvalue(F, G, k)
                sol[idxh][idxtau] = eig_max

            else:
                F = constructF2(a, b, c, h, tau) 
                G = constructG2(a, b, c, h, tau)
                k2 = np.append(k, 0).reshape(1, -1)
                    
                eig_max = maxEigenvalue(F, G, k2)
                sol[idxh][idxtau] = eig_max


    fig = plt.figure(figsize = (12,10))
    ax = plt.axes(projection='3d')

    X, Y = np.meshgrid(hs, taus)
    Z = sol.transpose()

    surf = ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)

    # Set axes label
    ax.set_xlabel('sampling time', labelpad=20)
    ax.set_ylabel('delay', labelpad=20)
    ax.set_zlabel('maximum eigenvalue', labelpad=20)

    fig.colorbar(surf, shrink=0.5, aspect=8)

    plt.show()


# choose h = 0.25

def computeLimitDelay(k, h=0.25, plot=False):
    taus = np.arange(0, 1, 0.01)
    max_eigs = []
    for tau in taus:
        F = constructF(a, b, c, h, tau)
        G = constructG(a, b, c, h, tau)
        max_eig = maxEigenvalue(F, G, k)
        max_eigs.append(max_eig)
    idx_min = np.where(np.array(max_eigs) > 1)[0]

    if plot:
        plotRhos(taus, max_eigs)
    
    # check no weird results are spewed out
    if len(idx_min) == 0 or idx_min[0] == 0:
        idx_min = [0]

    return taus[idx_min[0]]
    

def bayesOptTuning():
    """ Tune the dynamic controller using bayesian optimisation
    """
    def objective(param):
        k = np.array([[param['k1'], param['k2'], param['k3']]])
        unstable_tau = computeLimitDelay(k)
        return {'loss': -unstable_tau, 'status': STATUS_OK }
    
    param_hyperopt = {
        'k1' : hp.uniform('k1', -10, 10),
        'k2' : hp.uniform('k2', -10, 10),
        'k3' : hp.uniform('k3', -1, 1)
    }
    
    trials = Trials()
    best_param = fmin(objective, 
                      param_hyperopt, 
                      algo=tpe.suggest, 
                      max_evals=250, 
                      trials=trials)
    print(best_param)


if __name__ == '__main__':
    test = True
    if test:
        k_test = np.array([[-1.75,  -0.12, 0.54]])
        print(computeLimitDelay( k=k_test, h=0.25, plot=True))
    else:
        print(bayesOptTuning())

