import numpy as np
import utils
import solver
import math
import copy
from tqdm import tqdm
from scipy.special._logsumexp import logsumexp
from multiprocessing import Pool
import time
np.seterr(divide='ignore', invalid='ignore')
np.warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)

def calcNegLogPost(w, trajInfo, mdp, options):
    llh, grad1= calcLogLLH(w, trajInfo, mdp, options)
    # prior, grad2 = calcLogPrior(w, options)
    grad1 = np.reshape(grad1, (mdp.nFeatures,1))
    # grad2 = np.reshape(grad2, (mdp.nFeatures,1))
    # grad = grad1 + grad2
    # post = prior + llh
    # if(options.solverMethod == 'scipy'):
    #     grad = -grad
    #     post = -post

    if np.isinf(llh) or np.isinf(-llh):
        # print('ERROR: prior: %f, llh: %f, eta: %f, w: %f %f \n', prior, llh, options.eta, np.amin(w), np.amax(w))
        print('ERROR: prior: %f, llh: %f, eta: %f, w: %f %f \n', llh, options.eta, np.amin(w), np.amax(w))
    return llh, grad1

# def calcLogPrior(w, options):
#     if options.priorType == 'Gaussian':
#         x = w - options.mu
#         prior = np.sum(np.matmul(np.transpose(x), x) * -1 / 2 * math.pow(options.sigma, 2))
#         grad = -x / math.pow(options.sigma, 2)
#     else:
#         prior = math.log(1)
#         grad = np.zeros(w.shape)
        
#     return prior, grad

def calcLogLLH(w, trajInfo, mdp, options):

    mdp = utils.convertW2R(w, mdp)
    solver.Qaveraging(mdp,trajInfo) 
    # piL, VL, QL, H = solver.piMDPToolbox(mdp) 
    # dQ = calcGradQ(piL, mdp)
    QL = mdp.QL
    dQ = mdp.dQ
    nF = mdp.nFeatures
    nS = mdp.nStates
    nA = mdp.nActions
    eta = options.eta
    BQ = eta * QL   
    BQSum = np.reshape(logsumexp(BQ, axis=1),(nS,1))
    NBQ = BQ
    
    NBQ = NBQ - BQSum   # Log softmax policy

    llh = 0
    for i in range(len(trajInfo.cnt)):
        s = trajInfo.cnt[i, 0]
        a = trajInfo.cnt[i, 1]
        n = trajInfo.cnt[i, 2]
        llh += NBQ[s, a]*n

    # Soft-max policy
    pi_sto = np.exp(NBQ) 

    # calculate dlogPi/dw
    dlogPi = np.zeros((nF, nS * nA))
    z = np.zeros((nS,nA))
    for f in range(nF):
        x = np.reshape(dQ[f, :, :], (nS, nA), order='F')
        y = np.sum(np.multiply(pi_sto, x), axis=1).reshape(nS,1)
        z = eta * (x - y)  
        dlogPi[f, :] = np.reshape(z, (1, nS * nA), order='F')  

    # Calculating the gradient of the reward function
    grad = np.zeros(nF)
    for i in range(len(trajInfo.cnt)):
        s = trajInfo.cnt[i, 0]
        a = trajInfo.cnt[i, 1]
        n = trajInfo.cnt[i, 2]
        j = (a) * nS+s
        grad += n*dlogPi[:, j] 
    return llh, grad

# def calcGradQ(piL, mdp):
#     # Calculate Gradient Q value
#     nS = mdp.nStates
#     nA = mdp.nActions
#     Epi = np.zeros((nS, nS * nA)).astype(int)

#     act_ind = np.reshape(piL * nS + np.arange(0, nS).reshape((nS , 1)), nS)
    
#     for i in range(nS):
#         Epi[i, act_ind[i]] = 1

#     dQ = np.linalg.lstsq(np.eye(nS * nA) - np.matmul(mdp.T, Epi), mdp.F)[0]
#     return np.transpose(dQ)