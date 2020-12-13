from models import mdp
import numpy as np
import math
import utils
import mdptoolbox, mdptoolbox.example
from scipy import sparse
from scipy.special import logsumexp
np.seterr(divide='ignore', invalid='ignore')
from multiprocessing import Pool

def Qaveraging(mdp,trajInfo):
    unseen_sa = getUnseenSA(mdp, trajInfo)
    count = 0
    threshold = 0.0001
    maxchange = 2*threshold
    with Pool(processes = 5) as pool:
        while (count < 10000) and (maxchange > threshold):
            count += 1
            updateUnseenQvalues(mdp, unseen_sa)
            maxchange = (pool.apply_async(updateQ,(mdp,trajInfo))).get()
            # with Pool(processes = 5) as pool1:
            # updateUnseenQvalues(mdp, unseen_sa)
            # updateQ(mdp,trajInfo)
    print("Q update steps: ", count)

    count = 0
    maxchange = 2*threshold
    with Pool(processes = 5) as pool:
        while (count < 10000) and (maxchange > threshold):
            count += 1
            updateUnseenQgradients(mdp,unseen_sa)
            maxchange = (pool.apply_async(updategradQ,(mdp,trajInfo))).get()
            # updateUnseenQgradients(mdp, unseen_sa)
            # updategradQ(mdp,trajInfo)
    print("Q grad update steps: ", count)
    return

def getUnseenSA(mdp,trajInfo):
    sa_set = set()
    for s in range(mdp.nStates):
        for a in range(mdp.nActions):
            sa_set.add((s,a))
    
    nTrajs = trajInfo.nTrajs
    nSteps = trajInfo.nSteps
    for m in range(nTrajs):
        for h in range(nSteps-1):   # leaving the last step as unseen to set it with unseenSA gradient
            s = trajInfo.states[m, h]
            a = trajInfo.actions[m, h]
            if (s,a) in sa_set:
                sa_set.remove((s,a))
    return sa_set

def updateQ(mdp,trajInfo):
    maxchange = 0
    nTrajs = trajInfo.nTrajs
    nSteps = trajInfo.nSteps 
    oldQvalue = 0
    newQvalue = 0
    for m in range(nTrajs):
        for h in range(nSteps-1):
            s = trajInfo.states[m, h]
            a = trajInfo.actions[m, h]
            sprime = trajInfo.states[m, h+1]
            r = mdp.reward[s,a]
            oldQvalue = mdp.QL[s,a]
            mdp.QL[s,a] = mdp.QL[s,a] + mdp.alpha*(r + mdp.discount * np.average(mdp.QL[sprime][:]) - mdp.QL[s,a])
            newQvalue = mdp.QL[s,a]
            if abs(oldQvalue - newQvalue) > maxchange:
                maxchange = abs(oldQvalue - newQvalue)
    return maxchange

def updategradQ(mdp,trajInfo):
    maxchange = 0
    nF = mdp.nFeatures 
    nTrajs = trajInfo.nTrajs
    nSteps = trajInfo.nSteps
    dQold = 0
    dQnew = 0
    for m in range(nTrajs):
        for h in range(nSteps-1):
            s = trajInfo.states[m, h]
            a = trajInfo.actions[m, h]
            sprime = trajInfo.states[m, h+1]
            F = mdp.F[s,:]
            for f in range(nF):
                dQold = mdp.dQ[f][s][a]
                dQnew = dQold + mdp.alpha*(F[f] + mdp.discount * np.average(mdp.dQ[f][sprime][:]) - dQold)
                mdp.dQ[f][s][a] = dQnew
                if abs(dQold - dQnew) > maxchange:
                    maxchange = abs(dQold - dQnew)
    return maxchange

def updateUnseenQvalues(mdp, unseen_sa):
    for _ ,(s,a) in enumerate(unseen_sa):
        r = mdp.reward[s,a]
        mdp.QL[s,a] += mdp.alpha * ( r - mdp.QL[s,a] )
    return
    
def updateUnseenQgradients(mdp, unseen_sa):
    nF = mdp.nFeatures
    for _ ,(s,a) in enumerate(unseen_sa):
        F = mdp.F[s,:]
        for f in range(nF):
            mdp.dQ[f][s][a] += mdp.alpha * ( F[f] - mdp.dQ[f][s][a] )
    return

def piMDPToolbox(mdp):

    # print("\nRewards: \n",mdp.reward)
    MAX_ITERS = 10000
    EPS = 1e-12
    SHOW_MSG = False
    nS = mdp.nStates
    nA = mdp.nActions
    pi = mdptoolbox.mdp.PolicyIterationModified(np.transpose(
        mdp.transition), mdp.reward, mdp.discount, max_iter=MAX_ITERS, epsilon=EPS)
    pi.run()
    Q = utils.QfromV(pi.V, mdp)
    # print("\nQ table: \n",Q)
    # print("\nPolicy: ",pi.policy)
    # print("\nValue: \n",pi.V)
    piL = np.reshape(pi.policy, (nS, 1))
    H = evalToolbox(piL, mdp)
    # print("\nH value: \n",H)

    return piL, pi.V, Q, H


def evalToolbox(piL, mdp):

    w = mdp.weight
    nS = mdp.nStates
    nA = mdp.nActions
    I = np.eye(nS)
    Tpi = np.zeros((nS, nS))

    for a in range(nA):
        act_ind = utils.find(piL, lambda x: x == a)
        if act_ind is not None:
            Tpi[act_ind, :] = np.squeeze(
                np.transpose(mdp.transition[:, act_ind, a]))
    Epi = np.zeros((nS, nS * nA)).astype(int)
    act_ind = np.reshape(piL * nS + np.arange(0, nS).reshape((nS, 1)), nS)
    for i in range(nS):
        Epi[i, act_ind[i]] = 1

    H = np.linalg.lstsq(I - mdp.discount * Tpi, np.matmul(Epi, mdp.F))[0]

    return H


def policyIteration(mdp):

    ''' This is the naive way to do policy
    iteration. Since we have a toolbox available,
    this function is currently unused '''

    oldpi = np.zeros((nS, 1)).astype(int)
    oldV = np.zeros((nS, 1)).astype(int)

    for iter in range(MAX_ITERS):
        [V, H] = evaluate(oldpi, mdp)
        Q = utils.QfromV(V, mdp)
        # Sec 2.2 Theorem 2 Eq 3 Algo for IRL
        piL = np.reshape(np.argmax(Q, axis=1), (nS, 1))
        V = np.zeros((nS, 1))
        for i in range(nS):
            V[i, :] = Q[i, piL[i, :]]
        done = utils.approxeq(V, oldV, EPS) or np.array_equal(oldpi, piL)

        if done:
            break
        oldpi = piL
        oldV = V

    return piL, V, Q, H


def evaluate(piL, mdp):

    ''' This function is being called 
    from policy iteration function.
    Hence it's currently unused.'''
    w = mdp.weight
    nS = mdp.nStates
    nA = mdp.nActions
    I = np.eye(nS)
    Tpi = np.zeros((nS, nS))

    for a in range(nA):
        act_ind = utils.find(piL, lambda x: x == a)
        if act_ind is not None:
            Tpi[act_ind, :] = np.squeeze(
                np.transpose(mdp.transition[:, act_ind, a]))
    Epi = np.zeros((nS, nS * nA)).astype(int)
    act_ind = np.reshape(piL * nS + np.arange(0, nS).reshape((nS, 1)), nS)
    for i in range(nS):
        Epi[i, act_ind[i]] = 1

    H = np.linalg.lstsq(I - mdp.discount * Tpi, np.matmul(Epi, mdp.F))[0]
    V = np.matmul(H, w)
    return V, H