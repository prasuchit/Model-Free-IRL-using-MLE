import numpy as np
import math
import options
import utils2
import numpy.matlib
import time
from operator import mod
np.seterr(divide='ignore', invalid='ignore')

def approxeq(V, oldV, EPS):
    ''' Used in policy iteration '''
    return np.linalg.norm(np.reshape(V, len(V)) - np.reshape(oldV, len(oldV))) < EPS

def sampleWeight(problem, nF, seed=None):
    
    np.random.seed(seed)
    w = np.zeros((nF, 1))
    i = 2   # behavior setter
    if problem.name == 'gridworld':
        if i == 0:  # Random behaviour
            w = np.random.rand(nF, 1)
        else:   # Reaching last state is most preferred
            w[:] = -0.1
            w[-1] = 1
    elif problem.name == 'highway':
        # weights are assigned 1 for collision, n for nlanes, n for nspeeds
        if i == 1:              # fast driver avoids collisions and prefers high speed
            w[:] = -0.01
            w[0] = -0.1        # collision
            w[-1] = 1.0         # high-speed
            
        elif i == 2:            # safe driver avoids collisions and prefers right-most lane
            w[:] = -0.01
            w[0] = -0.1           # collision
            w[problem.nLanes] = 0.5 # right-most lane
        elif i == 3:            # erratic driver prefers collisions and high-speed
            w[:] = -0.01
            w[0] = 1            # collision
            w[-1] = 0.1         # high-speed
        else:
            w = np.random.rand(nF, 1)
            # w = np.array([-1.59840727, 0.3994327, 0.71807136, 0.15155268, -0.01017073])
    else:
        print("Unknown problem name!!")
        exit(0)

    return w
    
def convertW2R(weight, mdp):    
    mdp.weight = weight 
    reward = np.matmul(mdp.F, weight)
    reward = np.reshape(reward, (mdp.nStates, mdp.nActions), order='F')
    mdp.reward = reward
    return mdp

def sid2info(sid, nS, nL, nG):
    y = [None] * nL
    for i in range(nL-1,-1,-1):
        y[i] = int(mod(sid, nG))
        sid = (sid - y[i])/nG
    myx = int(mod(sid, nL))
    sid = int((sid - myx)/nL)
    spd = int(mod(sid, nS))
    return spd, myx, y

def info2sid(spd, myx, y, nS, nL, nG):
    sid = spd
    sid = (sid)*nL + myx
    for i in range(nL):
        sid = (sid)*nG + y[i]

    return sid

def QfromV(V, mdp): 
    nS = mdp.nStates
    nA = mdp.nActions
    Q = np.zeros((nS, nA))
    for a in range(nA):
        expected = np.matmul(np.transpose(mdp.transition[:, :, a]), V)
        Q[:, a] = mdp.reward[:, a] + mdp.discount * np.squeeze(expected)
    return Q

def find(arr, func):
    l = [i for (i, val) in enumerate(arr) if func(val)]
    if not l:
        return None
    else:
        return np.array(l).astype(int)

def getOrigTrajInfo(trajs, mdp):
    nS = mdp.nStates
    nA = mdp.nActions

    trajInfo = options.trajInfo()
    trajInfo.nTrajs = trajs.shape[0]
    trajInfo.nSteps = trajs.shape[1]
    occlusions, cnt, allOccNxtSts = utils2.processOccl(trajs, nS, nA, trajInfo.nTrajs, trajInfo.nSteps, mdp.transition)
   
    trajInfo.occlusions = np.array(occlusions)
    trajInfo.allOccNxtSts = np.array(allOccNxtSts)
    N = np.count_nonzero(cnt)
    trajInfo.cnt = np.zeros((N, 3)).astype(int)
    i = 0
    for s in range(nS):
        for a in range(nA):
            if cnt[s, a] > 0:
                trajInfo.cnt[i, 0] = s
                trajInfo.cnt[i, 1] = a
                trajInfo.cnt[i, 2] = cnt[s, a]
                i += 1
    return trajInfo

def getTrajInfo(trajs, mdp):
    nS = mdp.nStates
    nA = mdp.nActions

    trajInfo = options.trajInfo()
    trajInfo.nTrajs = trajs.shape[0]
    trajInfo.nSteps = trajs.shape[1]  
    cnt = np.zeros((nS, nA))
    trajInfo.states = np.zeros((trajInfo.nTrajs, trajInfo.nSteps), dtype=int)
    trajInfo.actions = np.zeros((trajInfo.nTrajs, trajInfo.nSteps), dtype=int)

    for m in range(trajInfo.nTrajs):
        for h in range(trajInfo.nSteps):
            s = trajInfo.states[m,h] = trajs[m, h, 0]
            a = trajInfo.actions[m,h] = trajs[m, h, 1]
            if -1 not in trajs[m, h, :]:
                cnt[s, a] += 1

    N = np.count_nonzero(cnt)
    trajInfo.cnt = np.zeros((N, 3)).astype(int)
    i = 0
    for s in range(nS):
        for a in range(nA):
            if cnt[s, a] > 0:
                trajInfo.cnt[i, 0] = s
                trajInfo.cnt[i, 1] = a
                trajInfo.cnt[i, 2] = cnt[s, a]
                i += 1
    return trajInfo

def sampleNewWeight(dims, options, seed=None):
    # np.random.seed(seed)
    np.random.seed(None)
    lb = options.lb 
    ub = options.ub    
    if options.priorType == 'Gaussian':
        # w0 = options.mu + np.random.randn(dims, 1)*options.sigma  ''' Naive way to do it '''
        # for i in range(len(w0)):
        #     w0[i] = max(lb, min(ub, w0[i])) # Check to ensure weights are within bounds

        mean = np.ones(dims) * options.mu
        cov = np.eye(dims) * options.sigma
        w0 = np.clip(np.random.multivariate_normal(mean, cov), a_min=lb, a_max=ub).reshape((dims, 1))
    else:
        w0 = np.random.uniform(low=lb, high=ub, size=(dims,1))
    return w0