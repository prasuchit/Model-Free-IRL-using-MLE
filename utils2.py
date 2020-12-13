import utils
import llh
import solver
import numpy as np
import copy
import time
from scipy.optimize._minimize import minimize

def gradientDescent(mdp, trajs, opts, currWeight = 0, currGrad = 0, cache = []):
    
    trajInfo = utils.getTrajInfo(trajs, mdp)
    print("======== MLE Inference ========")
    print("======= Gradient Ascent =======")
    oldMLE = 0
    error = np.inf
    eps = 10**-2
    threshold = eps * (1 - mdp.discount)/mdp.discount
    i = 0
    while(error > threshold):    # Finding this: R_new = R + δ_t * ∇_R P(R|X)
        weightUpdate = (opts.stepsize * opts.alpha * currGrad)
        # opts.alpha *= opts.decay
        weightUpdate = np.reshape(weightUpdate,(mdp.nFeatures,1))
        currWeight = currWeight + weightUpdate
        # opti = reuseCacheGrad(currWeight, cache)
        # if opti is None:
            # print("  No existing cached gradient reusable ")
            # pi, H = computeOptmRegn(mdp, currWeight)
        MLE, currGrad = llh.calcNegLogPost(currWeight, trajInfo, mdp, opts)
        i += 1
        error = abs(oldMLE - MLE)
        print(f"- iter {i}: error: {error}, threshold: {threshold}")
        oldMLE = MLE
            # cache.append([pi, H, currGrad])
        # else:
        #     print("  Found reusable gradient ")
        #     currGrad = opti[2]
    return currWeight

def nesterovAccelGrad(mdp, trajs, opts, currWeight = 0, currGrad = 0, cache = []):
    trajInfo = utils.getTrajInfo(trajs, mdp)
    print("======== MLE Inference ========")
    print("==== Nesterov Accel Gradient ====")
    oldMLE = 0
    error = np.inf
    eps = 10**-4
    threshold = eps * (1 - mdp.discount)/mdp.discount
    i = 0
    prevGrad = np.copy(currGrad)
    while(error > threshold):    # Finding this: R_new = R + δ_t * ∇_R P(R|X)
        # Step 1 - Partial update
        # weightUpdate = (opts.decay * prevGrad)
        weightUpdate = (prevGrad)
        weightUpdate = np.reshape(weightUpdate,(mdp.nFeatures,1))
        currWeight = currWeight + opts.stepsize/2 * weightUpdate
        # opti = reuseCacheGrad(currWeight, cache)
        # if opti is None:
        #     print("  No existing cached gradient reusable ")
        #     pi, H = computeOptmRegn(mdp, currWeight)
        #     trajInfo = utils.getTrajInfo(trajs, mdp)
        MLE, currGrad = llh.calcNegLogPost(currWeight, trajInfo, mdp, opts)
        # Step 2 - Full update
        weightUpdate = (prevGrad) + (opts.alpha * currGrad)
        weightUpdate = np.reshape(weightUpdate,(mdp.nFeatures,1))
        currWeight = currWeight + opts.stepsize/2 * weightUpdate
        prevGrad = currGrad
        i += 1
        error = abs(oldMLE - MLE)
        print(f"- iter {i}: error: {error}, threshold: {threshold}")
        oldMLE = MLE
            # cache.append([pi, H, currGrad])
        # else:
        #     print("  Found reusable gradient ")
        #     currGrad = opti[2]
    return currWeight
   
def computeOptmRegn(mdp, w):
    mdp = utils.convertW2R(w, mdp)
    piL, _, _, H = solver.piMDPToolbox(mdp)
    return piL, H

def reuseCacheGrad(w, cache):
    for opti in cache:
        H = opti[1]
        constraint = np.matmul(H, w)
        compare = np.where(constraint < 0)
        if compare[0].size > 0:
            return opti
    return None

def piInterpretation(policy, name):
    actions = {}
    if name == 'gridworld':
        for i in range(len(policy)):
            if(policy[i] == 0):
                actions[i] = 'North'
            elif(policy[i] == 1):
                actions[i] = 'East'
            elif(policy[i] == 2):
                actions[i] = 'West'
            elif(policy[i] == 3):
                actions[i] = 'South'
    else:
        print("Problem is not gridworld. This function doesn't work for other problems yet.")
    return actions

def computeResults(data, mdp, wL):

    mdp = utils.convertW2R(data.weight, mdp)
    piE, VE, QE, HE = solver.piMDPToolbox(mdp)
    vE = np.matmul(np.matmul(data.weight.T,HE.T),mdp.start)

    mdp = utils.convertW2R(wL, mdp)
    piL, VL, QL, HL = solver.piMDPToolbox(mdp)
    vL = np.matmul(np.matmul(wL.T,HL.T),mdp.start)

    d  = np.zeros((mdp.nStates, 1))
    for s in range(mdp.nStates):
        ixE = QE[s, :] == max(QE[s, :])
        ixL = QL[s, :] == max(QL[s, :])
        if ((ixE == ixL).all()):
            d[s] = 0
        else:
            d[s] = 1

    rewardDiff = np.linalg.norm(data.weight - wL)
    valueDiff  = abs(vE - vL)
    policyDiff = np.sum(d)/mdp.nStates

    return rewardDiff, valueDiff, policyDiff, piL, piE

def normalizedW(weights, normMethod):
    if normMethod == 'softmax':
        wL = (np.exp(weights))/(np.sum(np.exp(weights))) # Softmax normalization
    elif normMethod == '0-1':
        wL = (weights-min(weights))/(max(weights)-min(weights)) # Normalizing b/w 0-1
    else:   wL = weights # Unnormalized raw weights

    return wL