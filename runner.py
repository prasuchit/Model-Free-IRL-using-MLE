import birl
import parameters as params
import generator
import options
import utils
import utils2
import llh
import numpy as np
import copy
import solver
import time
from scipy.optimize._minimize import minimize
from scipy.special._logsumexp import logsumexp
from tqdm.gui import tqdm
np.seterr(divide='ignore', invalid='ignore')


def main():

    solverMethod = 'manual'
    # solverMethod = 'scipy'
    algoName = 'MAP_BIRL'
    llhName = 'BIRL'
    # priorName = 'Gaussian'
    priorName = 'Uniform'
    # probName = 'highway'
    probName = 'gridworld'
    # optimMethod = 'gradDesc'
    optimMethod = 'nesterovGrad'
    nTrajs = 1
    nSteps = 5
    problemSeed = 1
    init_gridSize = 2
    init_blockSize = 1
    init_nLanes = 2     # Highway problem
    init_nSpeeds = 2    # Highway problem
    init_noise = 0.3

    normMethod = 'softmax'  # '0-1' 'None'

    algo = options.algorithm(algoName, llhName, priorName)

    irlOpts = params.setIRLParams(algo, restart=1, solverMethod=solverMethod, optimMethod = optimMethod, normMethod = normMethod, disp=True)

    problem = params.setProblemParams(probName, nTrajs=nTrajs, nSteps=nSteps, gridSize=init_gridSize, 
                blockSize=init_blockSize, nLanes=init_nLanes, nSpeeds=init_nSpeeds, noise=init_noise, seed=problemSeed)  
    
    mdp = generator.generateMDP(problem)
    
    data = generator.generateDemonstration(mdp, problem)

    opts = irlOpts

    trajs = data.trajSet

    expertPolicy = data.policy
    
    if(opts.solverMethod == 'scipy'):

        if opts.alg == 'MAP_BIRL':
            print("Calling MAP BIRL")
            birl.MAP(data, mdp, opts)
        else:
            print('Incorrect algorithm chosen: ', opts.alg)

    elif(opts.solverMethod == 'manual'):
        repeat = True
        t0 = time.time()
        nS = mdp.nStates
        nA = mdp.nActions
        nF = mdp.nFeatures
        mdp.QL = np.zeros((nS,nA))
        mdp.dQ = np.zeros((nF,nS,nA))
        mdp.alpha = 0.1
        while(repeat):
            print("Sampling a new weight...")
            w0 = utils.sampleNewWeight(mdp.nFeatures, opts, problemSeed)
            
            cache = []

            trajInfo = utils.getTrajInfo(trajs, mdp)
            
            # print("Compute initial LE and gradient ...")
            # initPost, initGrad= llh.calcNegLogPost(w0, trajInfo, mdp, opts)
            # print("Compute initial opimality region ...")
            # pi, H = utils2.computeOptmRegn(mdp, w0)
            # print("Cache the results ...")
            # cache.append([pi, H, initGrad])
            initGrad = np.zeros(np.shape(w0))
            currWeight = np.copy(w0)
            currGrad = np.copy(initGrad)
            
            if optimMethod == 'gradDesc':
                wL = utils2.gradientDescent(mdp, trajs, opts, currWeight, currGrad, cache)
            elif optimMethod == 'nesterovGrad':
                wL = utils2.nesterovAccelGrad(mdp, trajs, opts, currWeight, currGrad, cache = cache)
            
            wL = utils2.normalizedW(wL, normMethod)

            rewardDiff, valueDiff, policyDiff, piL, piE = utils2.computeResults(data, mdp, wL)

            if policyDiff < 0.3:    # Learned behavior accuracy should be atleast < 30%
                repeat = False
                print("Learned weights: \n", wL)
                t1 = time.time()
                runtime = t1 - t0
                print("Same number of actions between expert and learned pi: ",(piL.squeeze()==piE.squeeze()).sum(),"/",mdp.nStates)
                print("Time taken: ", runtime," seconds")
                print(f"Policy Diff: {policyDiff} | Reward Diff: {rewardDiff}| Value Diff: {valueDiff.squeeze()}")

    else:
        print("Please check your input!")

if __name__ == "__main__":
    main()