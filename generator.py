import numpy as np
import options
import solver
import math
import utils
import gridworld
import highway3
import random
import time
np.seterr(divide='ignore', invalid='ignore')


def generateMDP(problem, discount=0.99):
    if problem.name == 'gridworld':
        mdp = gridworld.init(problem.gridSize, problem.blockSize, problem.noise, problem.discount)

    elif problem.name == 'highway':
        mdp = highway3.init(problem.gridSize, problem.nSpeeds, problem.nLanes, problem.discount, 1)

    nS = mdp.nStates
    nA = mdp.nActions
    I = np.tile(np.eye(nS), (nA, 1))
    mdp.T = mdp.transition
    mdp.T = np.reshape(mdp.T, (nS, nS * nA), order='F')
    mdp.T = mdp.discount * np.transpose(mdp.T)
    mdp.E = I - mdp.T
    return mdp

def generateDemonstration(mdp, problem):
    data = options.demonstrations()
    nF = mdp.nFeatures
    if mdp.weight is None:
        w = utils.sampleWeight(problem, nF, problem.seed)
    else:
        w = mdp.weight
    mdp = utils.convertW2R(w, mdp)
    trajs, policy = generateTrajectory(mdp, problem)
    data.seed = problem.seed
    data.weight = w
    data.policy = policy
    data.trajId = np.ones(problem.nTrajs).astype(int)
    data.nTrajs = problem.nTrajs
    data.nSteps = problem.nSteps
    data.trajSet = trajs

    return data

def generateTrajectory(mdp, problem):
    print('Generate Demonstrations')
    nF = mdp.nFeatures
    if mdp.weight is None or mdp.reward is None:
        w = utils.sampleWeight(problem, nF, problem.seed)
        print('  - sample a new weight')
        mdp = utils.convertW2R(w, mdp)
        print('  - assign weight to the problem')
        print(w)
    if problem.name == 'gridworld':
        policy, value, _, _ = solver.piMDPToolbox(mdp)
        # policy, value, _, _ = solver.Qlearning(mdp)
        optValue = np.matmul(np.transpose(mdp.start), value)
        print(' - Optimal value : %.4f' % (optValue))
        trajs, trajVmean, trajVvar = sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed)
        print(' - sample %d trajs: V mean: %.4f, V variance: (%.4f)' % (problem.nTrajs, trajVmean, trajVvar))

    elif problem.name == 'highway':
        print(f'solve {mdp.name}\n')
        tic = time.time()  
        policy, value, _ , H = solver.piMDPToolbox(mdp)
        toc = time.time()
        elapsedTime = toc - tic
        print('sample trajectory\n')
        trajs, trajVmean, trajVvar = sampleTrajectories(problem.nTrajs, problem.nSteps, policy, mdp, problem.seed)
        
        nFeatures = np.zeros((nF, 1))
        for t in range(problem.nSteps):
            s = trajs[0, t, 0]
            a = trajs[0, t, 1]
            f = mdp.F[(a)*mdp.nStates + s, :]
            nFeatures += np.reshape(f, (nF,1))

        print('\n# of collisions: ', nFeatures[0])
        print('\n# of lanes     : ')
        for i in range(problem.nLanes):
            print(nFeatures[1 + i])
        print('\n# of speeds    : ')
        for i in range(problem.nSpeeds):
            print(nFeatures[1 + problem.nLanes + i])
        print('\n')
        
        print('\nweight         : ')
        for i in range(mdp.nFeatures):
            print(mdp.weight[i])
        print(' - sample %d trajs: V mean: %.4f, V variance: (%.4f)' % (problem.nTrajs, trajVmean, trajVvar))
        optValue = np.matmul(np.transpose(mdp.start), value)
        print(' - Optimal value : %.4f' % (optValue))
        print('Time taken: ', elapsedTime, ' sec\n\n')
        
    return trajs, policy

def sampleTrajectories(nTrajs, nSteps, piL, mdp, seed=None):

    trajs = np.zeros((nTrajs, nSteps, 2)).astype(int)
    vList = np.zeros(nTrajs)

    np.random.seed(seed)

    for m in range(nTrajs):
        sample = np.random.multinomial(n=1, pvals=np.reshape(mdp.start, (mdp.nStates)))
        s = np.squeeze(np.where(sample == 1))
        v = 0
        for h in range(nSteps):
            a = np.squeeze(piL[s])
            r = mdp.reward[s, a]
            v = v + r * math.pow(mdp.discount, h)
            trajs[m, h, :] = [s, a]
            sample = np.random.multinomial(n=1, pvals=np.reshape(mdp.transition[:, s, a], (mdp.nStates)))
            s = np.squeeze(np.where(sample == 1))
        vList[m] = v
    Vmean = np.mean(vList)
    Vvar = np.var(vList)

    return trajs, Vmean, Vvar 