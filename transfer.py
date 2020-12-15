import utils
import numpy as np
import math
import parameters as params
import generator
import solver
import time

def fit_policy_and_values(V, pi, nSsmall, nLsmall, nGsmall, nSlarge, nLlarge, nGlarge):
    # print("small speed: {}, small lanes: {}, small grid: {}, large speed: {}, large lane: {}, large grid: {}".format(nSsmall, nLsmall, nGsmall, nSlarge, nLlarge, nGlarge))
    nS = int(nSlarge*nLlarge*math.pow(nGlarge,nLlarge))
    bigV = np.zeros(nS)
    bigPi = np.zeros(nS, dtype='int')
    for smallIdx in range(len(V)):

        spd, myx, y = utils.sid2info(smallIdx, nSsmall, nLsmall, nGsmall)

        largeIdx = utils.info2sid(spd, myx, y, nSlarge, nLlarge, nGlarge)
        bigV[largeIdx] = V[smallIdx]
        bigPi[largeIdx] = pi[smallIdx]
        

    return bigV, bigPi

def fit_policy_and_values_lanes(V, pi, nSsmall, nLsmall, nGsmall, nSlarge, nLlarge, nGlarge):
    ys = []
    for i in range(nGsmall):
        ys.append(i)
    nS = int(nSlarge*nLlarge*math.pow(nGlarge,nLlarge))
    bigV = np.zeros(nS)
    bigPi = np.zeros(nS, dtype='int')
    for i in range(len(V)):

        spd, myx, y = utils.sid2info(i, nSsmall, nLsmall, nGsmall)

        for j in range(len(ys)):
            big_y = y.copy()
            big_y.insert(0, ys[j])
            bigIdx = utils.info2sid(spd, myx, big_y, nSlarge, nLlarge, nGlarge)
            bigV[bigIdx] = V[i]
            bigPi[bigIdx] = pi[i]

    return bigV, bigPi

# Run policy iteration for the given transition and reward function, with the discount g.
# The policy and values can be passed in for transfer learning purposes, but otherwise they're initialized to the
# default actions and values of 0, respectively.
def policy_iteration(t, r, g = .9, pi = np.full(1, np.inf), vals = None):
    iters = 0 # Number of loops of policy evaluation and policy improvement.
    state_space = len(t)
    action_space = len(t[0][0])
    # Need to initialize pi and vals if not given.
    # Also if one is not given, the other is assumed likewise.
    if np.sum(pi) == np.inf:
        pi = np.zeros(state_space, dtype='int')
        vals = np.zeros(state_space)
    else:
        pass

    # The expected value of a state-action pair needs to be calculated before anything else.
    # If the passed in reward function is already in the form of R(s, a), this step is essentially skipped.
    q = 0
    if (r.ndim == 3):
        q = np.zeros((state_space, action_space))
        for s in range(state_space):
            for a in range(action_space):
                for sp in range(state_space):
                    q[s][a] += r[s][sp][a] * t[s][sp][a]
    else:
        q = r

    # print(q)

    converged = False # True if pi = pi'.
    pi_p = np.zeros(state_space)
    vals_p = np.zeros(state_space)

    while not converged:
        iters += 1
        # Policy evaluation.
        for s in range(state_space):
            next_s_sum = 0
            for sp in range(state_space):
                next_s_sum += t[s][sp][pi[s]] * vals[sp]
            # print(next_s_sum)
            vals_p[s] = q[s][pi[s]] + g * next_s_sum

        vals = np.copy(vals_p)
        # if state_space > 25:
        #     print("iter:", iters)
        #     show_grid_values(vals, 10)
        #     print()
        
        # Policy improvement.
        evals = np.zeros((state_space, action_space))
        for a in range(action_space):
            for s in range(state_space):
                next_s_sum = 0
                for sp in range(state_space):
                    next_s_sum += t[s][sp][a] * vals[sp]
                evals[s][a] = q[s][a] + g * next_s_sum

        # print(evals)
        
        pi_p = np.argmax(evals, axis=1)

        # Check for convergence.
        if np.array_equal(pi, pi_p):
            converged = True
        pi = np.copy(pi_p)

        print("Iteration {}".format(iters))

    return pi, vals, iters

def main():
    # Learning the 2 lane problem.
    probName = 'highway'
    problemSeed = 1
    init_gridSize = 8
    init_nLanes = 2     # Highway problem
    init_nSpeeds = 2    # Highway problem
    init_noise = 0.3

    problem = params.setProblemParams(probName, gridSize=init_gridSize, 
                nLanes=init_nLanes, nSpeeds=init_nSpeeds, noise=init_noise, seed=problemSeed)

    mdp = generator.generateMDP(problem)

    if mdp.weight == None:
        mdp.weight = utils.sampleWeight(problem, mdp.nFeatures, problem.seed)

    t = mdp.transition
    mdp = utils.convertW2R(mdp.weight, mdp)
    r = mdp.reward
    gamma = .9

    pis, Vs, iters = policy_iteration(t, r, gamma)

    # np.savetxt('small_policy.csv', pis, delimiter = ',')
    # np.savetxt('small_values.csv', Vs, delimiter = ',')

    # mdp = generator.generateMDP(problem)

    # if mdp.weight == None:
    #     mdp.weight = utils.sampleWeight(problem, mdp.nFeatures, problem.seed)

    # t = mdp.transition
    # mdp = utils.convertW2R(mdp.weight, mdp)
    # r = mdp.reward
    # gamma = .9

    # bigV, bigPi = fit_policy_and_values_lanes(Vs, pis, 2, init_nLanes, init_gridSize, init_nSpeeds, init_nLanes, init_gridSize)

    # t0 = time.time()

    # _, _, iters = policy_iteration(t, r, gamma, pi = bigPi, vals = bigV)

    # np.savetxt('small_policy.csv', pis, delimiter = ',')
    # np.savetxt('small_values.csv', Vs, delimiter = ',')

    # elapsed = time.time() - t0

    # print("Transferred: {}, {} seconds".format(iters, elapsed))

    smallLanes = init_nLanes


    # # Transfering to the 3 lane problem.
    init_nLanes = 3     # Highway problem

    problem = params.setProblemParams(probName, gridSize=init_gridSize, 
                nLanes=init_nLanes, nSpeeds=init_nSpeeds, noise=init_noise, seed=problemSeed)

    mdp = generator.generateMDP(problem)

    if mdp.weight == None:
        mdp.weight = utils.sampleWeight(problem, mdp.nFeatures, problem.seed)

    t = mdp.transition
    mdp = utils.convertW2R(mdp.weight, mdp)
    r = mdp.reward
    gamma = .9

    bigV, bigPi = fit_policy_and_values_lanes(Vs, pis, init_nSpeeds, smallLanes, init_gridSize, init_nSpeeds, init_nLanes, init_gridSize)

    # # # np.savetxt('policy.csv', bigPi, delimiter = ',')

    t0 = time.time()

    _, _, iters = policy_iteration(t, r, gamma, pi = bigPi, vals = bigV)

    elapsed = time.time() - t0

    print("Large: {}, {} seconds".format(iters, elapsed))

    # np.savetxt('control_policy.csv', piL, delimiter = ',')

    # print(piL)

    # pi_modified = np.genfromtxt('modified_trans_noise.csv', delimiter=",")
    # pi_true = np.genfromtxt('Orig_trans_noise.csv', delimiter=",")

    # v_modified = np.genfromtxt('modified_trans_noise_value.csv', delimiter=",")
    # v_true = np.genfromtxt('orig_trans_noise_value.csv', delimiter=",")

    # bigVmod, bigPimod = fit_policy_and_values(v_modified, pi_modified, init_nSpeeds, smallLanes, init_gridSize, init_nSpeeds, init_nLanes, init_gridSize)
    # bigVtrue, bigPitrue = fit_policy_and_values(v_true, pi_true, init_nSpeeds, smallLanes, init_gridSize, init_nSpeeds, init_nLanes, init_gridSize)

    # # np.savetxt('big modified.csv', bigPimod)
    # # np.savetxt('big true.csv', bigPitrue)

    # problem = params.setProblemParams(probName, gridSize=init_gridSize, 
    #             nLanes=init_nLanes, nSpeeds=init_nSpeeds, noise=init_noise, seed=problemSeed)

    # mdp = generator.generateMDP(problem)

    # if mdp.weight == None:
    #     mdp.weight = utils.sampleWeight(problem, mdp.nFeatures, problem.seed)

    # t = mdp.transition
    # mdp = utils.convertW2R(mdp.weight, mdp)
    # r = mdp.reward
    # gamma = .9

    # t0 = time.time()

    # _, _, iters = policy_iteration(t, r, gamma, bigPimod, bigVmod)

    # end = time.time() - t0

    # print("The modified problem took {} iterations and {} seconds".format(iters, end))

    # problem = params.setProblemParams(probName, gridSize=init_gridSize, 
    #             nLanes=init_nLanes, nSpeeds=init_nSpeeds, noise=init_noise, seed=problemSeed)

    # mdp = generator.generateMDP(problem)

    # if mdp.weight == None:
    #     mdp.weight = utils.sampleWeight(problem, mdp.nFeatures, problem.seed)

    # t = mdp.transition
    # mdp = utils.convertW2R(mdp.weight, mdp)
    # r = mdp.reward
    # gamma = .9

    # t0 = time.time()

    # _, _, iters = policy_iteration(t, r, gamma, bigPitrue, bigVtrue)

    # end = time.time() - t0

    # print("The modified problem took {} iterations and {} seconds".format(iters, end))

if __name__ == "__main__":
    main()

    