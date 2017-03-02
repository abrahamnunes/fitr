#
#   TASKS MODULE includes classes for each task
#
import numpy as np

class bandit(object):
    """
    Simple one-step bandit task.

    Attributes
    ----------
    narms : int
        Number of arms
    rewards : ndarray(shape=(2))
        First entry is the reward, if gained, and the second entry is the magnitude of the loss
    rprob : {ndarray(shape=(narms)), 'stochastic'}
        Probabilty of reward for each arm of the task. One can either specify the probabilities for each arm or enter 'stochastic,' which will vary the reward probability by a gaussian random walk

    Methods
    -------
    simulate(nsubjects,ntrials)
        Runs the task on simulated subjects

    """
    def __init__(self, narms=2, rewards=[1, 0], rprob='stochastic', rprob_sd=0.025, rprob_bounds=[0.2, 0.8]):
        self.narms = narms
        self.rewards = rewards

        if rprob == 'stochastic':
            self.rprob_bounds = rprob_bounds
            self.rprob = np.random.uniform(rprob_bounds[0], rprob_bounds[1], size=self.narms)
        else:
            if len(rprob) != self.narms:
                print('Reward probability vector must have one entry per arm')
                return
            if np.any(rprob > 1) or np.any(rprob < 0):
                print('Reward probabilities must all lie between 0 and 1')
                return

            self.rprob=rprob

    def simulate(self, nsubjects, ntrials, params):
        """
        Runs the task

        """

        #initialize reward paths
        path_max = 0.8
        path_min = 0.2
        path_sd  = 0.025
        paths = np.random.uniform(path_min, path_max, size=[ntrials+1, 2])


        results = {}

        for i in range(0, nsubjects):
            # initialize subject-level value table
            Q    = np.zeros(2)
            lr   = params[i,0]
            beta = params[i,1]

            results[i] = {'S': np.zeros(ntrials),
                          'A': np.zeros(ntrials),
                          'R': np.zeros(ntrials)}
            for t in range(0, ntrials):
                a = action(beta*Q)
                r = reward(a, paths[t, :])

                # learn
                Q[a] = Q[a] + lr*(r - Q[a])

                # store values
                results[i]['S'][t] = 0
                results[i]['A'][t] = a
                results[i]['R'][t] = r

                # update reward probabilities
                rand_step = np.random.normal(0, path_sd, size=2)
                paths[t+1, :] = np.maximum(np.minimum(paths[t,:] + rand_step, path_max), path_min)

        return results

def action(x):
    p = np.exp(x)/np.sum(np.exp(x))
    return np.argmax(np.random.multinomial(1, pvals=p))

def reward(a, paths):
    return np.random.binomial(1, p=paths[a])
