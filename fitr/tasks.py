"""
Classes representing each task
"""
import numpy as np
import matplotlib.pyplot as plt

class SyntheticData(object):
    """
    Object representing synthetic data

    Methods
    -------
    cumreward_param_plot :
        Plots the cumulative reward against model parameters. Useful to determine the relationship between reward acquisition and model parameters for a given task.
    plot_cumreward :
        Plots the cumulative reward over time for each subject
    """
    def __init__(self):
        self.data = {}
        self.params = None

    def cumreward_param_plot(self, alpha=0.9):
        if self.params is not None:
            nsubjects = np.shape(self.params)[0]
            creward = np.zeros(nsubjects)
            for i in range(0, nsubjects):
                creward[i] = np.sum(self.data[i]['R'])

            nparams = np.shape(self.params)[1]
            fig, ax = plt.subplots(1, nparams, figsize=(15, 5))
            for i in range(0, nparams):
                ax[i].scatter(self.params[:,i], creward, c='k', alpha=alpha)

            plt.show()
        else:
            print('ERROR: There are no parameters assigned')
            return

    def plot_cumreward(self):
        """ Plots cumulative reward over time for each subject"""
        nsubjects = len(self.data)
        fig, ax = plt.subplots(1, 1)
        for i in range(nsubjects):
            nsteps = len(self.data[i]['R'])
            ax.plot(np.arange(nsteps), np.cumsum(self.data[i]['R']))

        ax.set_title('Cumulative Reward by Subject\n')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Reward')
        plt.show()



#===============================================================================
#
#   SIMPLE BANDIT TASK
#
#===============================================================================

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
        self.rprob_sd = rprob_sd

        if rprob == 'stochastic':
            self.rprob_bounds = rprob_bounds
        else:
            if len(rprob) != self.narms:
                print('Reward probability vector must have one entry per arm')
                return
            if np.any(rprob > 1) or np.any(rprob < 0):
                print('Reward probabilities must all lie between 0 and 1')
                return

            self.rprob=rprob

    def simulate(self, ntrials, params):

        nsubjects = np.shape(params)[0]

        # Initialize reward paths
        path_max = self.rprob_bounds[0]
        path_min = self.rprob_bounds[1]
        path_sd  = self.rprob_sd

        results = SyntheticData()
        results.params = params

        for i in range(0, nsubjects):
            paths = np.random.uniform(path_min, path_max, size=[ntrials+1, 2])

            # initialize subject-level value table
            Q  = np.zeros(2)
            lr = params[i,0]
            cr = params[i,1]

            results.data[i] = {'S': np.zeros(ntrials),
                               'A': np.zeros(ntrials),
                               'R': np.zeros(ntrials),
                               'RPE': np.zeros(ntrials)}

            for t in range(0, ntrials):
                a = action(cr*Q)
                r = reward(a, paths[t, :])

                # learn
                rpe = r - Q[a]
                Q[a] = Q[a] + lr*rpe

                # store values
                results.data[i]['S'][t] = 0
                results.data[i]['A'][t] = a
                results.data[i]['R'][t] = r
                results.data[i]['RPE'][t] = rpe

                # update reward probabilities
                rand_step = np.random.normal(0, path_sd, size=2)
                paths[t+1, :] = np.maximum(np.minimum(paths[t,:] + rand_step, path_max), path_min)

        return results

#===============================================================================
#
#   TWO-STEP TASK
#
#===============================================================================

class twostep(object):
    """
    Model of the two-step task (Daw et al. 2011).

    Attributes
    ----------
    ptrans : ndarray
        Probability of transitioning from state 0 to either state 1 or 2 depending on the choice made at the first step of the task.

    Methods
    -------
    simulate
        Generates synthetic data from the task.

    References
    ----------
    [1] Daw, N.D. et al. (2011) Model-based influences on humans’ choices and striatal prediction errors. Neuron 69, 1204–1215
    """
    def __init__(self, ptrans=0.7, rewards=[1, 0]):
        self.ptrans = np.array([1-ptrans, ptrans])

    def simulate(self, ntrials, params):

        nsubjects = np.shape(params)[0]

        results = SyntheticData()
        results.params = params

        #initialize reward paths
        path_max = 0.8
        path_min = 0.2
        path_sd  = 0.025


        for i in range(nsubjects):
            # Initialize paths within subjects
            paths = np.random.uniform(path_min, path_max, size=[ntrials+1, 4])

            lr = params[i,0]
            cr = params[i,1]
            w  = params[i,2]

            results.data[i] = {
                'S': np.zeros([ntrials, 2]),
                'A': np.zeros([ntrials, 2]),
                'R': np.zeros(ntrials)
            }

            Q   = np.zeros([3, 2])
            Qmb = np.zeros([3, 2])
            Qmf = np.zeros([3, 2])

            for t in range(ntrials):
                s1 = int(0)
                a1 = int(action(cr*Q[s1,:]))

                s2 = int(np.random.binomial(1, p=self.ptrans[a1]) + 1)
                a2 = int(action(cr*Q[s2,:]))

                rprob = paths[t,:]
                rprob = np.reshape(rprob, (2, 2))
                r = np.random.binomial(1, p=rprob[s2-1, a2])

                # Update model-free values
                Qmf[s2, a2] = Qmf[s2, a2] + lr*(r - Qmf[s2, a2])
                Qmf[s1, a1] = Qmf[s1, a1] + lr*(Qmf[s2, a2]-Qmf[s1, a1])

                # Update model based values
                Qmb[0, a1] = self.ptrans[1]*np.max(Qmf[1,:]) + self.ptrans[0]*np.max(Qmf[2,:])
                Qmb[0, a2] = self.ptrans[1]*np.max(Qmf[2,:]) + self.ptrans[0]*np.max(Qmf[1,:])

                # Linear combination of MF and MB
                Q = w*Qmb + (1-w)*Qmf

                # Store data
                results.data[i]['S'][t,0] = s1
                results.data[i]['S'][t,1] = s2
                results.data[i]['A'][t,0] = a1
                results.data[i]['A'][t,1] = a2
                results.data[i]['R'][t]   = r

                # Update reward probabilities
                rand_step = np.random.normal(0, path_sd, size=4)
                paths[t+1, :] = np.maximum(np.minimum(paths[t,:] + rand_step, path_max), path_min)

        return results



#===============================================================================
#
#   UTILITY FUNCTIONS
#
#===============================================================================

def action(x):
    p = np.exp(x)/np.sum(np.exp(x))
    return np.argmax(np.random.multinomial(1, pvals=p))

def reward(a, paths):
    return np.random.binomial(1, p=paths[a])
