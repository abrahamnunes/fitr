#
#   TASKS MODULE includes classes for each task
#
import numpy as np

class orthogng:
    """
    The orthogonalized go-nogo task.

    References
    ----------
    Guitart-Masip, M. et al. (2014) Psychopharmacology (Berl). 231, 955–966

    """
    def __init__(self):
        self.pstates   = np.zeros(4) + 0.25
        self.outcomes  = np.array([1, -1])
        self.p_outcome = np.array([[0.8, 0.2]])

    def simulate(self, nsubjects, ntrials):
        """
        Simulates a cohort of subjects

        Parameters
        ----------
        nsubjects : int
            Number of subjects to simulate.
        ntrials : int
            Number of trials

        Returns
        ----------
        results : dict
            Data for all subjects
        """
        from utils import softmax, mnrandi

        results = {}

        for i in range(nsubjects):
            Q = np.array([4, 3])
            for t in range(ntrials):
                s = mnrandi(self.pstates)
