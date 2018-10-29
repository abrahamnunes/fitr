import numpy as np
import scipy.stats as ss

class binomial_exact(object):
    def __init__(self, nevents, ntotal, pnull=0.5, alpha=0.05, alternative='greater', comb_exact=False):
        """ Runs binomial exact test

        Arguments:

            nevents: `int`. Number of events in the sample
            ntotal: `int`. Number of observations in a sample
            pnull: `float`. Null event probability
            alpha: `0 <= float <= 1`. Significance threshold
            alternative: `{'greater', 'less'}`. Whether to compute probability that number of events is at least as much as what we saw (`greater`), or no more than what we saw (`less`)
            comb_exact: `bool`. Whether to compute exact binomial coefficient

        """
        self.nevents = nevents
        self.ntotal  = ntotal
        self.phat = self.nevents/self.ntotal
        self.pnull = pnull

        # Compute exact probability
        p = self.pnull # Just makes lines shorter
        X = np.arange(nevents, ntotal+1)
        binomcoef = np.array([comb(ntotal, x, exact=comb_exact) for x in X])
        ptrue = np.array([p**x for x in X])
        pfalse = np.array([(1.-p)**(ntotal-x) for x in X])
        prob = np.sum(binomcoef*ptrue*pfalse)
        if alternative == 'greater':
            self.p = prob
        elif alternative == 'less':
            self.p = 1.-prob

class binomial_twosample(object):
    def __init__(self,
                 nevents1,
                 ntotal1,
                 nevents2,
                 ntotal2,
                 alpha=0.05,
                 alternative='two-sided'):
        """ Runs comparison of two binomial random variables to determine whether they were drawn from the same binomial distribution.

        More specifically, given null hypothesis

        $$
        \mathcal H_o : p_1 = p_2,
        $$

        we have the alternative hypotheses

        $$
        H_a : p_1 \neq p_2
        $$

        or

        $$
        H_a : p_1 > p_2
        $$

        or

        $$
        H_a : p_1 < p_2.
        $$

        Here, the sample labeled ``2'' is by convention the ``control'' sample

        Hypothesis testing is done using the score test, with Agresti/Caffo (2000) confidence intervals.

        Arguments:

            nevents1: `int`. Number of events in the first sample
            ntotal1: `int`. Number of observations in first sample
            nevents2: `int`. Number of events in the control sample
            ntotal2: `int`. Number of observations in control sample
            alpha: `0<float<1`. Statistical significance threshold
            alternative: `{'two-sided', 'greater', 'less'}`. Whether to do a one or two-tailed test

        """
        self.nevents1 = nevents1
        self.nevents2 = nevents2
        self.ntotal1 = ntotal1
        self.ntotal2 = ntotal2
        self.n = np.array([[nevents1, ntotal1-nevents1],
                           [nevents2, ntotal2-nevents2]])

        # Compute Score
        self.phat1 = self.nevents1/self.ntotal1
        self.phat2 = self.nevents2/self.ntotal2
        self.dprob = self.phat1 - self.phat2
        self.phat = (self.nevents1 + self.nevents2)/(self.ntotal1 + self.ntotal2)
        self.score = (self.phat1 - self.phat2)/np.sqrt(self.phat*(1-self.phat)*((1/self.ntotal1) + (1/self.ntotal2)))
        self.pvalue = 1-ss.norm().cdf(np.abs(self.score))

        # Compute Agresti/Caffo interval
        n1_ = self.ntotal1 + 2
        n2_ = self.ntotal2 + 2
        p1_ = (self.nevents1 + 1)/n1_
        p2_ = (self.nevents2 + 1)/n2_

        if alternative == 'two-sided':
            Z = np.abs(ss.norm().ppf(alpha/2))
            dphat = p1_ - p2_
            interval = Z*np.sqrt(((p1_*(1-p1_))/n1_) + ((p2_*(1-p2_))/n2_))
            self.ci = np.array([dphat - interval, dphat + interval])
        elif alternative == 'greater':
            Z = ss.norm().ppf(alpha)
            dphat = p1_ - p2_
            interval = Z*np.sqrt(((p1_*(1-p1_))/n1_) + ((p2_*(1-p2_))/n2_))
            self.ci = np.array([dphat - interval, np.inf])
        elif alternative == 'less':
            Z = ss.norm().ppf(alpha)
            dphat = p2_ - p1_
            interval = Z*np.sqrt(((p1_*(1-p1_))/n1_) + ((p2_*(1-p2_))/n2_))
            self.ci = np.array([ np.inf, dphat - interval])
