# -*- coding: utf-8 -*-
from fitr.stats.model_evaluation import bic
from fitr.stats.model_evaluation import lme

from fitr.stats.correlations import pearson_rho
from fitr.stats.correlations import spearman_rho
from fitr.stats.proportions import binomial_exact
from fitr.stats.proportions import binomial_twosample
from fitr.stats.nonparametric import kruskal_wallis
from fitr.stats.nonparametric import conover
from fitr.stats.linear_regression import linear_regression
from fitr.stats import effect_size

from fitr.stats import meta_analysis


__all__ = ['bic',
           'lme',
           'pearson_rho',
           'spearman_rho', 
           'binomial_exact',
           'binomial_twosample',
           'kruskal_wallis',
           'conover',
           'linear_regression',
           'effect_size',
           'meta_analysis']
