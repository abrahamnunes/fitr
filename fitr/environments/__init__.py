# -*- coding: utf-8 -*-
from fitr.environments.graph import Graph
from fitr.environments.utils import generate_behavioural_data
from fitr.environments.dawtwostep import DawTwoStep
from fitr.environments.igt import IGT
from fitr.environments.kooltwostep import KoolTwoStep
from fitr.environments.mouthtask import MouthTask
from fitr.environments.orthogonal_gonogo import OrthogonalGoNoGo
from fitr.environments.randombandit import RandomContextualBandit
from fitr.environments.twoarmedbandit import TwoArmedBandit


__all__ = ['Graph',
           'generate_behavioural_data',
           'DawTwoStep',
           'IGT',
           'KoolTwoStep',
           'MouthTask'
           'OrthogonalGoNoGo'
           'TwoArmedBandit']
