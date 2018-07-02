# -*- coding: utf-8 -*-
from fitr.inference.optimization_result import OptimizationResult
from fitr.inference.mle_parallel import mlepar
from fitr.inference.batch_gradient_descent import batch_gradient_descent

__all__ = ['OptimizationResult',
           'mlepar',
           'batch_gradient_descent']
