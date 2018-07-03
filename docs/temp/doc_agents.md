# `fitr.agents`

A modular way to build and test reinforcement learning agents.

There are three main submodules:

- `fitr.agents.policies`: which describe a class of functions essentially representing $f:\mathcal X \to \mathcal U$
- `fitr.agents.value_functions`: which describe a class of functions essentially representing $\mathcal V: \mathcal X \to \mathbb R$ and/or $\mathcal Q: \mathcal Q \times \mathcal U \to \mathbb R$
- `fitr.agents.agents`: classes of agents that are combinations of policies and value functions, along with some convenience functions for generating data from `fitr.environments.Graph` environments.

{{agents}}
