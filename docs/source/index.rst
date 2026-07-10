Joint Multi-State Modeling
===========================

**jmstate** is a Python package for nonlinear multi-state joint modeling of longitudinal and time-to-event data. Built on PyTorch, it supports flexible regression and link functions, including neural networks, alongside parametric baseline hazards and tools for inference and prediction.

Features
--------

- **Flexible longitudinal models**: Supports user-defined regression and individual-effects functions.
- **General multi-state processes**: Handles arbitrary recurrent, absorbing, and monotone state graphs under a semi-Markov assumption.
- **Shared random effects**: Links longitudinal biomarkers and transition hazards through individual random effects.
- **Parametric baseline hazards**: Includes exponential, Weibull, Gompertz, and log-normal baseline hazard functions.
- **Automatic differentiation**: Uses PyTorch optimization for likelihood-based estimation.
- **Inference and prediction**: Provides MCMC diagnostics, parameter summaries, and trajectory prediction utilities.

Method
------

The longitudinal sub-model is

.. math::

   y_{ij} = h(t_{ij}, \psi_i) + \epsilon_{ij}, \qquad \epsilon_{ij} \sim \mathcal{N}(0, R),

where :math:`h` is a user-defined regression function and the individual parameters are

.. math::

   \psi_i = f(\gamma, X_i, b_i), \qquad b_i \sim \mathcal{N}(0, Q).

For a transition :math:`k \to k'` at time :math:`t` after entering the current state at :math:`t_0`, the multi-state sub-model specifies

.. math::

   \lambda^{k \to k'}(t_0, t) = \lambda_0^{k \to k'}(t_0, t) \exp\left(\alpha^{k \to k'} g^{k \to k'}(t, \psi_i) + \beta^{k \to k'} X_i\right).

The model estimates parameters by maximizing the observed-data log-likelihood. Its gradient is evaluated with the Fisher identity and approximated using a Metropolis-within-Gibbs sampler over the random effects combined with stochastic gradient optimization.

Installation
------------

Install the package from PyPI:

.. code-block:: bash

   python -m pip install jmstate

Usage
-----

Define a model design, initialize its parameters, and fit it to longitudinal and multi-state data:

.. code-block:: python

   import torch
   from jmstate import MultiStateJointModel
   from jmstate.functions.base_hazards import Exponential
   from jmstate.types import ModelData, ModelDesign, ModelParameters, PrecisionParameters

   def individual_parameters(fixed, x, random_effects):
       return fixed * torch.exp(random_effects)

   def regression(t, parameters):
       amplitude, elimination, absorption = parameters.chunk(3, dim=-1)
       return (amplitude * (torch.exp(-elimination * t) - torch.exp(-absorption * t))).unsqueeze(-1)

   design = ModelDesign(
       individual_parameters,
       regression_fn=regression,
       link_fns={(1, 2): regression},
   )
   parameters = ModelParameters(
       torch.ones(3),
       PrecisionParameters.from_covariance(torch.eye(3), "diag"),
       PrecisionParameters.from_covariance(torch.eye(1), "spherical"),
       {(1, 2): Exponential(1.0)},
       {(1, 2): torch.zeros(1)},
       {(1, 2): torch.zeros(1)},
   )
   model = MultiStateJointModel(design, parameters, torch.optim.Adam(parameters.parameters()))
   model.fit(ModelData(x, t, y, trajectories, c))

Configuration
-------------

``ModelDesign`` defines the individual-effects, regression, and transition-link functions. ``ModelParameters`` holds population effects, precision parameters, baseline hazards, and transition coefficients. The model accepts any PyTorch optimizer; the fitted estimator provides parameter summaries, diagnostics, and prediction methods.

API Reference
-------------

.. toctree::
   :maxdepth: 2

   modules
