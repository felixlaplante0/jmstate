jmstate
=======

**jmstate** is a Python package for nonlinear multi-state joint modeling of
longitudinal and time-to-event data. Built on PyTorch, it lets you specify
regression and link functions, including neural networks, and provides parametric
baseline hazards and utilities for inference and prediction.

.. code-block:: bash

   pip install jmstate

See :doc:`getting-started` for a first model, or the :doc:`paquid-test` example.
The package is available on `PyPI <https://pypi.org/project/jmstate/>`_, and the
method is described in the `paper <https://arxiv.org/abs/2510.07128>`_.

Why jmstate?
------------

jmstate links a longitudinal biomarker process to a multi-state event history
through shared individual random effects. You define the individual-effects,
regression and transition-link functions the model needs, on any state graph
(recurrent, absorbing or monotone) under a semi-Markov assumption. Parameters are
fitted with automatic differentiation, with MCMC diagnostics and prediction of model
quantities built in.

Quick example
-------------

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
   model = MultiStateJointModel(
       design, parameters, torch.optim.Adam(parameters.parameters())
   )
   model.fit(ModelData(x, t, y, trajectories, c))

The :doc:`getting-started` page explains each object, and the :doc:`model-guide`
covers the model specification and estimation.

Explore jmstate
---------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Get started
      :link: getting-started
      :link-type: doc

      Install jmstate and fit your first joint multi-state model.

   .. grid-item-card:: Model guide
      :link: model-guide
      :link-type: doc

      Read the model specification and understand the estimation workflow.

   .. grid-item-card:: Examples
      :link: paquid-test
      :link-type: doc

      Reproduce the PAQUID and simulated analyses from the repository scripts.

Citation
--------

If you use jmstate, please cite:

.. code-block:: bibtex

   @article{laplante2025jmstate,
     title   = {A General Framework for Joint Multi-State Models},
     author  = {Laplante, F{\'e}lix and Ambroise, Christophe},
     journal = {arXiv preprint arXiv:2510.07128},
     year    = {2025},
     doi     = {10.48550/arXiv.2510.07128}
   }

.. toctree::
   :hidden:

   getting-started
   model-guide
   paquid-test
   fitting-test
   modules
