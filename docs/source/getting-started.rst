Getting started
===============

Install ``jmstate`` from PyPI:

.. code-block:: bash

   python -m pip install jmstate

Define a model design, initialize its parameters, and fit it to longitudinal and
multi-state data:

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

The :class:`~jmstate.types.ModelDesign` object defines the individual-effects,
regression, and transition-link functions. ``ModelParameters`` holds population
effects, precision parameters, baseline hazards, and transition coefficients.

Next, read the :doc:`model-guide` or work through the :doc:`paquid-test` and
:doc:`fitting-test` analyses.
