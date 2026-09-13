jmstate
========

.. raw:: html

   <section class="hero">
     <img class="hero-logo" src="_static/jmstate-logo.svg" alt="jmstate logo">
     <p class="eyebrow">JOINT MULTI-STATE MODELING</p>
     <h1>Flexible models for longitudinal and event data.</h1>
     <p class="hero-copy">jmstate connects longitudinal biomarkers and multi-state event histories through shared random effects, automatic differentiation, and parametric baseline hazards.</p>
     <div class="hero-actions">
       <a class="primary" href="getting-started.html">Get started</a>
       <a class="secondary" href="paquid-test.html">See the examples</a>
     </div>
   </section>

.. raw:: html

   <aside class="pypi-card">
     <div>
       <span class="pypi-kicker">PYTHON PACKAGE</span>
       <strong>Available on PyPI</strong>
       <p>Install jmstate and build a joint model with familiar PyTorch objects.</p>
     </div>
     <a href="https://pypi.org/project/jmstate/">View package&nbsp;→</a>
   </aside>

Why jmstate?
------------

jmstate provides a flexible framework for nonlinear joint multi-state models of
longitudinal and time-to-event data.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Flexible model design
      :class-card: feature-card

      Define individual-effects, regression, and transition-link functions for the model you need.

   .. grid-item-card:: General state graphs
      :class-card: feature-card

      Work with recurrent, absorbing, and monotone processes under a semi-Markov assumption.

   .. grid-item-card:: Inference and prediction
      :class-card: feature-card

      Fit with automatic differentiation, inspect MCMC diagnostics, and predict model quantities.

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

.. toctree::
   :hidden:
   :maxdepth: 2

   getting-started
   model-guide
   paquid-test
   fitting-test
   modules
