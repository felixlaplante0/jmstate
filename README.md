# jmstate

**jmstate** is a Python package for **nonlinear multi-state joint modeling** of longitudinal and time-to-event data. Built on [PyTorch](https://pytorch.org/), it enables flexible specification of regression and link functions — including neural networks — while still offering built-in parametric baseline hazards and utilities for inference and prediction.

The package implements the methodology from:

> **A General Framework for Joint Multi-State Models**
> Félix Laplante & Christophe Ambroise (2025) — [arXiv:2510.07128](https://arxiv.org/abs/2510.07128)

---

## Installation

```bash
pip install jmstate
```

**Requirements:** Python ≥ 3.10, PyTorch, scikit-learn, NumPy, Matplotlib, rich, tqdm.

---

## Documentation

Full API reference and tutorials: [jmstate documentation](https://felixlaplante0.github.io/jmstate/)

---

## The Model

`jmstate` fits a **joint model** that links a longitudinal biomarker process to multi-state event history through shared individual random effects.

### Longitudinal sub-model

Individual observations follow

$$y_{ij} = h(t_{ij},\, \psi_i) + \epsilon_{ij}, \qquad \epsilon_{ij} \overset{\text{iid}}{\sim} \mathcal{N}(0,\, R)$$

where $h$ is a user-defined regression function (e.g. bi-exponential, logistic) and individual parameters are defined via

$$\psi_i = f(\gamma, X_i, b_i), \qquad b_i \sim \mathcal{N}(0,\, Q)$$

with $\gamma$ fixed population-level effects, $X_i$ covariates, and $b_i$ individual random effects.

### Multi-state sub-model

Let $G = (V, E)$ be a directed graph, where $V$ denotes the set of states and $E \subseteq V \times V$ the set of admissible transitions. The graph encodes all possible paths of the multi-state process, allowing for competing, recurrent, or absorbing transitions. The hazard for a transition $k \to k'$ at time $t$ given entry time $t_0$ satisfies

$$\lambda^{k \to k'}(t_0, t) = \lambda_0^{k \to k'}(t_0, t) \exp\left( \alpha^{k \to k'} g(t, \psi_i) + \beta^{k \to k'} X_i \right),$$

where $\lambda_0^{k \to k'}$ is a parametric baseline hazard, $g$ is a link function summarising the individual longitudinal trajectory, and $\alpha$, $\beta$ are transition-specific coefficients.

The model supports **arbitrary state graphs** (recurrent, absorbing, monotone, etc.) under a semi-Markov assumption.

### Estimation

Parameters are estimated by maximising the observed-data log-likelihood using the **Fisher identity**

$$\nabla_\theta \log \mathcal{L}(\theta;\, x) = \mathbb{E}_{b \sim p(\cdot \mid x, \theta)}\left[ \nabla_\theta \log \mathcal{L}(\theta;\, x, b) \right],$$

where $\mathcal{L}(\theta;\, x, b)$ is the complete likelihood of the data given the parameters and random effects.

This gradient is approximated via a **Metropolis-Within-Gibbs MCMC** sampler over the random effects, combined with a stochastic gradient ascent step. Convergence is monitored via an $R^2$-based stationarity test.

---

## Quick Start

### Step 1 — Define the model design

```python
import torch
from jmstate.types import ModelDesign


# Individual parameters
def indiv_effects_fn(
    fixed: torch.Tensor, x: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    return fixed * torch.exp(b)  # (..., n, q)


# PK function: bi-exponential biomarker
def pk_fn(t: torch.Tensor, indiv_params: torch.Tensor, D: float = 1.0):
    A, k, ka = indiv_params.chunk(3, dim=-1)
    conc = A * (torch.exp(-k * t) - torch.exp(-ka * t))
    return conc.unsqueeze(-1)


# PK integral function: bi-exponential cumulative link
def pk_integral_fn(t: torch.Tensor, indiv_params: torch.Tensor):
    A, k, ka = indiv_params.chunk(3, dim=-1)
    integral = A * (
        (1.0 / k) * (1 - torch.exp(-k * t)) - (1.0 / ka) * (1 - torch.exp(-ka * t))
    )
    return integral.unsqueeze(-1)


# Define the model design
design = ModelDesign(
    indiv_effects_fn=indiv_effects_fn,
    regression_fn=pk_fn,
    link_fns={
        (1, 1): pk_integral_fn,
        (1, 2): pk_integral_fn,
    },
)
```

### Step 2 — Set initial parameters

```python
from jmstate.functions.base_hazards import Exponential
from jmstate.types import ModelParameters, PrecisionParameters

# Define simple initial parameters
params = ModelParameters(
    torch.ones(3),
    PrecisionParameters.from_covariance(torch.eye(3), "diag"),
    PrecisionParameters.from_covariance(torch.eye(1), "spherical"),
    {(1, 1): Exponential(1.0), (1, 2): Exponential(1.0)},
    {(1, 1): torch.zeros(1), (1, 2): torch.zeros(1)},
    {(1, 1): torch.zeros(1), (1, 2): torch.zeros(1)},
)
```

### Step 3 — Prepare data

```python
from jmstate.types import ModelData

data = ModelData(
    x=x,  # (n, p) covariate matrix
    t=t_obs,  # (m,) or (n, m) measurement times; NaN-pad if variable
    y=y_obs,  # (n, m, d) longitudinal observations; NaN-pad if variable
    trajectories=trajectories,  # list[list[tuple[float, Any]]]
    c=c,  # (n, 1) right-censoring times
)
```

Each trajectory is a chronologically ordered list of `(time, state)` tuples representing the individual's event history.

### Step 4 — Fit the model

```python
import matplotlib.pyplot as plt
from jmstate import MultiStateJointModel

optimizer = torch.optim.Adam(params.parameters(), lr=0.1)
model = MultiStateJointModel(design, params, optimizer)

metrics = model.fit(data)
```

### Step 5 — Print the results

```python
from jmstate.utils import plot_params_history, summary

# Print summary statistics (nullity Wald statistics, p-values, AIC, BIC, etc.)
summary(model.params)

# Plot parameter history (stochastic optimization)
plot_params_history(metrics)
plt.show()
```

---

## Mathematical to code mapping

### `ModelDesign`

Specifies the functional form of the model.

| Argument                | Mathematical equivalent                                 | Description                                                                 |
| ----------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------- |
| `individual_effects_fn` | $(\gamma, X, b) \mapsto f(\gamma, X, b) \eqqcolon \psi$ | Maps fixed effects, covariates, and random effects to individual parameters |
| `regression_fn`         | $t \mapsto f(t, \psi)$                                  | Predicted longitudinal response                                             |
| `surv`                  | $\left\{ (k, k') \in E : g^{k \to k'} \right\}$         | Survival design: one entry per transition                                   |

All functions must support broadcasting across MCMC chain and individual dimensions.

### `ModelParams`

Holds all model parameters as PyTorch `nn.Parameter` objects.

| Field          | Mathmetaical equivalent                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------------------------- |
| `fixed_params` | Fixed population-level effects: $\gamma \in \mathbb{R}^q$                                                         |
| `random_prec`  | Representation of random-effect precision matrix: $\log$-Cholesky factor of $Q^{-1}$                              |
| `noise_prec`   | Representation of noise precision: $\log$-Cholesky factor of $R^{-1}$                                             |
| `base_hazards` | Dictionary of base hazards function in $\log$-scale: $\left\{ (k, k') \in E : \log \lambda_0^{k \to k'} \right\}$ |
| `link_coefs`   | Dictionary of link coefficients: $\left\{ (k, k') \in E : \alpha^{k \to k'} \right\}$                             |
| `x_coefs`      | Dictionary of covariate effects: $\left\{ (k, s) \in E : \beta^{k \to k'} \right\}$                               |

Use `PrecisionParameters.from_covariance` to initialize from a covariance matrix.
`precision_type` options: `"full"` ($\log$-Cholesky), `"diag"` ($\log$-diagonal), `"spherical"` ($\log$-scalar).

### `ModelData` and `SampleData`

`ModelData(x, t, y, trajectories, c)` — training data with NaN-padding support.

`SampleData(x, trajectories, indiv_params, t_cond)` — input for simulation (exact individual parameters).

---

## License

See [LICENSE](LICENSE).
