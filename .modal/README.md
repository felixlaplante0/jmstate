# `.modal` — T4 runner

`jmstate-modal.py` holds all Modal code. Notebooks in `../scripts` stay pure:
the repo lives on the `jmstate` Volume mounted at `/mnt/jmstate`; the runner
executes **one** notebook headless with `nbconvert` and saves it **with
outputs** back onto the volume.

## Setup (once)

```bash
uv tool install modal  # CLI (volume ls/get, setup)
modal setup            # browser login, writes ~/.modal.toml
```

GPUs need a payment method on file (Modal dashboard → Billing).

## Upload the repo (once, repeat after each edit)

```bash
modal volume create jmstate
modal volume put jmstate . /
```

## Use

Launch detached so you can close your computer; the run continues on Modal:

```bash
modal run --detach .modal/jmstate-modal.py --notebook fitting-test
```

Only the `--notebook` you name runs; the rest is just uploaded, never executed.

## Progress

Detaching prints an app ID. Stream the run's logs with it:

```bash
modal app logs ap-xxxxxxxx -f
```

Live logs show function output (start, errors, finish). The notebooks'
`tqdm` bars are captured per cell, so the full bars are in the executed
notebook afterwards — pull it from `results/` (see below).

## Results

The `jmstate` volume mirrors the git repo, so paths match local runs:
executed notebooks and CSVs under `results/`, figures under `figures/`.

```bash
modal volume ls jmstate results
modal volume get jmstate results/fitting-test-executed.ipynb ./fitting-test-executed.ipynb
modal volume ls jmstate figures
modal volume get jmstate figures/paquid-metrics.pdf ./paquid-metrics.pdf
```
