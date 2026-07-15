# AGENTS.md

## Conda environment

All dependencies for this project live in the **`heiplanet-data`** conda
environment (created via miniforge or another forge at
`~/miniforge3/envs/heiplanet-data`). The base environment does **not**
have `xarray`, `xesmf`, `cdo`, etc.

**Always activate the environment** before running code or tests — do not call
the environment's Python by its absolute path:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate heiplanet-data
```

Activation matters: some preprocessing steps shell out to command-line binaries
(notably `cdo`, and `esmf`/`esmpy` for `xESMF` downsampling). Those binaries are
installed in the environment's `bin/` directory. Invoking
`~/miniforge3/envs/heiplanet-data/bin/python` directly runs the right
interpreter but does **not** put the environment's `bin/` on `PATH`, so
`subprocess` calls fail with `FileNotFoundError: 'cdo'`. Activating the
environment fixes this.

## Running tests

With the environment activated, from the repository root:

```bash
pytest                                   # full suite
pytest heiplanet_data/test/test_preprocess.py -q   # one module
```

The test suite is expected to pass fully in an activated `heiplanet-data`
environment. If `cdo`/`xesmf` tests fail with a missing-binary error, the
environment is almost certainly not activated (or the optional deps were never
installed — see `README.md` for the `conda install -c conda-forge esmf esmpy`
and `python-cdo` steps).
