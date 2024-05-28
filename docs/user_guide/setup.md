# Setup and Installation

## Requirements

attpc_engine requires a Python version between 3.10 and 3.12 (inclusive)

## Installation

attpc_engine can be installed using pip:

```bash
pip install attpc_engine
```

It is recommended to install attpc_engine to a virtual environment to avoid dependency clashes.

## attpc_engine @ FRIB

In the FRIB computing infrastructure you may not have control over which Python versions are installed globally to your machine. In this case we recommend using the (typically) installed conda distribution to create a conda env that has the version of Python you need. 

Creating a conda environment with the needed Python is done by running something like

```bash
conda create --name engine-env python=3.11
```

Then you can activate that env using `conda activate engine-env` and use `pip install attpc_engine` as normal.

## Package dependencies

- [spyral-utils](https://attpc.github.com/spyral-utils)
- [h5py](https://www.h5py.org/)
- [numba](https://numba.readthedocs.io)
- [tqdm](https://github.com/tqdm/tqdm)

And all of these packages dependencies. Thanks to everyone who develops these tools!