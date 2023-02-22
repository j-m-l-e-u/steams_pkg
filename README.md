# steams

Space-time prediction with sparse and irregular space-time multi-timeseries.

Models presented in this packages are using an *adaptive distance attention mechanism*.

The weight of the attention are based either on the Ordinary Kriging equation system or the Nadaraya-Watson Kernel.

We use `Pytorch` to train the learnable parameters.

## Install from PyPi
```bash
pip install steams
```


## install from source
```bash
cd /tmp
git clone https://github.com/j-m-l-e-u/steams_pkg.git
cd steams_pkg
pip3 install -e .
```

Package `steams` has been tested on python 3.8 and 3.9

Running `steams` with CUDA (v11.3), requires a manual installation of `Pytorch`:
```bash
pip3 install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

This package was initially developed while working at NILU. The repo has been transferred here. This is now the official repo.
