# steams

Space-time prediction with sparse and irregular space-time multi-timeseries.


## install
```bash
cd /tmp
git clone https://git.nilu.no/aqdl/steams_pkg.git
cd steams_pkg
pip3 install -e .
```

Package 'steams' has been tested on python 3.8 and 3.9

Running 'steams' with CUDA (v11.3), requires a manual installation of pytorch:
```bash
pip3 install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
