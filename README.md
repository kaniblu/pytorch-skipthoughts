# Skip-Thoughts in PyTorch

This is a multi-GPU and general implementation of skip-thoughts in PyTorch.
The implementation has been optimized to maximize GPU utilization, while keeping
the memory footprint low by reading data from the disk.

## Prerequisites

**This library has only been tested on Python 3.6.**

This library uses `visdom` (https://github.com/facebookresearch/visdom) as the 
basic visualization tool. Additional requirements are listed under `requires` in
`setup.py`. You can install them by running

    python setup.py install_requires

Or you could simply install the entire package using the following command

    python setup.py install
    
The package is also availble from pypi.

    pip install pytorch-skipthoughts
    
    
## Training

1. Run visdom server by executing

    `python -m visdom.server`

2. Prepare a corpus in the same format as the Toronto Book Corpus (words separated
by spaces, sentences separated by new lines). Create a vocabulary file by executing

    `python -m torchtextutils.vocab --data-path <corpus path or directory>`
    
3. Create a configuration file for training. Available options are listed in `python train.py --help`. An example configuration file is listed under examples.

4. Start training by specifying the configuration file to `train.py`.

    `python -m torchst.train --config <path to configuration file>`
    
5. Get live results from the visdom server. Models are saved to `save-dir` and 
can be used for converting sentences to vectors.

Recommended directory structure is as follows

```
- code directory
- corpus directory
- experiment root directory
  - vocabulary file
  - training configuration file
  - trial 1 directory
    - checkpoint 1
    - checkpoint 2
    - ...
  - trial 2 directory
    - checkpoint 1
    - checkpoint 2
    - ...
  - ...

```

This ensures that codes, datasets and model files are all separated and stored
in their dedicated places.