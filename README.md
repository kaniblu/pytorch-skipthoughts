# Skip-Thoughts in PyTorch

This is a multi-GPU and general implementation of skip-thoughts in PyTorch.
The implementation has been optimized to maximize GPU utilization, while keeping
the memory footprint low by reading data from the disk.

## Prerequisites

**This library has only been tested on Python 3.6.**

### Visualization ###

This library uses either [`visdom`](https://github.com/facebookresearch/visdom) or
[`tensorboard`](ttps://github.com/dmlc/tensorboard) as the backend visualization tool. 
Details of using the libraries could be checked on each of the sites.

### Libraries ####

Additional requirements are listed under `requires` in
`setup.py`. You can install them by running

    python setup.py install_requires

Or you could simply install the entire package using the following command

    python setup.py install
    
The package is also availble from pypi.

    pip install pytorch-skipthoughts
    
## Training ##

1. Run visualization server by executing

    `python -m visdom.server`
    
    or

    `tensorboard --logdir=$SAVE_DIR`
    
    If you are using tensorboard as the backend, the directory where you save checkpoints
    (`save-dir` option) must be specified, as that's where summaries are written.

2. Prepare a corpus in the same format as the Toronto Book Corpus (words tokenizable
   by spaces, sentences separated by new lines). Then create a vocabulary file by executing

    `python -m torchtextutils.vocab --data_dir $CORPUS_PATH --vocab_path $VOCAB_PATH --cutoff 20000`
    
    `cutoff` specifies the number of top occurring words to leave in the vocabulary set.
    In R. Kiros' paper, `20000` was the cutoff threshold.

3. Create a configuration file for training. Available options are listed in `python train.py --help`. An example configuration file is listed under examples. Many skip-thoughts model options are available, including:

    - `encoder-cell`: encoder cell type (`lstm`, `gru`)
    - `decoder-cell`: decoder cell type (`lstm`, `gru`)
    - `before`: number of contextually preceding sentences to decode
    - `after`: number of contextually succeeding sentences to decode
    - `predict-self`: whether to predict oneself (Tang et al., Rethinking Skip-thought)
    - `encoder-direction`: directions to encode sentences (`uni`, `bi`, `combine`) (yes this library supports training combine-skip in a single run)
    - `dropout-prob`
    - `conditional-decoding`: whether to feed last encoder state to every time step of decoder(s)

    Both YAML and JSON are all supported.

4. After carefully designing an experimental setup, start training by specifying the configuration file to `train.py`.
    You must have the library installed for module-wise use. 

    `python -m torchst.train --config $CONFIG_PATH`
    
5. Get live results from the visualization server. Models are saved to `save-dir` and 
    can be used for converting sentences to vectors.

## Inference ##

Trained models can be used to encode sentences. As a starting point, encoding new sentences can be achieved
using `vectorize.py` script. Similar to training script, encoder script also supports configuration files,
which can be saved to disk and repeatedly used to call the script more efficiently.

    `python -m torchst.vectorize --config $CONFIG_PATH`
    
An example of `vectorize.py` configuration file has been provided in `examples`.

To save results of the encoding, simple pipe the results to disk:
    
    `python -m torchst.vectorize --config $CONFIG_PATH > vectors.txt`
    
The results can be read using [`numpy.loadtxt`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html).

Some tips:

  - the model parameters provided in the configuration file must be identical to those for
training the model.

  - `vectorize.py` could be run as a one-time executable or queryable server like `fasttext` (https://github.com/FacebookResearch/FastText). To run the script as a queryable server, do not specify `data-path` option. 

  - when run as a queryable server, the script could sometimes be required to return results immediately even if
  the number of input sentences has not reached `batch-size`. You could specify a special character with `flush-char` that would signify the script to return results immediately. By default, it is `0x05` (enquiry) in ascii code, which is `ctrl-d` in terminal.
  
### Vocabulary Expansion ###

(T. Mikolov, 2013)

If you did not freeze word embeddings during training, the chance is that the model would not
be able to effectively handle unseen words. Train a word embedding translation model from
pretrained to those learnt by our model using `wordembed.py`

    `python -m torchst.wordembed --config $CONFIG_PATH`
    
Likewise, you can use configuration files to organize experimental setups.

Trained word embedding translation models then could be fed into `vectorize.py` to translate
word embeddings from larger vocabulary sets such as GloVe to embeddings understood by our model.
