# Compositional Generalisation with Structured Reordering and Fertility Layers
This is the official code for our EACL 2023 paper [Compositional Generalisation with Structured Reordering and Fertility Layers](https://aclanthology.org/2023.eacl-main.159/).


## Usage
### Installation
Clone this repository, and unzip `data.zip`. Then create a conda environment with Python 3.8:
```
conda create -n f-r python=3.8
conda activate f-r
```
And install [PyTorch 1.8](https://pytorch.org/get-started/previous-versions/):

```
# CUDA 10.2
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102

# CUDA 11.1
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# CPU Only
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu
```
Then install the remaining requirements (this may take a while):
```
pip install -r requirements.txt
```

### Using an existing model
You can download some of models we have trained (go to releases). You can compute **evaluation** metrics on some data like this:
```
allennlp evaluate [path/to/model.tar.gz] [path/to/data.tsv] --include-package fertility
```
If you want to use a cuda device, add `--cuda-device [device_id]` to that. 
See `allennlp evaluate --help` for more options such as saving
the metrics without rounding.

If you want to save predictions of the model on same data (e.g. for error analysis), use this:
```
allennlp predict [path/to/model.tar.gz] [path/to/data.tsv] --include-package fertility --use-dataset-reader --output-file [outout-file.jsonl]
```

### Training a model
A configuration file describes a model, the data you want to train on and all hyperparameters. Pick a configuration file from `configs/`
Ensure that `[train|dev|test]_data_path` point to the right data; also make sure that `pretrained_file` points to your copy of the [GloVe embeddings](https://nlp.stanford.edu/data/glove.6B.zip). Then run
```
allennlp train [path/to/config/file.json] -s [path/to/model] --include-package fertility
```
If you want the experiment to be logged by neptune, make sure that `trainer.callbacks` contains an entry like this:
``` 
{ "type": "neptune", "project_name": "[project name]" }
```
I you want to tune hyperparameters automatically, you can use [this fork of allentune](https://github.com/namednil/allentune). Files with search spaces for hyperparameters are in `hyper_param_search/`.

## Fertility Layer
If you are mainly interested in the fertility layer, have a look at `fertility/fertility_numba.py`. 
It contains the dynamic program, and an example demonstrating its use. 
The code to use it as part of the overall model is in `fertility/fertility_model.py`.

We implemented the dynamic program and its backward pass manually using [numba](https://numba.pydata.org/), which was much easier than using PyTorch operations.
While the computation takes place on the CPU, we found this implementation is still reasonably fast.

## Citation

```
@inproceedings{lindemann-etal-2023-compositional,
    title = "Compositional Generalisation with Structured Reordering and Fertility Layers",
    author = "Lindemann, Matthias  and
      Koller, Alexander  and
      Titov, Ivan",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.159",
    pages = "2172--2186"
}
```
