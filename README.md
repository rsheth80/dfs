# DFS
Python code and Jupyter notebook demo related to the paper "Differentiable feature selection by discrete relaxation", AISTATS, 2020.

Rishit Sheth, 6/16/2020

## Running DFS

DFS is run through a command-line interface.
The only required argument is the name of a dataset whose properties (fn_train, fn_eval, ncols, nrows, nrows_test, zero_based, neg_label, binary, n_for_stats, disk) are populated in [datasets.yml](datasets.yml).
The properties of the datasets included in the paper experiments (see below) are provided.
Run `python dfs.py -h` to see options and defaults.
Open the Jupyter notebook [demo.ipynb](demo.ipynb) to see an example run plus evaluation.

## Classes

The two primary classes in [learnability.py](learnability.py) are `DatasetSVM` which provides data access and `Solver` which performs the differentiable feature selection.

## Auxiliary functions

In the case of disk reading, training can be sped up considerably by utilizing PyTorch's `num_workers>1` in `DataLoader` instantiations.
In order to take advantage of this efficiency, we include `create_mappings_file` and `load_mappings_file` in the `DatasetSVM` class.
The first function performs one pass through the train file, recording and dumping newline locations.
The second function loads a previously-created mappings file.
When the `mappings` attribute of a `DatasetSVM` instantiation is populated, random access to the train file is available, thus enabling the use of multiple workers by `DataLoader`s (as well as shuffling, if desired).

The `DatasetSVM` functions `dump_data_stats` and `load_data_stats` can be used to save/restore previously-computed values for feature (and label) means, standard deviations, and feature matrix spectral norm.

## Paper results

The file [extra.pdf](extra.pdf) shows the results of additional experiments on the rcv1 dataset where batchsize is varied between 500, 250, and 100 (top row, left to right), sequence of training examples is varied randomly (middle row), and the learning rate is varied between 1e-2, 1e-3, and 1e-4 (bottom, left to right).
In all cases, DFS has superior performance.
Specific observations: (a) for batch size, we note that DFS performance can improve in some cases for smaller batch sizes (as noted in the paper, using SGD helps provide some guard against local optima; (b) for the example sequence, there is some variance but it is not very significant; also, the sequence is usually fixed in the large-scale settings, e.g., limited memory, streaming; (c) as learning rate is decreased, while keeping total training iterations fixed, fewer features are "turned off";

The file [results.zip](results.zip) includes results from fresh runs of DFS on the three largest datasets of the paper experiments.
These newer results differ in two minor ways from the paper results:
1. They were generated using a coarser grid of penalty values.
2. The first 10000 example/label pairs of a dataset were used to estimate data statistics whereas, in the paper, 10000 randomly-selected pairs were used.
Using the first segment of a dataset is more appropriate for streaming scenarios.

## MISSION

A slightly modified version of the [MISSION](https://arxiv.org/abs/1806.04310) implementation is provided (original implementation [here](https://github.com/rdspring1/MISSION)).
In the paper, MISSION was utilized (i) as a competitor feature selection algorithm in the large scale setting, and (ii) as an evaluator of feature selection performance (on test) via its online SGD classifier.

[mission_logistic](MISSION/src/mission_logistic.cpp) runs MISSION feature selection/train + eval, taking a desired number of features as input.
[mission_logistic_eval](MISSION/src/mission_logistic_eval.cpp) runs MISSION train + eval, taking a text file containing the desired features as input.
The two columns of the output represent true {0,1} label and predicted probability of class 1.
Makefiles and source are in [MISSION/src](MISSION/src).

### Example: selecting 150 features with MISSION and saving predictions on test to file
```
mission_logistic rcv1_train.binary rcv1_test.binary 150 1 > features.150.eval.txt
```
### Example: utilizing MISSION to evaluate 150 previously-selected features and saving predictions on test results to file
```
mission_logistic_eval rcv1_train.binary rcv1_test.binary features.150.txt 1 > features.150.eval.txt
```
The last argument in the preceding examples is a boolean denoting whether the train/test SVM light files are encoded with {-1,+1} labels (1) or {0,1} labels (0).

## Datasets

* [*mnist*](mnist-35-noisy-binary.tar.gz) contains the train/test splits for classification of 3s versus 5s with added noise.

* *gisette* [train](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2) and [test](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.t.bz2).

* *rcv1* [train](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2) and [test](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2).

* The [*webgram*](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2) train/test split was 80/20 with the first portion of the file representing train (280000 rows).

* The [*criteo*](https://s3-us-west-2.amazonaws.com/criteo-public-svm-data/criteo.kaggle2014.svm.tar.gz) test set is unlabeled, so we split the train file 80/20 with the first portion of the file representing train (36672494 rows).

Dataset [properties](datasets.yml).
