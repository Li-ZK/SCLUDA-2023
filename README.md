## Supervised Contrastive Learning-Based Unsupervised Domain Adaptation for Hyperspectral Image Classification
This is a code demo for the paper "Supervised Contrastive Learning-Based Unsupervised Domain Adaptation for Hyperspectral
Image Classification"


## Requirements
CUDA = 11.4

Python = 3.9

Pytorch = 1.10.0

sklearn = 1.0.1

numpy = 1.21.2

cleanlab = 1.0

## dataset

You can download the hyperspectral datasets in mat format at:https://pan.baidu.com/s/184BXDD2KnlreqXX70Nar4Q?pwd=kfgj, and move the files to `./datasets` folder.

An example dataset folder has the following structure:

```
datasets
├── Houston
│   ├── Houston13.mat
│   └── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
├── Pavia
│   ├── paviaU.mat
│   └── paviaU_gt_7.mat
│   ├── pavia.mat
│   └── pavia_gt_7.mat
│── Shanghai-Hangzhou
│   └── DataCube.mat
```

## Usage:
Take SCLUDA method on the UP2PC dataset as an example: 
1. Open a terminal or put it into a pycharm project. 
2. Put the dataset into the correct path. 
3. Run SCLUDA_UP2PC.py.

