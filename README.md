# FADngs: Federated Learning for Anomaly Detection

Code for our paper on anomaly detection in federated learning settings, titled FADngs. 
This repository support self-supervised training of networks for federated anomaly detection.


## 1. Preparation
### Environments
Start by installing all dependencies. 

`pip install -r requirements.txt`

### Datasets
We need Cifar-10 and Cifar-100 datasets for experiments.
Please download the above datasets to `~/data`, 
or automatically download when running our code.



## 2. Training

To train the models using method FADngs in the paper, run this command:

```train
CUDA_VISIBLE_DEVICES=0,1 python federated_train_sed.py --arch <NETWORK> --training-mode simclr_CSI --shift_trans_type rotation 
--clusters 10 --ratio_pollution 0.01
```
Option:
* `--exp-name`: Experiment name. We will log results into a text file of this name.
* `--training-mode`: Choose from (`"SimCLR", "SupCon", "SupCE"`). This will choose the right network modules for the checkpoint.
* `--arch`: Choose from available architectures: `"resnet18", "resnet34", "resnet50", "resnet101"`.
* `--shift_trans_type`: Type of shifting transformations. Choose from (`"rotation", "cutperm"`).
* `--clusters`: number of clusters when performing k-means clustering.
* `--ratio_pollution`: ratio of anomaly samples in datasets


