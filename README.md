# DM-UAP
The official implementation of paper titled, "Improving Generalization of Universal Adversarial Perturbation via Dynamic Maximin Optimization. [AAAI 2025]
## Dependencies
The repo is recommended to be used with python=3.9, torch=2.0.1. All dependencies can be installed with following commend:
```
pip install -r requirements.txt
```
## Data preparation
To start with the repo, ImageNet validation set and a subset of ImageNet training set is need. Update the number of images for training, and the root directory of the training set and validation set in [run.sh](run.sh).
