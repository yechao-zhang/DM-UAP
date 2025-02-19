# DM-UAP
The official implementation of paper titled, "Improving Generalization of Universal Adversarial Perturbation via Dynamic Maximin Optimization. [AAAI 2025]
## Dependencies
The repo is recommended to be used with python=3.9, torch=2.0.1. All dependencies can be installed with following command:
```
pip install -r requirements.txt
```
## Data preparation
To start with the repo, the ImageNet validation set and a subset of ImageNet training set is need. The validation set is for UAP testing, and the subset of training set is for training. Note that for training subset, retaining the original file structure of training set is not necessary. But you need to update the number of images for training, and the root directory of the training set and validation set in [run.sh](run.sh).

## Training
To start training, run imagenet_attack.py as showed in run.sh:
```
python imagenet_attack.py --data_dir path/to/your/dataset/ 
    --uaps_save "path/to/your/save_dir/" 
    --batch_size 125 --alpha 10 --epoch 20 --dm 1 
    --num_images 500 
    --model_name vgg19 --Momentum 0 --cross_loss 1
    --rho 4 --steps 10 --aa 25 --cc 10 --smooth_rate 0.2 
```
This is to craft a uap from the surrogate model VGG19, with rho of min-theta 4 and eps of min-x 25. The uaps will be crafted in the save_dir you specified. More details can be found in [imagenet_attack.py](imagenet_attack.py).

## Testing
To start testing, run imagenet_eval.py as showed in run.sh:
```
python imagenet_eval.py --data_dir path/to/your/imagenet_val_set/ILSVRC2012_img_val/ 
  --uaps_save "path/to/your/save_dir/delta_file_name" 
  --batch_size 125  --number 1000 
  --model_name vgg19 2>&1|tee -a "path/to/your/save_dir/result.log"
```
This will start testing your uap on model VGG19, and record the results in result.log. More details can be found in [imagenet_eval.py](imagenet_eval.py).

## Acknowledgements
This repo is built on [SGA](https://github.com/liuxuannan/Stochastic-Gradient-Aggregation). We sincerely thank them for their outstanding work.
