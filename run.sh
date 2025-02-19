#/bin/bash

# This script gives a demo of using the item. The imagenet_attack.py is for delta generating, and the imagenet_eval.py is for delta evluating.
# The meaning of the parameters can be find in the py files seperately.


uaps_save_path="path/to/your/save_dir/"

CUDA_VISIBLE_DEVICES=1 python imagenet_attack.py --data_dir path/to/your/dataset/ \
    --uaps_save "$uaps_save_path" \
    --batch_size 125 --alpha 10 --epoch 20 --dm 1 \
    --num_images 500 \
    --model_name vgg19 --Momentum 0 --cross_loss 1\
    --rho 4 --steps 10 --aa 25 --cc 10 --smooth_rate 0.2 \




# Notice that the form of "delta_file_name" is [dm/sga]_[data_num]_[epoch]epoch_[batchsize]batch.pth

uaps_save_path="path/to/your/save_dir/delta_file_name"
uaps_save_path2="path/to/your/save_dir/result.log"

python imagenet_eval.py --data_dir path/to/your/imagenet_val_set/ILSVRC2012_img_val/ \
  --uaps_save "$uaps_save_path" \
  --batch_size 125  --number 1000 \
  --model_name vgg19 2>&1|tee -a "$uaps_save_path2"







   
