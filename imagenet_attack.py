import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import random
sys.path.append(os.path.realpath('..'))
import argparse
import datetime
from attack_min_x_theta_adam import uap_dm
from attacks_sga_ori import uap_sga

from utils import model_imgnet
from prepare_imagenet_data import create_imagenet_npy

def seed_torch(seed=0):  # random seed 
    torch.random.fork_rng()
    rng_state = torch.random.get_rng_state()
    torch.random.manual_seed(seed)
    torch.random.set_rng_state(rng_state)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    print(args)
    seed_torch(args.seed)
    time1 = datetime.datetime.now()
    dir_uap = args.uaps_save
    batch_size = args.batch_size
    DEVICE = torch.device("cuda:0")
    model_dimension = 299 if args.model_name == 'inception_v3' else 256
    center_crop = 299 if args.model_name == 'inception_v3' else 224
    X = create_imagenet_npy(args.data_dir, len_batch=args.num_images,model_dimension = model_dimension,center_crop=center_crop)
    loader = torch.utils.data.DataLoader(X,batch_size=batch_size,shuffle=True,num_workers=0)
    loader_eval = torch.utils.data.DataLoader(X,batch_size=100,shuffle=True,num_workers=0)
    model = model_imgnet(args.model_name)
    if not os.path.exists(dir_uap):
        os.makedirs(dir_uap)

    nb_epoch = args.epoch
    eps = args.alpha / 255
    beta = args.beta
    step_decay = args.step_decay
    if args.dm:
        uap, losses, losses_min = uap_dm(model, loader, nb_epoch, eps, beta, step_decay, loss_function=args.cross_loss, batch_size=batch_size, loader_eval=loader_eval, dir_uap=dir_uap, center_crop=center_crop, Momentum=args.Momentum, img_num=args.num_images,rho=args.rho,aa=args.aa, cc=args.cc, steps=args.steps,smooth_rate=args.smooth_rate)
    else:
        uap,losses = uap_sga(model, loader, nb_epoch, eps, beta, step_decay, loss_function=args.cross_loss, batch_size=batch_size, minibatch=args.minibatch, loader_eval=loader_eval, dir_uap = dir_uap,center_crop=center_crop,iter=args.iter,Momentum=args.Momentum,img_num=args.num_images)

    if args.dm:
        save_name = 'dm_' + args.model_name
    else:
        save_name = 'sga_' + args.model_name

    plt.plot(losses)
    np.save(dir_uap + "losses_min.npy", losses_min)
    np.save(dir_uap + "losses.npy", losses)
    plt.savefig(dir_uap + save_name + '_loss_epoch.png')

    time2 = datetime.datetime.now()
    print("time consumed: ", time2 - time1)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/../imagenet/train/',
                        help='training set directory')
    parser.add_argument('--uaps_save', default='./uaps_save/dm/',
                        help='training set directory')
    parser.add_argument('--batch_size', type=int, help='batch size', default=125)
    
    

    
    parser.add_argument('--alpha', type=float, default=10, help='aximum perturbation value (L-infinity) norm')
    parser.add_argument('--beta', type=float, default=9, help='clamping value')

    parser.add_argument('--epoch', type=int, default=20, help='epoch num')
    parser.add_argument('--dm', type=int,default=1, help='choose to run DM-UAP(1) or SGA(0)')
    parser.add_argument('--num_images', type=int, default=10000, help='num of training images')
    parser.add_argument('--model_name', default='vgg16', help='proxy model')
    parser.add_argument('--cross_loss', type=int, default=1, help='loss type,default is 1,cross_entropy loss')
    parser.add_argument('--step_decay', type=float, default=0.1, help='delta step size')
    
    
    # Parameters only for SGA
    parser.add_argument('--minibatch', type=int, help='inner batch size for SGA', default=10)
    parser.add_argument('--iter', type=int,default=4, help='inner iteration num')
    parser.add_argument('--Momentum', type=int, default=0, help='Momentum item')
    
    
    # Parameters only for DM-UAP
    parser.add_argument('--rho', type=float, default=4, help='rho of min-theta')
    parser.add_argument('--steps', type=int, default=10, help='min_theta_steps')
    parser.add_argument('--aa', type=float, default=25, help='eps of min-x, diffrent from eps of max-delta')
    parser.add_argument('--cc', type=int, default=10, help='min_x_steps')
    parser.add_argument('--smooth_rate', type=float, default=-0.2, help='label smoothing rate')
    parser.add_argument('--seed', type=int, default=0, help='seed used')


    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
