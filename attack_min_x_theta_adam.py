import torch
import torch.nn as nn
from utils import model_imgnet
from PGDL2 import PGD_theta_x_apart
import copy
import torch.optim as optim
import numpy as np

# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

def cal_loss(loader, model, delta, beta, loss_function):
    loss_total = 0
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    delta = delta.cuda()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            x_val = data.cuda()
            outputs_ori = model(x_val.cuda())
            _, target_label = torch.max(outputs_ori, 1)
            perturbed = torch.clamp((x_val + delta), 0, 1)
            outputs = model(perturbed)
            if loss_function:
                loss = torch.mean(loss_fn(outputs, target_label))
            else:
                loss = torch.mean(outputs.gather(1, (target_label.cuda()).unsqueeze(1)).squeeze(1))
            loss_total = loss_total + loss
    loss_total = loss_total / (i + 1)
    return loss_total


def uap_dm(model, loader, nb_epoch, eps, beta=9, step_decay=0.1, loss_function=None,
             uap_init=None, batch_size=None, loader_eval=None, dir_uap=None, center_crop=224, Momentum=0,
             img_num=10000, rho=1, aa=10, cc=10, steps=10,smooth_rate=-0.2):
    '''
    IcudaT
    model       model
    loader      dataloader
    nb_epoch    number of optimization epochs
    eps         maximum perturbation value (L-infinity) norm
    beta        clamping value
    step_decay  single step size
    loss_fn     custom loss function (default is CrossEntropyLoss)
    uap_init    custom perturbation to start from (default is random vector with pixel values {-eps, eps})
    center_crop image size
    Momentum    momentum item (default is false)

    log output
    batch_size  batch size
    loader_eval evaluation dataloader
    dir_uap     save patch
    img_num     total image num
    '''
    model.eval()
    DEVICE = torch.device("cuda:0")
    delta = -2 * eps * torch.rand((1, 3, center_crop, center_crop), device=DEVICE) + eps
    # delta = torch.zeros(1 , 3, center_crop, center_crop).cuda()
    delta.requires_grad = True
    losses = []
    losses_min = []
    if loss_function:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        beta = torch.cuda.FloatTensor([beta])

        def clamped_loss(output, target):
            loss = torch.mean(torch.min(loss_fn(output, target), beta))
            return loss

        criterion = nn.CrossEntropyLoss()

    else:
        def logits_loss(output, target):
            loss = -torch.mean(output.gather(1, target.unsqueeze(1)).squeeze(1))
            return loss

        criterion = logits_loss

    v = 0

    
    ori_state_dict = copy.deepcopy(model.state_dict())
    
    # curriculum learning, current rho and eps increase from zero to max, along with epoch growing
    rho_step = rho / nb_epoch
    eps_step1 = aa / nb_epoch
    rho_current = 0
    eps_current = 0
    optimizer = optim.AdamW([delta], lr=0.1)

    for epoch in range(nb_epoch):

        print('epoch %i/%i' % (epoch + 1, nb_epoch))
        rho_current += rho_step
        eps_current += eps_step1
        attacker = PGD_theta_x_apart(eps=eps_current, rho=rho_current, x_steps=cc, theta_steps=steps,
                                     random_start=False, eps_for_division=1e-10, loss_function=loss_function,
                                     smooth_rate=smooth_rate)
        # perturbation step size with decay
        eps_step = eps * step_decay

        loss_min_total = 0
        
        for i, data in enumerate(loader):
            with torch.no_grad():
                outputs_ori = model(data.cuda())
                _, target_label = torch.max(outputs_ori, 1)
            optimizer.zero_grad()
            model, x_val = attacker.forward(model, data, target_label)
                                    

            # perturbed = torch.clamp(x_val.cuda() + delta,0,1)
            perturbed = x_val.cuda() + delta
            outputs = model(perturbed)
            # loss function value
            if loss_function:
                loss = -clamped_loss(outputs, target_label.cuda())
            else:
                loss = torch.mean(outputs.gather(1, (target_label.cuda()).unsqueeze(1)).squeeze(1))
            loss.backward()
            # batch update
            optimizer.step()
            with torch.no_grad():
                delta.clamp_(-eps, eps)
            model.load_state_dict(ori_state_dict)
            model.eval()


        loss = cal_loss(loader_eval, model, delta.data, beta, loss_function)
        
        losses.append(torch.mean(loss.data).cpu())
        # if (epoch + 1) % 10 == 0:
        torch.save(delta.data,
                   dir_uap + 'dm_' + '%d_%depoch_%dbatch.pth' % (img_num, epoch + 1, batch_size))

    return delta.data, losses ,losses_min

