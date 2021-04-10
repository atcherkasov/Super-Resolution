import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import albumentations as A
import os
import sys

from data import LRandHR
from RCAN_models import RCAN
from discriminators import NLayerDiscriminator
from args import rcan_args
from train import train, setting_dirs


if __name__ == '__main__':
    # PARAMETERS
    # static
    BATCH_SIZE=16
    LR_PATCH=32
    LR_VAL_PATCH=256
    HR_VAL_PATCH=512
    HR_PATCH=64
    MAX_ITER=3e5
    CUR_ITER=0
    # infro
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    PATH, LOGS = setting_dirs(sys.argv)
    FREEZ_RCAN='/cache/chat_models/models_ECCV2018RCAN/RCAN_BIX2.pt'
    # customize
    coef = {'gamma': 0.1, 'cyc': 0, 'idt': 1, 'geo': 1}
    FREEZE_UPSCALE=False
    
    # MODELS
    models = {}
    optimizers = {}
#######################################################################################
    args_1 = rcan_args()
    models['Gyx'] = RCAN(args_1)
    models['Gyx'].to(device)
    optimizers['Gyx'] = Adam(models['Gyx'].parameters(), betas=(0.5, 0.999), eps=1e-8, lr=1e-4)
    if os.path.exists(f"{PATH}/last_{'Gyx'}.pth"):
        checkpoint = torch.load(f"{PATH}/last_{'Gyx'}.pth")
        models['Gyx'].load_state_dict(checkpoint['model_state_dict'])
        optimizers['Gyx'].load_state_dict(checkpoint['optimizer_state_dict'])
#######################################################################################
    args_2 = rcan_args()
    models['Gxy'] = RCAN(args_2)
    models['Gxy'].to(device)
    optimizers['Gxy'] = Adam(models['Gxy'].parameters(), betas=(0.5, 0.999), eps=1e-8, lr=1e-4)
    if os.path.exists(f"{PATH}/last_{'Gxy'}.pth"):
        checkpoint = torch.load(f"{PATH}/last_{'Gxy'}.pth")
        models['Gxy'].load_state_dict(checkpoint['model_state_dict'])
        optimizers['Gxy'].load_state_dict(checkpoint['optimizer_state_dict'])
#######################################################################################
    args_3 = rcan_args(n_resblocks=20, scale=[2])
    models['Uyy'] = RCAN(args_3)
    models['Uyy'].to(device)
    optimizers['Uyy'] = Adam(models['Uyy'].parameters(), betas=(0.9, 0.999), eps=1e-8, lr=1e-4)
    if FREEZE_UPSCALE:
        # loading pretrain U (https://drive.google.com/file/d/10bEK-NxVtOS9-XSeyOZyaRmxUTX3iIRa/view?usp=sharing)
        checkpoint = torch.load(FREEZ_RCAN)
        models['Uyy'].load_state_dict(checkpoint) 
        for param in models['Uyy'].parameters():
            param.requires_grad = False
    elif os.path.exists(f"{PATH}/last_{'Uyy'}.pth"):
            checkpoint = torch.load(f"{PATH}/last_{'Uyy'}.pth")
            models['Uyy'].load_state_dict(checkpoint['model_state_dict'])
            optimizers['Uyy'].load_state_dict(checkpoint['optimizer_state_dict'])        
#######################################################################################
    # todo что-то тут не совпадает со статьёй. Написан, что n_layers=5,
    #  но тут челы говорят, что нужно меньше слоёв (работает только с n_layers=2)
    #  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/776
    models['Dx'] = NLayerDiscriminator(3, n_layers=2)
    models['Dx'].to(device)
    optimizers['Dx'] = Adam(models['Dx'].parameters(), betas=(0.5, 0.999), eps=1e-8, lr=1e-4)
    if os.path.exists(f"{PATH}/last_{'Dx'}.pth"):
        checkpoint = torch.load(f"{PATH}/last_{'Dx'}.pth")
        models['Dx'].load_state_dict(checkpoint['model_state_dict'])
        optimizers['Dx'].load_state_dict(checkpoint['optimizer_state_dict'])
#######################################################################################
    models['Dy'] = NLayerDiscriminator(3, n_layers=2)
    models['Dy'].to(device)
    optimizers['Dy'] = Adam(models['Dy'].parameters(), betas=(0.5, 0.999), eps=1e-8, lr=1e-4)
    if os.path.exists(f"{PATH}/last_{'Dy'}.pth"):
        checkpoint = torch.load(f"{PATH}/last_{'Dy'}.pth")
        models['Dy'].load_state_dict(checkpoint['model_state_dict'])
        optimizers['Dy'].load_state_dict(checkpoint['optimizer_state_dict'])
#######################################################################################
    models['Du'] = NLayerDiscriminator(3, n_layers=4)
    models['Du'].to(device)
    optimizers['Du'] = Adam(models['Du'].parameters(), betas=(0.5, 0.999), eps=1e-8, lr=1e-4)
    if os.path.exists(f"{PATH}/last_{'Du'}.pth"):
        checkpoint = torch.load(f"{PATH}/last_{'Du'}.pth")
        models['Du'].load_state_dict(checkpoint['model_state_dict'])
        optimizers['Du'].load_state_dict(checkpoint['optimizer_state_dict'])
        CUR_ITER = checkpoint['iter']
#######################################################################################

    # TRANSFORMERS
    lr_transform = A.Compose([
        A.RandomCrop(width=LR_PATCH, height=LR_PATCH),
        A.HorizontalFlip(),
        A.RandomRotate90(),
        A.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5), p=1)
    ])

    hr_transform = A.Compose([
        A.RandomCrop(width=HR_PATCH, height=HR_PATCH),
        A.HorizontalFlip(),
        A.RandomRotate90(),
        A.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5), p=1)
    ])

    lr_val_transform = A.Compose([
        A.RandomCrop(width=LR_VAL_PATCH, height=LR_VAL_PATCH),
        A.HorizontalFlip(),
        A.RandomRotate90(),
        A.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5), p=1)
    ])

    hr_val_transform = A.Compose([
        A.RandomCrop(width=HR_VAL_PATCH, height=HR_VAL_PATCH),
        A.HorizontalFlip(),
        A.RandomRotate90(),
        A.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5), p=1)
    ])

    # DATALOADERS
    val_dataset = LRandHR('/cache/chat_data/LR_valid/', '/cache/chat_data/DIV2K_train_HR/', lr_val_transform, hr_val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,)

    for low_res, high_res in val_dataloader:
        low_res = low_res.permute(0, 3, 1, 2).contiguous()
        fixed_lr = low_res.to(device)

        high_res = high_res.permute(0, 3, 1, 2).contiguous()
        fixed_hr = high_res.to(device)
        break

    train_dataset = LRandHR('/cache/chat_data/LR/', '/cache/chat_data/HR/', lr_transform, hr_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,)

    # TRAINING
    train(models, train_dataloader, optimizers, coef, MAX_ITER, (fixed_lr, fixed_hr), MODEL_PATH=PATH, logs=LOGS, device=device, FREEZE_UPSCALE=FREEZE_UPSCALE, cur_iter=CUR_ITER)
    