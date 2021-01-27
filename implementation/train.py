import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

import cv2
from tqdm import tqdm

from data import LRandHR
from utils import Downscale
from RCAN_models import RCAN
from discriminators import NLayerDiscriminator
from args import rcan_args


# def geo_loss(generator, x):
#     simple_inference = generator(x)
#
#     arr = []
#     for pic in x:
#         arr.append(cv2.rotate(pic, cv2.cv2.ROTATE_90_CLOCKWISE))
#     arr = generator(arr)
#     res = []
#     for pic in arr:
#         res.append(cv2.rotate(pic, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE))
#     geo_mean_inference = torch.tensor(res)




def train(models, train_dataloader, optimizers, coef, max_iter, device="cuda:0", logs=''):
    for model in models.values():
        model.train()

    writer_lr = SummaryWriter(f"logs/" + logs + '/LR')
    writer_hr = SummaryWriter(f"logs/" + logs + '/HR')

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    fixed_lr = next(iter(train_dataloader))[0].to(device)

    tbord_step = 0

    cur_iter = 0
    while cur_iter < max_iter:
        for low_res, high_res in tqdm(train_dataloader, position=0):
            low_res, high_res = low_res.to(device), high_res.to(device)
            # todo fix sigma param
            clean_lr = Downscale(low_res, (32, 32), None)

            ### running all generators
            fake_x = models['Gyx'](clean_lr)
            #
            fake_y = models['Gxy'](low_res)
            pseudo_clean_lr = models['Gxy'](fake_x)
            #
            upscale_x = models['Uyy'](fake_y)
            upscale_y = models['Uyy'](pseudo_clean_lr)

            ### counting L1 losses
            L_cyc = coef['cyc'] * L1(pseudo_clean_lr, clean_lr)
            L_idt = coef['idt'] * L1(fake_y, clean_lr)
            # L_geo = coef['geo'] * torch.ones(len(L_idt))  # todo придумать как это нормально реализовать
            L_rec = L1(upscale_y, high_res)

            ### counting generator's loss
            disc_x_fake = MSE(models['Dx'](fake_x), torch.zeros(len(fake_x)))
            disc_y_fake = MSE(models['Dy'](fake_y), torch.zeros(len(fake_y)))
            disc_U_fake = MSE(models['Du'](upscale_y), torch.zeros(len(upscale_y)))
            generator_loss = L_cyc + L_idt + L_rec + \
                             disc_x_fake + disc_y_fake + coef['gamma'] * disc_U_fake # todo добавить L_geo

            ### backward on generator
            models['Gyx'].zero_grad()
            models['Gxy'].zero_grad()
            models['Uyy'].zero_grad()
            generator_loss.backward()
            optimizers['Gyx'].step()
            optimizers['Gxy'].step()
            optimizers['Uyy'].step()

            ### counting discriminator's loss
            disc_x_real = MSE(models['Dx'](low_res), torch.ones(len(low_res)))
            disc_y_real = MSE(models['Dy'](clean_lr), torch.ones(len(clean_lr)))
            disc_U_real = MSE(models['Du'](upscale_x), torch.ones(len(upscale_x)))
            discriminator_loss = disc_x_fake + disc_x_real + disc_y_fake + disc_y_real + \
                         coef['gamma'] * disc_U_fake + coef['gamma'] * disc_U_real

            ### backward on discriminator
            models['Dx'].zero_grad()
            models['Dy'].zero_grad()
            models['Du'].zero_grad()
            discriminator_loss.backward()
            optimizers['Dx'].step()
            optimizers['Dy'].step()
            optimizers['Du'].step()

            if cur_iter % 100 == 0:
                print(
                    f"Iter [{cur_iter}/{max_iter}] \
                              loss G: {generator_loss:.4f}, Loss D: {discriminator_loss:.4f}"
                )

                with torch.no_grad():
                    fake_y = models['Gxy'](fixed_lr)
                    upscale_x = models['Uyy'](fake_y)

                    img_grid_real = torchvision.utils.make_grid(
                        fixed_lr, normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        upscale_x, normalize=True
                    )

                    writer_lr.add_image("LR", img_grid_real, global_step=tbord_step)
                    writer_hr.add_image("HR", img_grid_fake, global_step=tbord_step)

                tbord_step += 1

            cur_iter += 1
            if cur_iter >= max_iter:
                break


if __name__ == '__main__':
    import albumentations as A

    # PARAMETERS
    BATCH_SIZE=16
    LR_PATCH=32
    HR_PATCH=64
    MAX_ITER=3e5
    coef = {'gamma': 0.1, 'cyc': 1, 'idt': 1, 'geo': 1}
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # MODELS
    models = {}
    args_1 = rcan_args()
    models['Gyx'] = RCAN(args_1)

    args_2 = rcan_args()
    models['Gxy'] = RCAN(args_2)

    args_3 = rcan_args(n_resblocks=20, res_scale=2)
    models['Uyy'] = RCAN(args_3)

    models['Dx'] = NLayerDiscriminator(3, n_layers=5)
    models['Dy'] = NLayerDiscriminator(3, n_layers=5)
    models['Du'] = NLayerDiscriminator(3, n_layers=5)

    # OPTIMIZERS
    optimizers = {}
    optimizers['Gyx'] = Adam(models['Gyx'].parameters(), betas=(0.5, 0.999), eps=1e-8, lr=1e-4)
    optimizers['Gxy'] = Adam(models['Gxy'].parameters(), betas=(0.5, 0.999), eps=1e-8, lr=1e-4)
    optimizers['Uyy'] = Adam(models['Uyy'].parameters(), betas=(0.9, 0.999), eps=1e-8, lr=1e-4)

    optimizers['Dx'] = Adam(models['Dx'].parameters(), betas=(0.5, 0.999), eps=1e-8, lr=1e-4)
    optimizers['Dy'] = Adam(models['Dy'].parameters(), betas=(0.5, 0.999), eps=1e-8, lr=1e-4)
    optimizers['Du'] = Adam(models['Du'].parameters(), betas=(0.5, 0.999), eps=1e-8, lr=1e-4)

    # DATALOADER
    lr_transform = A.Compose([
        A.RandomCrop(width=LR_PATCH, height=LR_PATCH),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=90, interpolation=1, border_mode=4, p=0.5)
    ])

    hr_transform = A.Compose([
        A.RandomCrop(width=HR_PATCH, height=HR_PATCH),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=90, interpolation=1, border_mode=4, p=0.5)
    ])
    dataset = LRandHR('../DATA/LR_valid/', '../DATA/DIV2K_train_HR/', lr_transform, hr_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    train(models, dataloader, optimizers, coef, MAX_ITER, device="cuda:0")
