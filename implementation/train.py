import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

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


def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = torch.tensor((image - from_min) / float(from_range), dtype=torch.float)
    return to_min + (scaled * to_range)


def train(models, train_dataloader, optimizers, coef, max_iter, fixed_lr,
          device=torch.device("cuda:0"), logs='', PATH='MODELS'):
    for model in models.values():
        model.train()

    writer = SummaryWriter(f"logs/" + logs)
    # writer_hr = SummaryWriter(f"logs/" + logs + '/HR')

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    tbord_step = 0
    cur_iter = 0
    while cur_iter < max_iter:
        for low_res, high_res in tqdm(train_dataloader, position=0):



            # hr = high_res.numpy()
            # clean_lr = []
            # for img in hr:
            #     clean_lr.append(Downscale(img, (32, 32)))
            #
            # clean_lr = torch.tensor(clean_lr)

            clean_lr = clean_lr.permute(0, 3, 1, 2).contiguous()
            low_res = low_res.permute(0, 3, 1, 2).contiguous()
            high_res = high_res.permute(0, 3, 1, 2).contiguous()
            clean_lr = clean_lr.to(device)
            low_res, high_res = low_res.to(device), high_res.to(device)

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
            L_geo = coef['geo'] * torch.ones_like(L_idt)  # todo придумать как это нормально реализовать
            L_rec = L1(upscale_y, high_res)
            
            for param in models['Dx'].parameters():
                param.requires_grad = False
            for param in models['Dy'].parameters():
                param.requires_grad = False
            for param in models['Du'].parameters():
                param.requires_grad = False

            ### counting generator's loss
            disc_x_fake = models['Dx'](fake_x)
            disc_x_fake_loss = MSE(disc_x_fake, torch.zeros_like(disc_x_fake))

            disc_y_fake = models['Dy'](fake_y)
            disc_y_fake_loss = MSE(disc_y_fake, torch.zeros_like(disc_y_fake))

            disc_U_fake = models['Du'](upscale_y)
            disc_U_fake_loss = MSE(disc_U_fake, torch.zeros_like(disc_U_fake))

            generator_loss = L_cyc + L_idt + L_rec + \
                             disc_x_fake_loss + disc_y_fake_loss + coef['gamma'] * disc_U_fake_loss # todo добавить L_geo

            ### backward on generator
            models['Gyx'].zero_grad()
            models['Gxy'].zero_grad()
            models['Uyy'].zero_grad()
            generator_loss.backward()
            optimizers['Gyx'].step()
            optimizers['Gxy'].step()
            optimizers['Uyy'].step()
            
            for param in models['Dx'].parameters():
                param.requires_grad = True
            for param in models['Dy'].parameters():
                param.requires_grad = True
            for param in models['Du'].parameters():
                param.requires_grad = True

            ### counting discriminator's loss
            # on real
            disc_x_real = models['Dx'](low_res)
            disc_x_real_loss = MSE(disc_x_real, torch.ones_like(disc_x_real))

            disc_y_real = models['Dy'](clean_lr)
            disc_y_real_loss = MSE(disc_y_real, torch.ones_like(disc_y_real))

            disc_U_real = models['Du'](upscale_x.detach())
            disc_U_real_loss = MSE(disc_U_real, torch.ones_like(disc_U_real))

            # on fake
            disc_x_fake = models['Dx'](fake_x.detach())
            disc_x_fake_loss = MSE(disc_x_fake, torch.zeros_like(disc_x_fake))

            disc_y_fake = models['Dy'](fake_y.detach())
            disc_y_fake_loss = MSE(disc_y_fake, torch.zeros_like(disc_y_fake))

            disc_U_fake = models['Du'](upscale_y.detach())
            disc_U_fake_loss = MSE(disc_U_fake, torch.zeros_like(disc_U_fake))

            discriminator_loss = disc_x_real_loss + disc_x_fake_loss + disc_y_real_loss + disc_y_fake_loss + \
                                 coef['gamma'] * disc_U_real_loss + coef['gamma'] * disc_U_fake_loss

            ### backward on discriminator
            models['Dx'].zero_grad()
            models['Dy'].zero_grad()
            models['Du'].zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                discriminator_loss.backward()       # добавление лосса даёт ошибку вот тут
            optimizers['Dx'].step()
            optimizers['Dy'].step()
            optimizers['Du'].step()

            ### saving models
            if cur_iter % 10 ** 3 == 0:
                for model in models.items():
                    torch.save({
                                'iter' : cur_iter,
                                'model_state_dict' : model[1].state_dict(),
                                'optimizer_state_dict': optimizers[model[0]].state_dict()
                                }, f"{PATH}/{cur_iter // 10 ** 3}_{model[0]}.pth")
                    torch.save({
                                'iter': cur_iter,
                                'model_state_dict': model[1].state_dict(),
                                'optimizer_state_dict': optimizers[model[0]].state_dict()
                                }, f"{PATH}/last_{model[0]}.pth")

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

                    writer.add_image("LR", interval_mapping(img_grid_real, 0.0, 1.0, 0.0, 255.0), global_step=tbord_step)
                    writer.add_image("HR", interval_mapping(img_grid_fake, 0.0, 1.0, 0.0, 255.0), global_step=tbord_step)
                    writer.add_scalar('Generator loss', generator_loss, global_step=tbord_step)
                    writer.add_scalar('Discriminator loss', discriminator_loss, global_step=tbord_step)

                    writer.add_scalar('Cycle loss', L_cyc, global_step=tbord_step)
                    writer.add_scalar('Identity loss', L_idt, global_step=tbord_step)
                    writer.add_scalar('Geometric loss', L_geo, global_step=tbord_step)
                    writer.add_scalar('Reconstruct loss', L_rec, global_step=tbord_step)

                tbord_step += 1

            cur_iter += 1
            if cur_iter >= max_iter:
                break


if __name__ == '__main__':
    import albumentations as A
    import os

    # PARAMETERS
    BATCH_SIZE=16
    LR_PATCH=32
    LR_VAL_PATCH=256
    HR_PATCH=64
    MAX_ITER=3e5
    coef = {'gamma': 0.1, 'cyc': 1, 'idt': 1, 'geo': 1}
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    PATH='MODELS'

    # MODELS
    models = {}
    optimizers = {}
    if os.path.exists(f"{PATH}/last_{'Gyx'}.pth"):
        checkpoint = torch.load(f"{PATH}/last_{'Gyx'}.pth")
        models['Gyx'] = checkpoint['model_state_dict']
        optimizers['Gyx'] = checkpoint['optimizer_state_dict']
    else:
        args_1 = rcan_args()
        models['Gyx'] = RCAN(args_1)
        optimizers['Gyx'] = Adam(models['Gyx'].parameters(), betas=(0.5, 0.999), eps=1e-8, lr=1e-4)

    if os.path.exists(f"{PATH}/last_{'Gxy'}.pth"):
        checkpoint = torch.load(f"{PATH}/last_{'Gxy'}.pth")
        models['Gxy'] = checkpoint['model_state_dict']
        optimizers['Gxy'] = checkpoint['optimizer_state_dict']
    else:
        args_2 = rcan_args()
        models['Gxy'] = RCAN(args_2)
        optimizers['Gxy'] = Adam(models['Gxy'].parameters(), betas=(0.5, 0.999), eps=1e-8, lr=1e-4)

    if os.path.exists(f"{PATH}/last_{'Uyy'}.pth"):
        checkpoint = torch.load(f"{PATH}/last_{'Uyy'}.pth")
        models['Uyy'] = checkpoint['model_state_dict']
        optimizers['Uyy'] = checkpoint['optimizer_state_dict']
    else:
        args_3 = rcan_args(n_resblocks=20, scale=[2])
        models['Uyy'] = RCAN(args_3)
        optimizers['Uyy'] = Adam(models['Uyy'].parameters(), betas=(0.9, 0.999), eps=1e-8, lr=1e-4)

    # todo что-то тут не совпадает со статьёй. Написан, что n_layers=5,
    #  но тут челы говорят, что нужно меньше слоёв (работает только с n_layers=2)
    #  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/776
    if os.path.exists(f"{PATH}/last_{'Dx'}.pth"):
        checkpoint = torch.load(f"{PATH}/last_{'Dx'}.pth")
        models['Dx'] = checkpoint['model_state_dict']
        optimizers['Dx'] = checkpoint['optimizer_state_dict']
    else:
        models['Dx'] = NLayerDiscriminator(3, n_layers=2)
        optimizers['Dx'] = Adam(models['Dx'].parameters(), betas=(0.5, 0.999), eps=1e-8, lr=1e-4)

    if os.path.exists(f"{PATH}/last_{'Dy'}.pth"):
        checkpoint = torch.load(f"{PATH}/last_{'Dy'}.pth")
        models['Dy'] = checkpoint['model_state_dict']
        optimizers['Dy'] = checkpoint['optimizer_state_dict']
    else:
        models['Dy'] = NLayerDiscriminator(3, n_layers=2)
        optimizers['Dy'] = Adam(models['Dy'].parameters(), betas=(0.5, 0.999), eps=1e-8, lr=1e-4)

    if os.path.exists(f"{PATH}/last_{'Du'}.pth"):
        checkpoint = torch.load(f"{PATH}/last_{'Du'}.pth")
        models['Du'] = checkpoint['model_state_dict']
        optimizers['Du'] = checkpoint['optimizer_state_dict']
    else:
        models['Du'] = NLayerDiscriminator(3, n_layers=4)
        optimizers['Du'] = Adam(models['Du'].parameters(), betas=(0.5, 0.999), eps=1e-8, lr=1e-4)

    for name in models:
        models[name] = models[name].to(device)

    # TRANSFORMERS
    lr_transform = A.Compose([
        A.RandomCrop(width=LR_PATCH, height=LR_PATCH),
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

    hr_transform = A.Compose([
        A.RandomCrop(width=HR_PATCH, height=HR_PATCH),
        A.HorizontalFlip(),
        A.RandomRotate90(),
        A.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5), p=1)
    ])

    # DATALOADERS
    val_dataset = LRandHR('../DATA/LR_train/', '../DATA/DIV2K_train_HR/', lr_val_transform, hr_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,)

    for low_res, _ in val_dataloader:
        low_res = low_res.permute(0, 3, 1, 2).contiguous()
        fixed_lr = low_res.to(device)
        break

    train_dataset = LRandHR('../DATA/LR_train/', '../DATA/DIV2K_train_HR/', lr_transform, hr_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,)

    # TRAINING
    train(models, train_dataloader, optimizers, coef, MAX_ITER, fixed_lr, device=device, PATH=PATH)
