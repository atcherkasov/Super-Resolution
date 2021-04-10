import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from kornia.filters import GaussianBlur2d
from tqdm import tqdm
import os
import sys
import random
random.seed(228)

from data import LRandHR
from utils import Downscale
from RCAN_models import RCAN
from discriminators import NLayerDiscriminator
from args import rcan_args


ROTATE = [-1, 0, 1, 2]
FLIP = [0, 1]


def geo_loss(simple_inference, generator, batch, rot, flip, loss):
    rotate_batch = torch.rot90(batch, k=rot, dims=(2, 3))
    if flip:
        rotate_batch = torch.flip(rotate_batch, [-2])
    rotate_inference = generator(rotate_batch)
    if flip:
        rotate_inference = torch.flip(rotate_inference, [-2])
    rotate_inference = torch.rot90(rotate_inference, k=-rot, dims=(2, 3))

    return loss(simple_inference, rotate_inference)


def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = torch.tensor((image - from_min) / float(from_range), dtype=torch.float)
    return to_min + (scaled * to_range)


def gen_experiment_num(exp_path, exp_name):
    '''
    will found all experiments with same name as <exp_name>
    and return <exp_name>_<biggesr number + 1>
    '''
    exp_name = exp_name.split('_')[0]
    experiments = os.listdir(exp_path)
    exp_num = -1
    for exp in experiments:
        if exp.split('_')[0] == exp_name:
            exp_num = max(exp_num, int(exp.split('_')[1]))
    return str(exp_num + 1)


def setting_dirs(argv):
    # SETTING DIRS
    glob_path = '/cache/chat/experiments/'
    exp_name = argv[0][:-3].split('/')[-1]
    if len(argv) > 1:
        # so we need to load old experiment
        number = int(sys.argv[1])
        glob_path += f'/{exp_name}_{number}'
    else:
        # so we will create brand new experiment
        exp_name = f'{exp_name}_{gen_experiment_num(glob_path, exp_name)}'
        config_name = argv[0].split('/')[-1]
        
        os.mkdir(f'{glob_path}{exp_name}')
        os.mkdir(f'{glob_path}{exp_name}/runs')
        os.mkdir(f'{glob_path}{exp_name}/models')
        glob_path += exp_name
        
        fout = open(f'{glob_path}/{config_name}', 'w')
        with open(argv[0]) as f:
            content = f.readlines()
            for line in content:
                print(line, file=fout)
            fout.close()
    PATH=f'{glob_path}/models'
    LOGS=f'{glob_path}/runs'
    
    return PATH, LOGS
  

def train(models, train_dataloader, optimizers, coef, max_iter, fixed_batch, 
          MODEL_PATH, logs, stochastic=None, device=torch.device("cuda:0"),
          FREEZE_UPSCALE=False, cur_iter=0):
    for model in models.values():
        model.train()
    
    writer = SummaryWriter(logs)
    
    print('\n\tstart training!')
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()
    gauss = GaussianBlur2d((11, 11), (1, 1))

    fixed_lr, fixed_hr = fixed_batch
    tbord_step = 0
    cur_iter = cur_iter
    while cur_iter < max_iter:
        for low_res, high_res in tqdm(train_dataloader, position=0):
            clean_lr = high_res.permute(0, 3, 1, 2).contiguous()
            blured_clean_lr = gauss(clean_lr)
            clean_lr = F.interpolate(blured_clean_lr, size=(32, 32), mode='nearest')
            if stochastic:
                if stochastic['case'] == 1:
                    # add Gausse noise as 4th channel
                    noise = torch.rand(stochastic['BATCH_SIZE'], 
                                       1, 
                                       stochastic['LR_PATCH'],   
                                       stochastic['LR_PATCH'])
                    noise.to(device)
                    clean_lr = torch.cat((clean_lr, noise), 1)
                if stochastic['case'] == 2:
                    # add Gausse noise as a filter in the middle of Gyx model
                    pass

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
            if stochastic['case'] == 1:
                L_cyc = coef['cyc'] * L1(pseudo_clean_lr, clean_lr[:, :3, :, :])
                L_idt = coef['idt'] * L1(models['Gxy'](clean_lr[:, :3, :, :]), clean_lr[:, :3, :, :])
            else:
                L_cyc = coef['cyc'] * L1(pseudo_clean_lr, clean_lr)
                L_idt = coef['idt'] * L1(models['Gxy'](clean_lr), clean_lr)
            L_geo = coef['geo'] * geo_loss(fake_y, models['Gxy'], low_res,
                                           random.choice(ROTATE), random.choice(FLIP), L1)
            L_rec = L1(upscale_y, high_res)
            
            for param in models['Dx'].parameters():
                param.requires_grad = False
            for param in models['Dy'].parameters():
                param.requires_grad = False
            for param in models['Du'].parameters():
                param.requires_grad = False

            ### counting generator's loss
            disc_x_fake = models['Dx'](fake_x)
            disc_x_fake_loss = MSE(disc_x_fake, torch.ones_like(disc_x_fake))

            disc_y_fake = models['Dy'](fake_y)
            disc_y_fake_loss = MSE(disc_y_fake, torch.ones_like(disc_y_fake))

            disc_U_fake = models['Du'](upscale_y)
            disc_U_fake_loss = MSE(disc_U_fake, torch.ones_like(disc_U_fake))

            disc_U_real = models['Du'](upscale_x)
            disc_U_real_loss = MSE(disc_U_real, torch.zeros_like(disc_U_real))

            generator_loss = L_cyc + L_idt + L_rec + L_geo + \
                             disc_x_fake_loss + disc_y_fake_loss + \
                             coef['gamma'] * disc_U_fake_loss + coef['gamma'] * disc_U_real_loss

            ### backward on generator
            models['Gyx'].zero_grad()
            models['Gxy'].zero_grad()
            models['Uyy'].zero_grad()
            generator_loss.backward()
            optimizers['Gyx'].step()
            optimizers['Gxy'].step()
            if not FREEZE_UPSCALE:
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

            disc_y_real = models['Dy'](clean_lr[:, :3, :, :])
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
                                 disc_U_real_loss + disc_U_fake_loss

            ### backward on discriminator
            models['Dx'].zero_grad()
            models['Dy'].zero_grad()
            models['Du'].zero_grad()
            discriminator_loss.backward()
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
                                }, f"{MODEL_PATH}/{cur_iter // 10 ** 3}_{model[0]}.pth")
                    torch.save({
                                'iter': cur_iter,
                                'model_state_dict': model[1].state_dict(),
                                'optimizer_state_dict': optimizers[model[0]].state_dict()
                                }, f"{MODEL_PATH}/last_{model[0]}.pth")

            if cur_iter % 100 == 0:
                print(
                    f"Iter [{cur_iter}/{max_iter}] \
                              loss G: {generator_loss:.4f}, Loss D: {discriminator_loss:.4f}"
                )

                with torch.no_grad():
                    ### Actual Flow

                    fake_y = models['Gxy'](fixed_lr)
                    upscale_x = models['Uyy'](fake_y)

                    grid_fixed_lr = torchvision.utils.make_grid(
                        fixed_lr, normalize=False
                    )
                    writer.add_image("LR source", interval_mapping(grid_fixed_lr, -1.0, 1.0, 0.0, 1.0), global_step=tbord_step)

                    grid_fake_y = torchvision.utils.make_grid(
                        fake_y, normalize=False
                    )
                    writer.add_image("Fake y", interval_mapping(grid_fake_y, -1.0, 1.0, 0.0, 1.0), global_step=tbord_step)

                    grid_upscale_x = torchvision.utils.make_grid(
                        upscale_x, normalize=False
                    )
                    writer.add_image("Upscale x (LR source)", interval_mapping(grid_upscale_x, -1.0, 1.0, 0.0, 1.0), global_step=tbord_step)

                    ### Pseudo Flow
                    blured_clean_lr = gauss(fixed_hr)
                    clean_lr = F.interpolate(blured_clean_lr, size=(128, 128), mode='nearest')
                    noise = torch.rand(stochastic['BATCH_SIZE'], 
                                       1, 
                                       128,   
                                       128)
                    noise = noise.to(device)
                    clean_lr = torch.cat((clean_lr, noise), 1)
                    
                    fake_x = models['Gyx'](clean_lr)
                    pseudo_clean_lr = models['Gxy'](fake_x)
                    upscale_y = models['Uyy'](pseudo_clean_lr)

                    grid_fixed_hr = torchvision.utils.make_grid(
                        fixed_hr, normalize=False
                    )
                    writer.add_image("HR source", interval_mapping(grid_fixed_hr, -1.0, 1.0, 0.0, 1.0),
                                     global_step=tbord_step)

                    grid_clean_lr = torchvision.utils.make_grid(
                        clean_lr, normalize=False
                    )
                    writer.add_image("Clean LR", interval_mapping(grid_clean_lr, -1.0, 1.0, 0.0, 1.0),
                                     global_step=tbord_step)

                    grid_fake_x = torchvision.utils.make_grid(
                        fake_x, normalize=False
                    )
                    writer.add_image("Fake x", interval_mapping(grid_fake_x, -1.0, 1.0, 0.0, 1.0),
                                     global_step=tbord_step)

                    grid_pseudo_clean_lr = torchvision.utils.make_grid(
                        pseudo_clean_lr, normalize=False
                    )
                    writer.add_image("Pseudo-clean LR", interval_mapping(grid_pseudo_clean_lr, -1.0, 1.0, 0.0, 1.0),
                                     global_step=tbord_step)

                    grid_upscale_y = torchvision.utils.make_grid(
                        upscale_y, normalize=False
                    )
                    writer.add_image("Upscale y (HR source) ", interval_mapping(grid_upscale_y, -1.0, 1.0, 0.0, 1.0),
                                     global_step=tbord_step)

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
    import sys
    
    
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
    coef = {'gamma': 0.1, 'cyc': 1, 'idt': 1, 'geo': 1}
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

    # take first batch from val_dataloader
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
    