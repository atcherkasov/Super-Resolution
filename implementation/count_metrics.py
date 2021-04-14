import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from kornia.filters import GaussianBlur2d
import numpy as np
import skimage
import cv2
import albumentations as A
from tqdm import tqdm
import random
random.seed(228)
import warnings
warnings.filterwarnings('ignore')

from data import LRandHR
from RCAN_models import RCAN
from args import rcan_args
from train import interval_mapping


def get_downscale(batch, size):
    gauss = GaussianBlur2d((11, 11), (1, 1))
    blured_clean_lr = gauss(batch)
    clean_lr = F.interpolate(blured_clean_lr, size=(size, size), mode='nearest')
    return clean_lr


def batch_inferense(batch, cleaner, upscaler):
    with torch.no_grad():
        cleaned = cleaner(batch)
        cleaned = interval_mapping(cleaned, torch.min(cleaned), torch.max(cleaned), -1, 1)
        upscaled = upscaler(cleaned)
        return interval_mapping(upscaled, torch.min(upscaled), torch.max(upscaled), -1, 1)

def PSNR(batch_true, batch_test, data_range=2):
    mse = torch.mean((batch_true - batch_test) ** 2, dim=[1, 2, 3])    
    return 20 * torch.log10(data_range / (mse ** 0.5))



def dssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    if img1.is_cuda != img2.is_cuda:
        raise ValueError('Images dtype is not consistent')

    filter_size = max(1, filter_size)

    kernel = np.arange(0, filter_size, dtype=np.float32)
    kernel -= (filter_size - 1 ) / 2.0
    kernel = kernel**2
    kernel *= ( -0.5 / (filter_sigma**2) )
    kernel = np.reshape (kernel, (1,-1)) + np.reshape(kernel, (-1,1) )
    kernel = torch.from_numpy(kernel).view(1, -1)
    kernel = F.softmax(kernel, dim=1).view(1, 1, filter_size, filter_size).repeat(3, 1, 1, 1)
    if img1.is_cuda:
        kernel = kernel.cuda()

    def reducer(x):
        return F.conv2d(x, kernel, groups=3)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mean0 = reducer(img1)
    mean1 = reducer(img2)
    num0 = mean0 * mean1 * 2.0
    den0 = torch.square(mean0) + torch.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    num1 = reducer(img1 * img2) * 2.0
    den1 = reducer(torch.square(img1) + torch.square(img2))
    c2 *= 1.0
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    lcs = luminance * cs
    ssim_val = lcs.view(lcs.shape[0], lcs.shape[1], -1).mean(2)
    dssim = (1.0 - ssim_val ) / 2.0

    return dssim


if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    BATCH_SIZE=64
    LR_VAL_PATCH=200
    HR_VAL_PATCH=800

    lr_val_transform = A.Compose([
            A.RandomCrop(width=LR_VAL_PATCH, height=LR_VAL_PATCH),
            A.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5), p=1)
        ])

    hr_val_transform = A.Compose([
        A.RandomCrop(width=HR_VAL_PATCH, height=HR_VAL_PATCH),
        A.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5), p=1)
    ])

    val_dataset = LRandHR('/cache/chat_data/LR_valid/', '/cache/chat_data/DIV2K_train_HR/', 
                          lr_val_transform, hr_val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,)

    for low_res, high_res in val_dataloader:
            low_res = low_res.permute(0, 3, 1, 2).contiguous()
            low_res = low_res.to(device)
            low_res.requires_grad = False
            high_res = high_res.permute(0, 3, 1, 2).contiguous()
            high_res = high_res.to(device)
            high_res.requires_grad = False
            break
            
            
    path = '/cache/chat/experiments/'
#     exp_name = 'default_1'
    number = 12

#     all_exps = ['default_1', 'no_cycle_0', 'no_geometric_0', 
#                'no_identity_0', 'stochasticChannel_7', 'freezing_0']
# x4default_2  x4no_cyc_0  x4no_geo_0  x4noIdentity_0  x4stochasticChannel_3
    all_exps = ['x4default_2']
    for exp_name in all_exps:
        print()
        print(exp_name)
        PATH = f'{path}{exp_name}/models'

        args_2 = rcan_args()
        cleaner = RCAN(args_2)
        for param in cleaner.parameters():
            param.requires_grad = False
        cleaner.to(device)
        checkpoint = torch.load(f"{PATH}/{number}_{'Gxy'}.pth")
        cleaner.load_state_dict(checkpoint['model_state_dict'])
        cleaner.eval()


        args_3 = rcan_args(n_resblocks=20, scale=[4])
        upscaler = RCAN(args_3)
        for param in upscaler.parameters():
            param.requires_grad = False
        upscaler.to(device)
        checkpoint = torch.load(f"{PATH}/{number}_{'Uyy'}.pth")
        upscaler.load_state_dict(checkpoint['model_state_dict'])
        upscaler.eval()

        downscaled = get_downscale(high_res, LR_VAL_PATCH)
        upscaled = batch_inferense(downscaled, cleaner, upscaler)

        print(f'PSNR: {torch.mean(PSNR(high_res, upscaled)).item()}')
    #     res = []
    #     for number in range(upscaled.shape[0]):
    #         test = upscaled[number]
    #         test = test.permute(1, 2, 0).contiguous()
    #         test = np.array(test.to('cpu'))

    #         gr_truth = high_res[number]
    #         gr_truth = gr_truth.permute(1, 2, 0).contiguous()
    #         gr_truth = np.array(gr_truth.to('cpu'))

    #         res.append(calculate_ssim(gr_truth, test))
    #     print(dssim(high_res, upscaled, 1).shape)
        upscaled = interval_mapping(upscaled, torch.min(upscaled), torch.max(upscaled), 0, 1)
        high_res = interval_mapping(high_res, torch.min(high_res), torch.max(high_res), 0, 1)

        print(f'SSIM: {torch.mean(dssim(high_res, upscaled, 1))}')
