class rcan_args:
    def __init__(self, n_resgroups=5, n_resblocks=10, scale=[1], n_feats=64,
                 reduction=16, rgb_range=255, in_colors=3, out_colors=3, res_scale=1):
        # n_resgroups  n_resblocks    n_feats  reduction  scale   rgb_range   n_colors   res_scale

        self.n_resgroups = n_resgroups
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.reduction = reduction
        self.scale = scale
        self.rgb_range = rgb_range
        self.in_colors = in_colors
        self.out_colors = out_colors
        self.res_scale = res_scale
      
    
import torch
from PIL import Image
import os
import cv2


class LRandHR(torch.utils.data.Dataset):
    def __init__(self, lr_dir, hr_dir, lr_transform, hr_transform):
        self.lr_dir = lr_dir  # example: 'data/LR_train/'
        self.hr_dir = hr_dir

        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.lr_pathes = []  # pathes for all LR images
        self.hr_pathes = []  # pathes for all HR images

        for dirname, dirs, files in os.walk(self.lr_dir):
            for filename in files:
                self.lr_pathes.append(filename)

        for dirname, dirs, files in os.walk(self.hr_dir):
            for filename in files:
                self.hr_pathes.append(filename)

    def __getitem__(self, idx):
        lr_img_name = self.lr_pathes[idx % len(self.lr_pathes)]
        hr_img_name = self.hr_pathes[idx % len(self.hr_pathes)]

        # cv2 case
        lr_img = cv2.imread(self.lr_dir + lr_img_name)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)

        hr_img = cv2.imread(self.hr_dir + hr_img_name)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

        # PIL case
        # lr_img = Image.open(lr_img)
        # hr_img = Image.open(hr_img)

        if self.lr_transform is not None:
            lr_img = self.lr_transform(image=lr_img)["image"]
        if self.hr_transform is not None:
            hr_img = self.hr_transform(image=hr_img)["image"]

        return lr_img, hr_img

    def __len__(self):
        return max(len(self.lr_pathes), len(self.hr_pathes))


if __name__ == '__main__':
    import albumentations as A
    # import matplotlib.pyplot as plt

    lr_transform = A.Compose([
        A.RandomCrop(width=32, height=32),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=90, interpolation=1, border_mode=4, p=0.5)
    ])

    hr_transform = A.Compose([
        A.RandomCrop(width=64, height=64),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=90, interpolation=1, border_mode=4, p=0.5)
    ])

    data = LRandHR('/cache/chat_data/LR_valid/', '/cache/chat_data/DIV2K_train_HR/', lr_transform, hr_transform)

    lr = data[0][0] # LR image
    hr = data[0][1] # HR image

    assert lr.shape == (32, 32, 3)
    assert hr.shape == (64, 64, 3)
    assert len(data) > 0

    # plt.imshow(lr)

"""
source from https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py
Copyright https://github.com/yulunzhang/RCAN
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

        
"""
source from https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py
Copyright https://github.com/yulunzhang/RCAN
"""

import torch.nn as nn


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(RCAN, self).__init__()
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        # define head module
        modules_head = [conv(args.in_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.out_colors, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res = res + x

        x = self.tail(res)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

                
def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = torch.tensor((image - from_min) / float(from_range), dtype=torch.float)
    return to_min + (scaled * to_range)


def batch_inferense(batch, cleaner, upscaler):
    return upscaler(cleaner(batch))



import matplotlib.pyplot as plt

def draw_one_pic(batch, number, tag=121):
    pic = batch[number].permute(1, 2, 0).contiguous()
    pic = interval_mapping(pic, -1, 1, 0, 1)
    pic = pic.to('cpu')
    plt.subplot(tag), plt.imshow(pic)

from kornia.filters import GaussianBlur2d

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

from torch.utils.data import DataLoader


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE=16
LR_VAL_PATCH=32
HR_VAL_PATCH=64

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
exp_name = 'default_1'
number = 55

PATH = f'{path}{exp_name}/models'

args_2 = rcan_args()
cleaner = RCAN(args_2)
cleaner.to(device)
checkpoint = torch.load(f"{PATH}/{number}_{'Gxy'}.pth")
cleaner.load_state_dict(checkpoint['model_state_dict'])
cleaner.eval()

for param in cleaner.parameters():
    param.requires_grad = False

args_3 = rcan_args(n_resblocks=20, scale=[2])
upscaler = RCAN(args_3)
upscaler.to(device)
checkpoint = torch.load(f"{PATH}/{number}_{'Uyy'}.pth")
upscaler.load_state_dict(checkpoint['model_state_dict'])
upscaler.eval()

for param in upscaler.parameters():
    param.requires_grad = False

downscaled = get_downscale(high_res, LR_VAL_PATCH)
upscaled = batch_inferense(downscaled, cleaner, upscaler)


plt.figure(figsize=(15, 10))

num = 4

draw_one_pic(high_res, num, 221)
plt.title('HR')
draw_one_pic(downscaled, num, 222)
plt.title('Downscaled')
draw_one_pic(upscaled, num, 223)
plt.title('Upscaled')


