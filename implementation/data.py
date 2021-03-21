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
