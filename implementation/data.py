import torch
from sklearn.utils import shuffle
from PIL import Image
import os
from os.path import join
import numpy as np

RANDOM_SEED = 228

class LRandHR(torch.utils.data.Dataset):
    def __init__(self, lr_dir, hr_dir, transform):
        self.lr_dir = lr_dir  # example: 'data/LR_train'
        self.hr_dir = hr_dir
        self.transform = transform

        self.lr_pathes = []  # pathes for all LR images
        self.hr_pathes = []  # pathes for all HR images

        for dirname, dirs, files in os.walk(self.lr_dir):
            for filename in files:
                self.lr_pathes.append(filename)

        for dirname, dirs, files in os.walk(self.hr_dir):
            for filename in files:
                self.hr_pathes.append(filename)

        self.lr_pathes = shuffle(self.lr_pathes,
                                 random_state=RANDOM_SEED)
        self.hr_pathes = shuffle(self.hr_pathes,
                                 random_state=RANDOM_SEED)

    def __getitem__(self, idx):
        lr_img = self.lr_pathes[idx % len(self.lr_pathes)]
        hr_img = self.hr_pathes[idx % len(self.hr_pathes)]

        lr_img = cv2.imread(lr_img)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.imread(hr_img)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

        # lr_img = Image.open(lr_img)
        # hr_img = Image.open(hr_img)


        # отрисовка картиноки до аугментации
        # fig=plt.figure(figsize=(8, 8))
        # fig.add_subplot(1, 3, 1)
        # plt.imshow(np.asarray(image))
        # image = Downscale(image, (64, 64), (101, 101))


        if self.transform is not None:
            lr_img = self.transform(lr_img)["image"]
            hr_img = self.transform(hr_img)["image"]


        # отрисовка картиноки после аугментации
        #         fig.add_subplot(1, 3, 2)
        #         plt.imshow(image)

        return lr_img, hr_img

    def __len__(self):
        return max(len(self.lr_pathes), len(self.hr_pathes))


if __name__ == '__main__':
    import albumentations as A
    import cv2



    transform = A.Compose([
        A.RandomCrop(width=64, height=64),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    data = LRandHR('../DATA/LR_valid', '../DATA/DIV2K_train_HR', None)