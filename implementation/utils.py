from typing import *
import cv2
import torch


def Downscale(img, target_dim: Tuple[int, int],
              kernel_size: Tuple[int, int] = (11, 11), scale_factor: float = 2.0):
    """
    Downscaling source/target y ∈ Y domain --> y↓ (LR img, same dimention as x)
    :param img:
    :param target_dim: dimension after bicubic downscaling
    :param scale_factor: constant that we choose at the beginning of the algorithm
    :param kernel_size: size of blurring kernel from Gauss distribution
    :return:
    """
    sigma = scale_factor / 2.0      # like in the article
    blurred = cv2.GaussianBlur(img, kernel_size, sigmaX=sigma, sigmaY=sigma)
    resized = cv2.resize(blurred, target_dim, interpolation=cv2.INTER_CUBIC)

    return resized
