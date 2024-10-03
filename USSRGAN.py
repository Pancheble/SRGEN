# Standard libraries #
import random
import shutil
import time
import math
import os
from os import listdir
from os.path import join
from math import log10

# External libraries #
import torch
import torch.optim as optim
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.vgg import vgg16
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from tqdm import tqdm



# Hyperparameter #
crop_size = 96

if (crop_size % 4) != 0:
  crop_size = crop_size - (crop_size%4)

upscale_factor = 4
batch_size = 64
epochs = 250

dataset_dir = r'C:\SR\SRGAN\data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Train_Images'
val_dataset_dir = r'C:\SR\SRGAN\data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Val_Images'

save_path = r'C:\SR\SRGAN\result'