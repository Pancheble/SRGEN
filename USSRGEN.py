import os
import random
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16
from tqdm import tqdm
import numpy as np
from PIL import Image

# 하이퍼파라미터
crop_size = 96
upscale_factor = 4
epochs = 250
batch_size = 32  # 배치 크기를 줄였습니다.
dataset_dir = r'C:\SR\SRGEN\data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
val_dataset_dir = r'C:\SR\SRGEN\data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationObject'
save_path = r'C:\SR\SRGEN\result'

# 모델 저장 경로가 존재하지 않으면 생성
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 데이터셋 클래스 정의
class TrainDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDataset, self).__init__()
        self.image_filenames = []
        for x in os.listdir(dataset_dir):
            if self.is_image_file(x):
                img_path = os.path.join(dataset_dir, x)
                img = Image.open(img_path)
                if img.size[0] >= crop_size and img.size[1] >= crop_size:
                    self.image_filenames.append(img_path)

        self.hr_transform = transforms.Compose([
            
    
            transforms.RandomCrop(crop_size),
            transforms.ToTensor()  # HR 이미지를 텐서로 변환
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
            transforms.ToTensor()  # LR 이미지를 텐서로 변환
        ])

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index]).convert('RGB')  # HR 이미지를 RGB로 변환
        hr_image = self.hr_transform(hr_image)  # HR 이미지를 텐서로 변환

        lr_image = hr_image.permute(1, 2, 0).numpy()  # Tensor를 NumPy 배열로 변환
        lr_image = Image.fromarray((lr_image * 255).astype(np.uint8))  # NumPy 배열을 PIL 이미지로 변환
        lr_image = self.lr_transform(lr_image)  # LR 이미지 변환 적용
        
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class ValDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(ValDataset, self).__init__()
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if self.is_image_file(x)]
        self.upscale_factor = upscale_factor

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        hr_image = transforms.CenterCrop(crop_size)(hr_image)
        lr_image = transforms.Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)(hr_image)
        bicubic_hr_image = transforms.Resize(crop_size, interpolation=Image.BICUBIC)(lr_image)
        return transforms.ToTensor()(lr_image), transforms.ToTensor()(bicubic_hr_image), transforms.ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)

# 모델 정의
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)  # Batch Normalization 추가
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)  # Batch Normalization 추가

    def forward(self, x):
        residual = x
        x = self.bn1(self.conv1(x))  # Batch Normalization 적용
        x = self.prelu(x)
        x = self.bn2(self.conv2(x))  # Batch Normalization 적용
        return x + residual  # Residual 연결

class Generator(nn.Module):
    def __init__(self, scale_factor, num_residual_blocks=16):
        super(Generator, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),  # 입력 채널 3 (RGB 이미지), 출력 채널 64
            nn.PReLU()
        )

        # Residual 블록들
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )

        # 마지막 레이어를 RGB로 변환
        self.block8 = nn.Sequential(
            nn.Conv2d(64, 64 * (scale_factor ** 2), kernel_size=3, padding=1),  # 64 * (scale_factor ** 2) 채널
            nn.PixelShuffle(scale_factor),  # upscale_factor가 4인 경우 입력 채널 수는 64 * 4 = 256이 되어야 함
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)  # 마지막 Conv 레이어에서 RGB로 변환
        )

    def forward(self, x):
        x = self.block1(x)
        residual = self.residual_blocks(x)
        x = x + residual  # Residual 연결
        x = self.block8(x)
        return x

# 손실 함수 정의
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:30].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.loss_net = vgg
        self.mse_loss = nn.MSELoss()

    def forward(self, netD_out, fake_img, real_img):
        content_loss = self.mse_loss(self.loss_net(fake_img), self.loss_net(real_img))
        adversarial_loss = torch.mean(1 - netD_out)
        return content_loss + adversarial_loss

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    # 모델 초기화 및 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")  # 사용 중인 디바이스 출력
    generator = Generator(upscale_factor).to(device)
    discriminator = Discriminator().to(device)
    generator_loss = GeneratorLoss().to(device)

    # 옵티마이저 설정
    optimizer_g = optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4)

    # 데이터 로더 설정
    train_dataset = TrainDataset(dataset_dir, crop_size, upscale_factor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = ValDataset(val_dataset_dir, crop_size, upscale_factor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 훈련 루프
    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        for lr_images, hr_images in tqdm(train_loader):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            # 생성자 훈련
            optimizer_g.zero_grad()
            fake_images = generator(lr_images)
            netD_out = discriminator(fake_images)
            loss_g = generator_loss(netD_out, fake_images, hr_images)
            loss_g.backward()
            optimizer_g.step()

            # 판별자 훈련
            
            optimizer_d.zero_grad()
            real_out = discriminator(hr_images)
            fake_out = discriminator(fake_images.detach())
            loss_d = -torch.mean(real_out) + torch.mean(fake_out)
            loss_d.backward()
            optimizer_d.step()

        # 에포크마다 손실 출력
        print(f"Epoch [{epoch + 1}/{epochs}] - Generator Loss: {loss_g.item():.4f}, Discriminator Loss: {loss_d.item():.4f}")

        # 모델 저장
        if (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), os.path.join(save_path, f'generator_epoch_{epoch + 1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(save_path, f'discriminator_epoch_{epoch + 1}.pth'))

    # 최종 모델 저장
    torch.save(generator.state_dict(), os.path.join(save_path, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(save_path, 'discriminator_final.pth'))