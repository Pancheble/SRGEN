import os
import shutil
import random
from PIL import Image



# 원본 이미지 경로 (VOC2012의 모든 이미지가 있는 폴더)
dataset_dir = r'C:\SR\SRGAN\data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'

# 학습용, 검증용 이미지가 저장될 폴더 경로 설정
train_dataset_dir = r'C:\SR\SRGAN\data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Train_Images'
val_dataset_dir = r'C:\SR\SRGAN\data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Val_Images'

all_images = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

valid_images = []

for filename in all_images:
    img_path = os.path.join(dataset_dir, filename)
    with Image.open(img_path) as img:
        width, height = img.size
        if width > 96 and height > 96:
            valid_images.append(filename)

val_images = random.sample(valid_images, 300) if len(valid_images) >= 300 else valid_images

# 검증용 이미지 복사
for filename in val_images:
    src_path = os.path.join(dataset_dir, filename)
    dst_path = os.path.join(val_dataset_dir, filename)
    shutil.copyfile(src_path, dst_path)

# 학습용 이미지 (검증용 이미지를 제외한 나머지)
train_images = [f for f in valid_images if f not in val_images]

# 학습용 이미지 복사
for filename in train_images:
    src_path = os.path.join(dataset_dir, filename)
    dst_path = os.path.join(train_dataset_dir, filename)
    shutil.copyfile(src_path, dst_path)
    
print(f"Train_Images: {len(train_images)}")
print(f"Val_Images: {len(val_images)}")