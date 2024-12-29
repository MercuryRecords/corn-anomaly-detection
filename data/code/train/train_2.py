# encoding: utf-8
import os
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from albumentations import Compose, HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomResizedCrop
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import Unet
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from tif2pngs import Tif2Pngs

warnings.filterwarnings('ignore')

# 当前文件处于 /data/code/train 目录下
# 获取当前文件的路径
current_file_path = os.path.abspath(__file__)
# 获取父目录 train 的路径
train_directory_path = os.path.dirname(current_file_path)
# 获取 code 目录
code_directory_path = os.path.dirname(train_directory_path)
# 获取根目录
ROOT = os.path.dirname(code_directory_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(3407)
random.seed(3407)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss

class SegmentationDataset(Dataset):
    def __init__(self,
                 root_dir,
                 transform=None,
                 image_prefix='result',
                 mask_prefix='standard',
                 inference=False,
                 ):
        self.root_dir = root_dir
        self.transform = transform
        self.image_prefix = image_prefix
        self.mask_prefix = mask_prefix
        self.images = os.listdir(os.path.join(root_dir, 'images'))
        self.inference = inference

    def __len__(self):
        return len(self.images)

    def get_images(self):
        return self.images

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, 'images', img_name)
        image = Image.open(img_path).convert("RGB")

        # 对图像应用相同的转换
        if self.transform:
            image = self.transform(image)

        if self.inference:
            return image, img_name

        mask_path = os.path.join(self.root_dir, 'masks', img_name.replace(self.image_prefix, self.mask_prefix))
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # 创建三个二进制掩码，每个类别一个
        mask_0 = (mask == 0).astype(np.uint8)  # 类别0的掩码
        mask_1 = (mask == 127).astype(np.uint8)  # 类别1的掩码
        mask_2 = (mask == 254).astype(np.uint8)  # 类别2的掩码

        # 将三个掩码堆叠成一个新的数组，形状为(3, 256, 256)
        mask = np.stack([mask_0, mask_1, mask_2], axis=0)

        # 将numpy数组转换为torch tensor
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask


def init_data_dir(target_dir):
    # 创建目标目录结构
    sub_dirs = ['train']
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(target_dir, sub_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, sub_dir, 'masks'), exist_ok=True)

# 数据增强函数
def augment_images(image_path, mask_path, save_dir, transform):
    # 加载图像和掩码
    image = np.array(Image.open(image_path))  # (H, W, C)
    mask = np.array(Image.open(mask_path))  # 掩码保持单通道格式

    # 筛选掩码
    unique_classes = np.unique(mask)
    if len(unique_classes) < 2 or np.all(mask == 0):
        return

    # 应用数据增强
    augmented = transform(image=image, mask=mask)
    transformed_image = augmented['image']  # 增强后的图像
    transformed_mask = augmented['mask']    # 增强后的掩码

    # 保存增强后的图像和掩码
    save_image_path = os.path.join(save_dir, 'images', os.path.basename(image_path).replace('.png', '_aug.png'))
    save_mask_path = os.path.join(save_dir, 'masks', os.path.basename(mask_path).replace('.png', '_aug.png'))

    # 保存图像为 .png 格式
    Image.fromarray(transformed_image.permute(1, 2, 0).numpy()).save(save_image_path)
    # 保存掩码为 .png 格式
    Image.fromarray(transformed_mask.numpy()).save(save_mask_path)

# 数据增强目录和逻辑
def augment_dataset(source_dir, target_dir, transform):
    os.makedirs(os.path.join(target_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'masks'), exist_ok=True)

    image_files = os.listdir(os.path.join(source_dir, 'images'))
    for image_file in tqdm(image_files, desc="Augmenting dataset"):
        if image_file.endswith('_aug.png'):
            continue
        image_path = os.path.join(source_dir, 'images', image_file)
        mask_path = os.path.join(source_dir, 'masks', image_file.replace('result', 'standard'))

        # 增强数据并保存到目标目录
        augment_images(image_path, mask_path, target_dir, transform)

# 阶段一：处理原始 tif 文件

init_data_dir(os.path.join(ROOT, 'user_data', 'dataV1'))
mask_file_path = os.path.join(ROOT, 'raw_data', '训练集文件', 'standard.tif')
image_file_path = os.path.join(ROOT, 'raw_data', '训练集文件', 'result.tif')
tif2pngs = Tif2Pngs(image_file_path, os.path.join(ROOT, 'user_data', 'dataV1', 'train', 'images'), stride=128) # TODO: 修改步长
tif2pngs.process_tif()
tif2pngs = Tif2Pngs(mask_file_path, os.path.join(ROOT, 'user_data', 'dataV1', 'train', 'masks'), stride=128)
tif2pngs.process_tif()

blocks_dir = os.path.join(ROOT, 'user_data', 'dataV1', 'train', 'images')

all_black_files = []

for filename in os.listdir(blocks_dir):
    if filename.endswith('.png'):
        file_path = os.path.join(blocks_dir, filename)
        block = np.array(Image.open(file_path))

        if np.all(block == 0):
            all_black_files.append(filename)
            # break

np.random.seed(42)
print(f'Black block found count: {len(all_black_files)}')

files_to_keep = np.random.choice(all_black_files, size=100, replace=False)

for filename in all_black_files:
    if filename not in files_to_keep:
        file_path = os.path.join(blocks_dir, filename)
        os.remove(file_path)

# 数据增强方法
transform = Compose([
    HorizontalFlip(p=0.5),  # 随机水平翻转
    VerticalFlip(p=0.5),  # 随机垂直翻转
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),  # 随机仿射变换
    RandomResizedCrop(height=256, width=256, scale=(0.6, 1.0), p=0.5),  # 随机裁剪和调整大小
    # OneOf([
    #     MotionBlur(p=0.2),
    #     MedianBlur(blur_limit=3, p=0.1),
    #     Blur(blur_limit=3, p=0.1),
    # ], p=0.5),
    ToTensorV2()
])

# 调用增强函数
train_source_dir = os.path.join(ROOT, 'user_data', 'dataV1', 'train')
train_target_dir = os.path.join(ROOT, 'user_data', 'dataV1', 'train')

augment_dataset(train_source_dir, train_target_dir, transform)
print("数据增强完成")

# 阶段二：加载数据集

# 定义变换
transform = T.Compose([
    T.ToTensor(),
])

# 设置路径
train_dir = os.path.join(ROOT, 'user_data', 'dataV1', 'train')

# 创建数据集
train_dataset = SegmentationDataset(root_dir=train_dir, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 阶段三：训练模型

# 创建权重路径
os.makedirs(os.path.join(ROOT, 'user_data', 'weightsV1'), exist_ok=True)

# 创建模型
model = Unet(encoder_name='resnet34',
             encoder_weights='imagenet',
             classes=3,
             activation='softmax'
             ).to(device)

# 损失函数和优化器
loss_fn = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 初始化最小loss为正无穷大
min_loss = float('inf')

# 设置训练轮次
epoch_num = 50

# 训练模型
for epoch in tqdm(range(1, epoch_num + 1)):
    model.train()
    for batch in train_loader:
        images, masks = batch
        # 将数据和模型都移动到 GPU
        images = images.to(device)
        masks = masks.to(device)

        # 现在 masks 是一个一维的 tensor，每个元素对应一个像素的类别索引
        outputs = model(images)
        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每个epoch结束时保存模型
    torch.save(model.state_dict(), os.path.join(ROOT, 'user_data', 'weightsV1', f'best_model_epoch_{epoch}.pth'))
