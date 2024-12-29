# encoding: utf-8
import os
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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

# 阶段一：处理原始 tif 文件

init_data_dir(os.path.join(ROOT, 'user_data', 'dataV0'))
mask_file_path = os.path.join(ROOT, 'raw_data', '训练集文件', 'standard.tif')
image_file_path = os.path.join(ROOT, 'raw_data', '训练集文件', 'result.tif')
tif2pngs = Tif2Pngs(image_file_path, os.path.join(ROOT, 'user_data', 'dataV0', 'train', 'images'))
tif2pngs.process_tif()
tif2pngs = Tif2Pngs(mask_file_path, os.path.join(ROOT, 'user_data', 'dataV0', 'train', 'masks'))
tif2pngs.process_tif()

# 阶段二：加载数据集

# 定义变换
transform = T.Compose([
    T.ToTensor(),
])

# 设置路径
train_dir = os.path.join(ROOT, 'user_data', 'dataV0', 'train')

# 创建数据集
train_dataset = SegmentationDataset(root_dir=train_dir, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 阶段三：训练模型

# 创建权重路径
os.makedirs(os.path.join(ROOT, 'user_data', 'weightsV0'), exist_ok=True)

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
    torch.save(model.state_dict(), os.path.join(ROOT, 'user_data', 'weightsV0', f'best_model_epoch_{epoch}.pth'))
