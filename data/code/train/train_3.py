# encoding: utf-8
import os
import random
import warnings

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from albumentations import Compose, HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomResizedCrop
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import Unet
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from combine_channels import combine
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

class SegmentationDatasetV2(Dataset):
    def __init__(self,
                 root_dir,
                 transform=None,
                 array_prefix='block',
                 mask_prefix='standard',
                 array_suffix='.npy',
                 mask_suffix='.png',
                 inference=False,
                 ):
        self.root_dir = root_dir
        self.transform = transform
        self.array_prefix = array_prefix
        self.mask_prefix = mask_prefix
        self.array_suffix = array_suffix
        self.mask_suffix = mask_suffix
        self.arrays = os.listdir(os.path.join(root_dir, 'blocks'))
        self.inference = inference

    def __len__(self):
        return len(self.arrays)

    def get_arrays(self):
        return self.arrays

    def __getitem__(self, idx):
        array_name = self.arrays[idx]
        array_path = os.path.join(self.root_dir, 'blocks', array_name)
        array = np.load(array_path)

        if self.inference:
            return array, array_name

        mask_path = os.path.join(self.root_dir, 'masks',
                                 array_name.replace(self.array_prefix, self.mask_prefix).replace(self.array_suffix,
                                                                                                 self.mask_suffix))
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

        return array, mask

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

# 数据增强函数
def augment_npy_images(image_path, mask_path, save_dir, transform):
    # 加载 .npy 格式的图像和 .png 格式的掩码
    image = np.load(image_path)  # 加载 block 数据，形状 (C, H, W)
    image = np.moveaxis(image, 0, -1)  # 转换为 (H, W, C)
    mask = np.array(Image.open(mask_path))  # 掩码保持单通道格式

    # 筛选掩码：仅处理包含两种及以上类别的掩码
    unique_classes = np.unique(mask)
    if len(unique_classes) < 2:
        return

    # 应用数据增强
    augmented = transform(image=image, mask=mask)
    transformed_image = augmented['image']  # 增强后的图像
    transformed_mask = augmented['mask']  # 增强后的掩码

    # 保存增强后的图像和掩码
    save_image_path = os.path.join(save_dir, 'blocks', os.path.basename(image_path).replace('.npy', '_aug.npy'))
    save_mask_path = os.path.join(save_dir, 'masks', os.path.basename(mask_path).replace('.png', '_aug.png'))

    # 保存图像为 .npy 格式
    np.save(save_image_path, transformed_image.numpy())
    # 保存掩码为 .png 格式
    Image.fromarray(transformed_mask.numpy()).save(save_mask_path)

# 数据增强目录和逻辑
def augment_dataset(source_dir, target_dir, transform):
    os.makedirs(os.path.join(target_dir, 'blocks'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'masks'), exist_ok=True)

    image_files = os.listdir(os.path.join(source_dir, 'blocks'))
    for image_file in tqdm(image_files, desc="Augmenting dataset"):
        if image_file.endswith('_aug.npy'):
            continue
        image_path = os.path.join(source_dir, 'blocks', image_file)
        mask_path = os.path.join(source_dir, 'masks',
                                 image_file.replace('block', 'standard').replace('.npy', '.png'))

        # 增强数据并保存到目标目录
        augment_npy_images(image_path, mask_path, target_dir, transform)

os.makedirs(os.path.join(ROOT, 'user_data', 'dataV2'), exist_ok=True)

combine(
    root_path=ROOT,
    reference_tif=os.path.join(ROOT, 'raw_data', '训练集文件', 'result.tif'),
    dsm_tif=os.path.join(ROOT, 'raw_data', '训练集文件', 'dsm.tif'),
    nir_tif=os.path.join(ROOT, 'raw_data', '训练集文件', 'result_NIR.tif'),
    red_tif=os.path.join(ROOT, 'raw_data', '训练集文件', 'result_RED.tif'),
    output_tif=os.path.join(ROOT, 'user_data', 'dataV2', 'output.tif')
)

combined_file_path = os.path.join(ROOT, 'user_data', 'dataV2', 'output.tif')

os.makedirs(os.path.join(ROOT, 'user_data', 'dataV2', 'train', 'blocks'), exist_ok=True)

with rasterio.open(combined_file_path) as src:
    print(src.meta)
    width = src.meta['width']
    height = src.meta['height']
    channels = src.meta['count']
    # 分块保存ndarray，每块大小为 256 * 256
    block_size = 256
    for i in tqdm(range(0, height, block_size)): # TODO: 修改步长
        for j in range(0, width, block_size):
            block = src.read(window=(
                (i, min(i + block_size, height)),
                (j, min(j + block_size, width)))
            )

            if block.shape[1:] != (block_size, block_size):
                pad_height = block_size - block.shape[1]
                pad_width = block_size - block.shape[2]
                block = np.pad(block, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant')

            # 保存 block 数组
            np.save(os.path.join(ROOT, 'user_data', 'dataV2', 'train', 'blocks', f'block_{i}_{j}.npy'), block)




mask_file_path = os.path.join(ROOT, 'raw_data', '训练集文件', 'standard.tif')
tif2pngs = Tif2Pngs(mask_file_path, os.path.join(ROOT, 'user_data', 'dataV2', 'train', 'masks'), stride=256) # TODO: 修改步长
tif2pngs.process_tif()

# 数据增强方法
transform = Compose([
    HorizontalFlip(p=0.5),  # 随机水平翻转
    VerticalFlip(p=0.5),  # 随机垂直翻转
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),  # 随机仿射变换
    RandomResizedCrop(height=256, width=256, scale=(0.6, 1.0), p=0.5),  # 随机裁剪和调整大小
    ToTensorV2()
])

# 调用增强函数
train_source_dir = os.path.join(ROOT, 'user_data', 'dataV2', 'train')
train_target_dir = os.path.join(ROOT, 'user_data', 'dataV2', 'train')

augment_dataset(train_source_dir, train_target_dir, transform)
print("数据增强完成")


train_dir = os.path.join(ROOT, 'user_data', 'dataV2', 'train')
train_dataset = SegmentationDatasetV2(root_dir=train_dir)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 创建权重路径
os.makedirs(os.path.join(ROOT, 'user_data', 'weightsV2'), exist_ok=True)

# 模型参数
model_config = {
    'model': Unet,
    'encoder_name': 'resnet34',
    'classes': 3,
    'channels': 5,
    'activation': 'softmax',
}

# 创建模型
model = model_config['model'](
    encoder_name=model_config['encoder_name'],
    classes=model_config['classes'],
    in_channels=model_config['channels'],
    activation=model_config['activation'],
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
        # 将数据和模型都移动到GPU
        images = images.to(device)
        masks = masks.to(device)

        # 现在masks是一个一维的tensor，每个元素对应一个像素的类别索引
        outputs = model(images)
        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每个epoch结束时保存模型
    torch.save(model.state_dict(), os.path.join(ROOT, 'user_data', 'weightsV2', f'best_model_epoch_{epoch}.pth'))