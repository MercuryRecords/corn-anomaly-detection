import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

# 获取当前文件的路径
current_file_path = os.path.abspath(__file__)
# 获取父目录 code 的路径
parent_directory_path = os.path.dirname(current_file_path)
# 获取项目的根目录
ROOT = os.path.dirname(parent_directory_path)

class SegmentationDataset(Dataset):
    def __init__(self,
                 root_dir,
                 transform=None,
                 image_prefix='result',
                 mask_prefix='standard',
                 ):
        self.root_dir = root_dir
        self.transform = transform
        self.image_prefix = image_prefix
        self.mask_prefix = mask_prefix
        self.images = os.listdir(os.path.join(root_dir, 'images'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, 'images', img_name)
        mask_path = os.path.join(self.root_dir, 'masks', img_name.replace(self.image_prefix, self.mask_prefix))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # 加载mask并转换为numpy数组
        mask = Image.open(mask_path).convert("L")  # 确保mask是灰度图像
        mask = np.array(mask)

        # 创建三个二进制掩码，每个类别一个
        mask_0 = (mask == 0).astype(np.uint8)  # 类别0的掩码
        mask_1 = (mask == 1).astype(np.uint8)  # 类别1的掩码
        mask_2 = (mask == 2).astype(np.uint8)  # 类别2的掩码

        # 将三个掩码堆叠成一个新的数组，形状为(3, 256, 256)
        mask = np.stack([mask_0, mask_1, mask_2], axis=0)

        # 将numpy数组转换为torch tensor
        mask = torch.tensor(mask, dtype=torch.float32)

        # 对图像应用相同的转换
        if self.transform:
            image = self.transform(image)

        return image, mask



if __name__ == '__main__':
    train_dir = os.path.join(ROOT, 'data', 'train')
    val_dir = os.path.join(ROOT, 'data', 'val')

    # 定义变换
    transform = T.Compose([
        T.ToTensor(),
    ])

    # 创建数据集
    train_dataset = SegmentationDataset(root_dir=train_dir, transform=transform)
    val_dataset = SegmentationDataset(root_dir=val_dir, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
