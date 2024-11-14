import os
from os.path import basename

import numpy as np
import torch
from PIL import Image
from torch import unique
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

# 获取当前文件的路径
current_file_path = os.path.abspath(__file__)
# 获取父目录 code 的路径
parent_directory_path = os.path.dirname(current_file_path)
# 获取项目的根目录
ROOT = os.path.dirname(parent_directory_path)

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(3407)

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
        # mask = Image.open(mask_path)

        # 加载mask并转换为numpy数组
        # mask = Image.open(mask_path).convert("L")  # 确保mask是灰度图像
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # if np.unique(mask).size > 1:
        #     print(np.unique(mask))
        #     pass
        #
        # base_name = os.path.basename(mask_path)
        # if "9728_11520" in base_name:
        #     pass

        # 创建三个二进制掩码，每个类别一个
        mask_0 = (mask == 0).astype(np.uint8)  # 类别0的掩码
        mask_1 = (mask == 127).astype(np.uint8)  # 类别1的掩码
        mask_2 = (mask == 254).astype(np.uint8)  # 类别2的掩码

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

    from matplotlib import pyplot as plt
    import numpy as np

    one_hot_vector1 = torch.tensor([1., 0., 0.]).repeat(256, 256, 1)
    one_hot_vector2 = torch.tensor([0., 1., 0.]).repeat(256, 256, 1)
    one_hot_vector3 = torch.tensor([0., 0., 1.]).repeat(256, 256, 1)

    i = 0
    while i < len(train_dataset):
        image, mask = train_dataset[i]
        mask = mask.permute(1, 2, 0)
        comparison1 = mask == one_hot_vector1
        comparison2 = mask == one_hot_vector2
        comparison3 = mask == one_hot_vector3
        judge1 = comparison1.all(dim=2).any().item()
        judge2 = comparison2.all(dim=2).any().item()
        judge3 = comparison3.all(dim=2).any().item()
        if judge1 and judge2 and judge3:
            print(f"第{i}张图片包含所有类别")
            break
        # uni = np.unique(mask)
        i += 1

    # 显示图像和掩码
    image = image.permute(1, 2, 0)
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title('Mask')
    plt.show()