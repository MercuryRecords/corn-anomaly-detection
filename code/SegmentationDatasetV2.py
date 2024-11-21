import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# 获取当前文件的路径
current_file_path = os.path.abspath(__file__)
# 获取父目录 code 的路径
parent_directory_path = os.path.dirname(current_file_path)
# 获取项目的根目录
ROOT = os.path.dirname(parent_directory_path)

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(3407)


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


if __name__ == '__main__':
    pass