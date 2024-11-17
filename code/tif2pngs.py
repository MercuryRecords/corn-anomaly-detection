import os
import random
import shutil

import rasterio
import numpy as np
from PIL import Image
from tqdm import tqdm
from shutil import copyfile

# 获取当前文件的路径
current_file_path = os.path.abspath(__file__)
# 获取父目录 code 的路径
parent_directory_path = os.path.dirname(current_file_path)
# 获取项目的根目录
ROOT = os.path.dirname(parent_directory_path)

random.seed(3407)


def init_data_dir():
    # 定义源目录和目标目录
    target_dir = os.path.join(ROOT, 'data')

    # 创建目标目录结构
    sub_dirs = ['train', 'val']
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(target_dir, sub_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, sub_dir, 'masks'), exist_ok=True)


def split_train_val(val_ratio=0.2, image_prefix='result', mask_prefix='standard'):
    source_dir = os.path.join(ROOT, 'data', 'train')
    target_dir = os.path.join(ROOT, 'data', 'val')

    files = os.listdir(os.path.join(source_dir, 'images'))

    num_files = len(files)
    num_val_files = int(num_files * val_ratio)
    val_files = random.sample(files, num_val_files)
    # 将抽取的文件复制到目标目录
    for file in tqdm(val_files):
        # 复制图像文件
        copyfile(os.path.join(source_dir, 'images', file), os.path.join(target_dir, 'images', file))
        # 复制掩码文件
        copyfile(os.path.join(source_dir, 'masks', file.replace(image_prefix, mask_prefix)),
                 os.path.join(target_dir, 'masks', file.replace(image_prefix, mask_prefix)))
        # 删除源目录下的文件
        os.remove(os.path.join(source_dir, 'images', file))
        os.remove(os.path.join(source_dir, 'masks', file.replace(image_prefix, mask_prefix)))


class Tif2Pngs:
    def __init__(self, tif_path, output_dir, block_size=256, stride=256, inference=False):
        self.tif_path = tif_path
        self.output_dir = output_dir
        self.block_size = block_size
        self.stride = stride
        self.inference = inference
        if self.inference:
            shutil.rmtree(self.output_dir, ignore_errors=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def process_tif(self):
        with rasterio.open(self.tif_path) as src:
            data = src.read()  # 读取数据
            meta = src.meta  # 获取元数据

        i_len = data.shape[1] // self.stride + (1 if data.shape[1] % self.stride != 0 else 0)
        j_len = data.shape[2] // self.stride + (1 if data.shape[2] % self.stride != 0 else 0)

        # 遍历图像数据，生成图像块
        with tqdm(total=i_len * j_len) as pbar:
            for i in range(0, data.shape[1], self.stride):
                for j in range(0, data.shape[2], self.stride):
                    # 计算图像块的实际大小，确保不会超出边界
                    i_end = min(i + self.block_size, data.shape[1])
                    j_end = min(j + self.block_size, data.shape[2])

                    # 切割图像块
                    block = data[:, i:i_end, j:j_end]

                    # 如果块大小小于block_size，我们需要填充
                    if block.shape[1:] != (self.block_size, self.block_size):
                        pad_height = self.block_size - block.shape[1]
                        pad_width = self.block_size - block.shape[2]
                        block = np.pad(block, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant')

                    # 将 block 从 (1/4, 256, 256) 转换为 (256, 256, 1/4)
                    block = np.transpose(block, (1, 2, 0))

                    if block.shape[2] == 1:
                        block = block.squeeze()
                        block = block * 127

                    # 将numpy数组转换为PIL图像
                    block_image = Image.fromarray(block)

                    # 确定文件名
                    file_name = f"{os.path.splitext(os.path.basename(self.tif_path))[0]}_{i}_{j}.png"

                    # 保存图像块
                    block_image.save(os.path.join(self.output_dir, file_name))

                    pbar.update(1)


if __name__ == '__main__':
    init_data_dir()
    mask_file_path = os.path.join(ROOT, 'datasets', 'standard.tif')
    image_file_path = os.path.join(ROOT, 'datasets', 'main', 'result.tif')
    tif2pngs = Tif2Pngs(image_file_path, os.path.join(ROOT, 'data', 'train', 'images'))
    tif2pngs.process_tif()
    tif2pngs = Tif2Pngs(mask_file_path, os.path.join(ROOT, 'data', 'train', 'masks'))
    tif2pngs.process_tif()
    split_train_val()
    print("Done")
