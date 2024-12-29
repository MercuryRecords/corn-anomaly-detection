import os
import shutil

import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm


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