# encoding: utf-8
import os
import random
import re

import numpy as np
import rasterio
import torch
from PIL import Image
from segmentation_models_pytorch import Unet
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from tif2pngs import Tif2Pngs

# 当前文件处于 /data/code/infer 目录下
# 获取当前文件的路径
current_file_path = os.path.abspath(__file__)
# 获取父目录 infer 的路径
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

infer_dir = os.path.join(ROOT, 'user_data', 'dataV0', 'infer')
result_file_path = os.path.join(ROOT, 'raw_data', '测试集文件', 'result.tif')
# tif2pngs = Tif2Pngs(result_file_path, os.path.join(infer_dir, 'images'), inference=True, stride=64)
# tif2pngs.process_tif()

model_path = os.path.join(ROOT, 'user_data', 'models', 'model_3c_no_aug.pth')
model_config = {
    'model': Unet,
    'encoder_name': 'resnet34',
    'classes': 3,
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model_config['model'](
    encoder_name=model_config['encoder_name'],
    classes=model_config['classes'],
).to(device)

try:
    model.load_state_dict(torch.load(model_path, weights_only=True))
except Exception as e:
    model = nn.DataParallel(model)  # TODO: 推理模型是在服务器上训练的故有这行代码
    model.load_state_dict(torch.load(model_path, weights_only=True))

model.eval()

images_dir = infer_dir

transform = T.Compose([
    T.ToTensor(),
])
infer_dataset = SegmentationDataset(root_dir=images_dir, transform=transform, inference=True)

images = infer_dataset.get_images()
max_x = 0
max_y = 0

for image in images:
    match = re.search(r'_(\d+)_(\d+)\.png', image)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y

# 新建一个 ndarray
array = np.zeros((3, max_x + 256, max_y + 256), dtype=np.float16)

# 推理
print('开始推理')

with torch.no_grad():
    for input_image, image_name in tqdm(infer_dataset):
        tmp = input_image.unsqueeze(0).to(device)
        output = model(tmp)
        result = output.squeeze().cpu().numpy()
        match = re.search(r'_(\d+)_(\d+)', image_name)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            array[:, x:x + 256, y:y + 256] += result
        else:
            print(f"Error: {image_name} does not match the pattern")

del model

array = np.argmax(array, axis=0)

with rasterio.open(result_file_path) as src:
    meta = src.meta.copy()
    print(meta)
    pruned_shape = (meta['height'], meta['width'])

meta.update({
    'count': 1,
})

# 将 array 裁剪到与 result.tif 相同的大小
array = array[:pruned_shape[0], :pruned_shape[1]]

output_path = os.path.join(ROOT, 'user_data', 'submission_1.tif')

with rasterio.open(output_path, 'w', **meta) as dst:
    dst.write(array, 1)