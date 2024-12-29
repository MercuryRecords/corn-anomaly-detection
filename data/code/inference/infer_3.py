# encoding: utf-8
import os
import re
import torch
import random
import warnings
import rasterio
import numpy as np
from torch import nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from combine_channels import combine
from segmentation_models_pytorch import Unet

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

combine(
    root_path=ROOT,
    reference_tif=os.path.join(ROOT, 'raw_data', '测试集文件', 'result.tif'),
    dsm_tif=os.path.join(ROOT, 'raw_data', '测试集文件', 'dsm.tif'),
    nir_tif=os.path.join(ROOT, 'raw_data', '测试集文件', 'result_NIR.tif'),
    red_tif=os.path.join(ROOT, 'raw_data', '测试集文件', 'result_RED.tif'),
    output_tif=os.path.join(ROOT, 'user_data', 'dataV2', 'output.tif')
)

combined_file_path = os.path.join(ROOT, 'user_data', 'dataV2', 'output.tif')

os.makedirs(os.path.join(ROOT, 'user_data', 'dataV2', 'infer', 'blocks'), exist_ok=True)

with rasterio.open(combined_file_path) as src:
    # print(src.meta)
    width = src.meta['width']
    height = src.meta['height']
    channels = src.meta['count']
    # 分块保存ndarray，每块大小为 256 * 256
    block_size = 256
    for i in tqdm(range(0, height, block_size // 2)): # TODO: 修改步长
        for j in range(0, width, block_size // 2):
            block = src.read(window=(
                (i, min(i + block_size, height)),
                (j, min(j + block_size, width)))
            )

            if block.shape[1:] != (block_size, block_size):
                pad_height = block_size - block.shape[1]
                pad_width = block_size - block.shape[2]
                block = np.pad(block, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant')

            # 保存 block 数组
            np.save(os.path.join(ROOT, 'user_data', 'dataV2', 'infer', 'blocks', f'block_{i}_{j}.npy'), block)


images_dir = os.path.join(ROOT, 'user_data', 'dataV2', 'infer')
infer_dataset = SegmentationDatasetV2(root_dir=images_dir, inference=True)

images = infer_dataset.get_arrays()
max_x = 0
max_y = 0

for image in images:
    match = re.search(r'_(\d+)_(\d+)\.npy', image)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y

# 新建一个 ndarray
array = np.zeros((3, max_x + 256, max_y + 256), dtype=np.float16)

model_path = os.path.join(ROOT, 'user_data', 'models', 'model_5c_aug.pth')
model_config = {
    'model': Unet,
    'encoder_name': 'resnet34',
    'classes': 3,
    'channels': 5,
    'activation': 'softmax',
}

model = model_config['model'](
    encoder_name=model_config['encoder_name'],
    classes=model_config['classes'],
    in_channels=model_config['channels'],
    activation=model_config['activation'],
).to(device)

try:
    model.load_state_dict(torch.load(model_path, weights_only=True))
except Exception as e:
    model=nn.DataParallel(model) # TODO: 推理模型是在服务器上训练的故有这行代码
    model.load_state_dict(torch.load(model_path, weights_only=True))

model.eval()

with torch.no_grad():
    with tqdm(total=len(images)) as pbar:
        for input_image, image_name in infer_dataset:
            tmp = torch.tensor(input_image).unsqueeze(0).to(device)
            output = model(tmp)
            result = output.squeeze().cpu().numpy()
            match = re.search(r'_(\d+)_(\d+)', image_name)
            if match:
                x = int(match.group(1))
                y = int(match.group(2))
                array[:, x:x + 256, y:y + 256] += result
            else:
                print(f"Error: {image_name} does not match the pattern")

            pbar.update(1)

array = np.argmax(array, axis=0)

result_file_path = os.path.join(ROOT, 'raw_data', '测试集文件', 'result.tif')
with rasterio.open(result_file_path) as src:
    meta = src.meta.copy()
    meta.update({
        'count': 1,
    })
    pruned_shape = (meta['height'], meta['width'])

array = array[:pruned_shape[0], :pruned_shape[1]]

output_path = os.path.join(ROOT, 'user_data', 'submission_3.tif')

with rasterio.open(output_path, 'w', **meta) as dst:
    dst.write(array, 1)