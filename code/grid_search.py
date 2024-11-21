import os
import torch
import random
import logging
import warnings
from torch.utils.data import DataLoader
from torchvision import transforms as T

from SegmentationDataset import SegmentationDataset, ROOT
from loss import FocalLoss
from eval import cal_score, cal_score_v2

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

# 创建一个handler，用于写入日志文件
file_handler = logging.FileHandler('my_log.log')
file_handler.setLevel(logging.INFO)

# 创建一个handler，用于输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加handler到logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

import torch
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

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

import os
from SegmentationDataset import ROOT
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch
import random

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(3407)
random.seed(3407)

# 阶段二，加载数据集
train_dir = os.path.join(ROOT, 'dataV2', 'train')
val_dir = os.path.join(ROOT, 'dataV2', 'val')

# 定义变换
transform = T.Compose([
    T.ToTensor(),
])

# 创建数据集
train_dataset = SegmentationDatasetV2(root_dir=train_dir)
val_dataset = SegmentationDatasetV2(root_dir=val_dir)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



def train(modelName, model, loss_fn, optimizer, epoch_num):
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    os.makedirs('../model', exist_ok=True)

    # # 初始化最小loss为正无穷大
    # min_loss = float('inf')

    # 初始化分数为 0.0
    max_score = 0.0

    # 训练模型
    for epoch in range(1, epoch_num + 1):
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

        # 计算分数
        score = cal_score_v2(model, val_loader)

        logger.info(f"Epoch {epoch}/{epoch_num}, Loss: {loss.item()}, Score: {score}")

        if max_score < score:
            max_score = score

    torch.save(model, f'../model/{modelName}_{int(time.time())}.pth')

    return max_score


import torch
from segmentation_models_pytorch import DeepLabV3Plus, UnetPlusPlus, Unet
import csv
import time
import torch.optim as optim

# Define the available models
models = {
    'Unet': Unet,
    # 'UnetPlusPlus': UnetPlusPlus,
    # 'DeepLabV3Plus': DeepLabV3Plus,
}

# Define the available loss functions
loss_functions = {
    "FocalLoss": FocalLoss(),
    # "FocalTverskyLoss": FocalTverskyLoss(),

}

# Define the available optimizers
optimizers = {
    # "Adam": optim.Adam,
    # "SGD": optim.SGD,
    "AdamW": optim.AdamW,
}

# Define learning rates to test
learning_rates = [
    # 1e-5,
    1e-4,
    # 1e-3,
]

epoch_num = 10

encoder_names = [
    # 'resnet34',
    'resnet50',
    # 'densenet121',
                 ]

activation_names = [
    'sigmoid',
    'softmax',
    'logsoftmax',
    'tanh',
    'identity',
    None,
                 ]


# Create a function to perform grid search
def grid_search(epoch_num):
    best_combination = None
    best_score = 0.0

    # Open CSV file once outside the loop
    with open('grid_search_results_v2.csv', 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Loss Function', 'Optimizer', 'Encoder Name', 'Activation Function', 'Learning Rate', 'Max Score',
                      'Epoch Average Training Time (s)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()  # Write header once

        for model_name, model_class in models.items():
            for loss_name, loss_fn_class in loss_functions.items():
                for optimizer_name, optimizer_class in optimizers.items():
                    for encoder_name in encoder_names:
                        for activation_name in activation_names:
                            for lr in learning_rates:
                                print("---------------------------------Start---------------------------------")

                                tmp_str = f"Training {model_name} with {loss_name}, {encoder_name}, {optimizer_name}, {activation_name}, lr={lr}"

                                print(tmp_str)

                                logger.info(tmp_str)

                                # Initialize the model, loss function, and optimizer
                                model = model_class(encoder_name=encoder_name, classes=3, in_channels=5,
                                                    activation=activation_name)  # Make sure this is the correct parameter for your model
                                loss_fn = loss_fn_class
                                optimizer = optimizer_class(model.parameters(), lr=lr)

                                # Train the model
                                start_time = time.time()
                                max_score = train(model_name, model, loss_fn, optimizer,
                                                  epoch_num)  # Ensure the train function works
                                elapsed_time = (time.time() - start_time) / epoch_num

                                # Check if the current model has the best performance
                                if max_score > best_score:
                                    best_score = max_score
                                    best_combination = (model_name, loss_name, optimizer_name, encoder_name, lr)

                                print(f"Epoch time: {elapsed_time:.2f} seconds, Max score: {max_score:.4f}")

                                # Record the results
                                result = {
                                    'Model': model_name,
                                    'Loss Function': loss_name,
                                    'Optimizer': optimizer_name,
                                    'Encoder Name': encoder_name,
                                    'Activation Function': activation_name,
                                    'Learning Rate': lr,
                                    'Max Score': max_score,
                                    'Epoch Average Training Time (s)': elapsed_time
                                }
                                writer.writerow(result)  # 写入当前结果
                                csvfile.flush()  # 确保数据被写入文件

                                print(f"Best model combination: {best_combination} with max score: {best_score:.4f}")

                                print("----------------------------------End----------------------------------")

        return best_combination, best_score


# Call grid_search with appropriate epoch_num
best_combination, best_loss = grid_search(epoch_num=epoch_num)
