import os
import torch
import random
import logging
import warnings
from torch.utils.data import DataLoader
from torchvision import transforms as T

from SegmentationDataset import SegmentationDataset, ROOT
from eval import cal_score

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

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(3407)
random.seed(3407)

from tif2pngs import Tif2Pngs, init_data_dir, split_train_val, ROOT
import os

# 阶段一，处理原始 tif 文件，并划分训练集和验证集（只用执行一遍）
# init_data_dir()
# mask_file_path = os.path.join(ROOT, 'datasets', 'standard.tif')
# image_file_path = os.path.join(ROOT, 'datasets', 'main', 'result.tif')
# tif2pngs = Tif2Pngs(image_file_path,
#                     os.path.join(ROOT, 'data', 'train', 'images'),
#                     block_size=512,
#                     stride=512)
# tif2pngs.process_tif()
# tif2pngs = Tif2Pngs(mask_file_path,
#                     os.path.join(ROOT, 'data', 'train', 'masks'),
#                     block_size=512,
#                     stride=512)
# tif2pngs.process_tif()
# split_train_val()
print("Done")

# 阶段二，加载数据集
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
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  # 注意这里已经使用1-dice
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


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


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=2):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky


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
        score = cal_score(model, val_loader)

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
    with open('grid_search_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Loss Function', 'Optimizer', 'Encoder Name', 'Learning Rate', 'Max Score',
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
                                model = model_class(encoder_name=encoder_name, classes=3,
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
